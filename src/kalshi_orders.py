"""Order placement, polling, and cancellation on Kalshi.

Uses the **V2** order API (`/portfolio/events/orders`). The legacy
`/portfolio/orders` POST path is on a deprecation track — see
docs/kalshi/_llms.txt:83 — so new code should not use it. Listing orders is
still `GET /portfolio/orders`.

## The single-book model

V2 expresses everything in YES terms. There is no "buy NO" order:

    side="bid"  ->  buy YES  at `price`
    side="ask"  ->  sell YES at `price`  ==  buy NO at (1 - price)

So a signal to buy NO at $0.30 is submitted as an **ask at $0.70**. Getting this
backwards puts on exactly the opposite position, so the translation lives in one
place — `signal_to_order()` — rather than at each call site.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.kalshi_client import KalshiClient
from src.kalshi_signal import Signal
from src.utils import get_logger

logger = get_logger(__name__)

ORDERS_PATH = "portfolio/events/orders"      # V2: create / cancel
ORDERS_LIST_PATH = "portfolio/orders"        # listing is still on the old path

# Kalshi quotes some markets in deci-cents (e.g. 0.914), so don't round to whole
# cents. Six decimals is the API maximum; four covers every observed tick.
PRICE_DECIMALS = 4
COUNT_DECIMALS = 2


class LiveOrderBlocked(RuntimeError):
    """Raised when a production order is attempted without explicit opt-in."""


# A run of rejections with no success in between means the bot is not trading at
# all, and every individual rejection looks identical to a routine one. Between
# 2026-08-28 and 08-31 the weather bot logged 2,688 consecutive failures and
# placed zero orders; nothing in the log said "this is an outage", so it ran
# dead for three days. Escalating on a streak makes that one greppable line.
OUTAGE_AFTER_REJECTS = 25
_CONSECUTIVE_REJECTS = [0]


def _note_rejection(what: str, exc: Exception) -> None:
    _CONSECUTIVE_REJECTS[0] += 1
    streak = _CONSECUTIVE_REJECTS[0]
    logger.error(f"Order rejected ({what}): {exc!r}")
    if streak % OUTAGE_AFTER_REJECTS == 0:
        logger.critical(
            f"ORDER OUTAGE: {streak} consecutive rejections with no successful "
            f"order in between. The bot is not trading. Last error: {exc!r}"
        )


def _assert_orders_allowed(client: KalshiClient) -> None:
    """Refuse to send production orders unless the operator has opted in.

    Demo is fake money and always allowed. Production spends real money, so it
    requires `KALSHI_ALLOW_LIVE_ORDERS=true` — a deliberate, visible action that
    a stray `KALSHI_ENV` edit can't cause on its own.
    """
    if client.demo:
        return
    if os.getenv("KALSHI_ALLOW_LIVE_ORDERS", "").lower() != "true":
        raise LiveOrderBlocked(
            "Refusing to place a PRODUCTION order: KALSHI_ALLOW_LIVE_ORDERS is not "
            "'true'. This is real money. Set it in .env only when you intend to "
            "trade live."
        )


def signal_to_order(signal: Signal) -> tuple[str, float]:
    """Translate a Signal into `(book_side, yes_price)` for the V2 API.

    `signal.price` is what you pay for your chosen side. The book is quoted in
    YES terms, so a NO purchase becomes a YES sale at the complement.
    """
    if signal.side == "YES":
        return "bid", signal.price
    if signal.side == "NO":
        return "ask", round(1.0 - signal.price, PRICE_DECIMALS)
    raise ValueError(f"Unknown signal side: {signal.side!r}")


def place_limit_order(
    client: KalshiClient,
    ticker: str,
    side: str,
    price: float,
    count: float,
    *,
    time_in_force: str = "immediate_or_cancel",
    post_only: bool = False,
    client_order_id: str | None = None,
) -> dict[str, Any] | None:
    """Submit a limit order. Returns the order dict, or None on failure.

    Defaults to `immediate_or_cancel`: the order either trades against what's
    already resting or it dies. On a market that resolves in 15 minutes, an
    order that rests unfilled is mostly a way to get picked off when the price
    moves against you.
    """
    _assert_orders_allowed(client)

    if not 0.0 < price < 1.0:
        raise ValueError(f"price must be strictly between 0 and 1, got {price}")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if side not in ("bid", "ask"):
        raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")

    body = {
        "ticker": ticker,
        "side": side,
        "count": f"{count:.{COUNT_DECIMALS}f}",
        "price": f"{price:.{PRICE_DECIMALS}f}",
        "time_in_force": time_in_force,
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": client_order_id or str(uuid.uuid4()),
        "post_only": post_only,
    }

    try:
        resp = client.post(ORDERS_PATH, json=body, authed=True)
    except Exception as exc:
        _note_rejection(f"{side} {count} {ticker} @ {price}", exc)
        return None

    _CONSECUTIVE_REJECTS[0] = 0
    order = resp.get("order") or resp
    logger.info(
        f"Order placed: {side} {count} {ticker} @ {price} "
        f"tif={time_in_force} id={order.get('order_id')}"
    )
    return order


def get_order(client: KalshiClient, order_id: str) -> dict[str, Any] | None:
    """Fetch a single order, or None if it can't be read.

    Expect a 404 if you read an order immediately after creating it — the
    exchange takes a moment to make it readable by id (observed on demo, and
    docs/kalshi/_llms.txt:41 notes the general lag). `client.get` retries with
    back-off, which absorbs this. Do **not** treat a single 404 here as proof
    the order doesn't exist; that reads as a phantom order and tempts a
    duplicate submission.
    """
    try:
        resp = client.get(f"{ORDERS_LIST_PATH}/{order_id}", authed=True)
    except Exception as exc:
        logger.warning(f"Could not read order {order_id}: {exc!r}")
        return None
    return resp.get("order") or resp


def poll_order_status(client: KalshiClient, order_id: str) -> str | None:
    """Return an order's status: 'resting', 'canceled', 'executed', or None."""
    order = get_order(client, order_id)
    return None if order is None else order.get("status")


def get_resting_orders(
    client: KalshiClient,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """List orders still live on the exchange. This is the source of truth."""
    params: dict[str, Any] = {"status": "resting", "limit": 200}
    if ticker:
        params["ticker"] = ticker
    try:
        resp = client.get(ORDERS_LIST_PATH, params=params, authed=True)
    except Exception as exc:
        logger.warning(f"Could not list resting orders: {exc!r}")
        return []
    return resp.get("orders") or []


def cancel_order(client: KalshiClient, order_id: str) -> bool:
    """Cancel an order. Returns True only if the exchange confirmed it."""
    try:
        resp = client.delete(f"{ORDERS_PATH}/{order_id}", authed=True)
    except Exception as exc:
        logger.error(f"Cancel failed for {order_id}: {exc!r}")
        return False
    logger.info(f"Cancelled {order_id} (reduced_by={resp.get('reduced_by')})")
    return True


@dataclass
class OpenPositionTracker:
    """Tracks orders this bot has placed, so it doesn't double-enter a cycle.

    In-memory state is a *cache*, never the authority. The exchange is the
    authority, which is why `reconcile()` exists and why `cancel_all_open()`
    keeps entries it failed to cancel.
    """

    _orders: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(
        self,
        order_id: str,
        side: str,
        ticker: str,
        count: float,
        limit_price: float,
        placed_at: float,
    ) -> None:
        self._orders[order_id] = {
            "order_id": order_id,
            "side": side,
            "ticker": ticker,
            "count": count,
            "limit_price": limit_price,
            "placed_at": placed_at,
        }
        logger.info(f"Tracking order {order_id}: {side} {count} {ticker} @ {limit_price}")

    def remove(self, order_id: str) -> None:
        removed = self._orders.pop(order_id, None)
        if removed:
            logger.info(f"Untracking order {order_id} ({removed['side']} {removed['ticker']})")

    def has_open_position(self, side: str) -> bool:
        return any(o["side"] == side for o in self._orders.values())

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._orders.values())

    def cancel_all_open(self, client: KalshiClient) -> int:
        """Cancel every tracked order. Returns the number actually cancelled.

        Orders whose cancel failed are deliberately **kept** in the tracker. The
        obvious alternative — drop them regardless, to avoid stuck state — is
        worse: a failed cancel usually means the order is still live, and
        forgetting it produces a position the bot believes it doesn't have.
        Stale-but-known beats invisible-and-live. `reconcile()` clears the
        genuinely-dead ones on the next pass.
        """
        cancelled = 0
        for order_id in list(self._orders):
            if cancel_order(client, order_id):
                self.remove(order_id)
                cancelled += 1
            else:
                logger.warning(
                    f"Keeping {order_id} tracked — cancel failed, so it may still be live"
                )
        logger.info(f"cancel_all_open: cancelled {cancelled}/{cancelled + len(self._orders)}")
        return cancelled

    # Orders younger than this are exempt from reconciliation. The exchange
    # takes a moment to list a new order, and a reconcile that runs inside that
    # window concludes the order died — which invites an immediate duplicate
    # into the same market. Same read-after-write lag as `get_order`.
    RECONCILE_GRACE_S = 30.0

    def reconcile(self, client: KalshiClient) -> None:
        """Drop tracked orders the exchange no longer lists as resting."""
        live = {o.get("order_id") for o in get_resting_orders(client)}
        now = time.time()
        for order_id, order in list(self._orders.items()):
            if order_id in live:
                continue
            age = now - order.get("placed_at", 0.0)
            if age < self.RECONCILE_GRACE_S:
                logger.debug(
                    f"Reconcile: {order_id} not listed yet but only {age:.0f}s old — keeping"
                )
                continue
            logger.info(f"Reconcile: {order_id} no longer resting on the exchange")
            self.remove(order_id)


# Module-level singleton, as the tutorial's Task 1 specifies.
position_tracker = OpenPositionTracker()
