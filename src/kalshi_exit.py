"""Take-profit and hold-to-settlement logic for open positions.

Default behaviour is to hold every contract to settlement, which turns each
trade into an all-or-nothing outcome. Exiting early when a position has moved
in your favour trades some of that variance away.

## What this does and does not buy

On a fairly-priced market, selling at 0.75 and holding a 75% position to
settlement have the **same expected value** — and exiting costs a taker fee. So
take-profit does not create edge. What it does is stop every position being a
coin flip, which matters when the edge is unproven: it loses more slowly rather
than winning more.

## The rules

* **Stop loss** once the position is down `STOP_LOSS_PCT` on cost.
* **Take profit** once unrealized gain reaches `TAKE_PROFIT_PCT`.
* **Unless** the contract is close to expiry and already deep in the money
  (market price above `HOLD_TO_SETTLE_PROB`), in which case hold. Near expiry a
  90c contract is very likely to pay $1, and paying a fee to capture 90c
  instead is worse than waiting.

## Why a stop matters more than the take-profit

Measured on 60 settled BTC 15M markets: holding losers to settlement returned
-$1.39 per contract, while cutting them at -15% returned -$0.56. **Cutting the
losers recovered 60% of the loss** — far more than the take-profit contributes.

The asymmetry is structural. A binary held to settlement loses 100% of the
stake when it goes wrong, and there is no partial credit for being nearly
right. Exiting at -15% converts a 100% loss into a 15% one, at the cost of
occasionally cutting a position that would have recovered.

Tighter is not better: stops at -10% and -5% both performed worse than -15%,
because they cut positions that still had time to come back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.kalshi_client import KalshiClient
from src.kalshi_markets import _parse_iso8601, get_top_of_book
from src.utils import get_logger

logger = get_logger(__name__)

TAKE_PROFIT_PCT = 0.15          # exit at +15% on cost
STOP_LOSS_PCT = -0.15           # exit at -15%; measured best of -0.05/-0.10/-0.15
HOLD_TO_SETTLE_PROB = 0.75      # ...unless already this likely to win
NEAR_EXPIRY_S = 300             # "close to expiry" window


@dataclass(frozen=True)
class ExitDecision:
    ticker: str
    action: str                 # 'exit' | 'hold'
    reason: str
    contracts: float = 0.0
    limit_price: float = 0.0
    book_side: str = ""
    gain_pct: float = 0.0


def evaluate_position(
    client: KalshiClient,
    position: dict[str, Any],
    close_time_iso: str | None = None,
) -> ExitDecision | None:
    """Decide whether to close `position` now. None if it can't be evaluated."""
    ticker = position.get("ticker")
    try:
        contracts = float(position.get("position_fp") or 0)
        cost = abs(float(position.get("market_exposure_dollars") or 0))
    except (TypeError, ValueError):
        return None
    if not ticker or contracts == 0 or cost <= 0:
        return None

    long_yes = contracts > 0
    size = abs(contracts)

    # Only the side we would trade against has to exist. Requiring both sides
    # rejects exits that are perfectly possible — on a one-sided book a long
    # YES can still be sold into the resting YES bid.
    book = get_top_of_book(client, ticker)
    if long_yes:
        if book.yes_bid is None:
            return ExitDecision(ticker, "hold", "no YES bid to sell into")
        exit_price = book.yes_bid
        limit_price = book.yes_bid
        book_side = "ask"          # selling YES
    else:
        if book.yes_ask is None:
            return ExitDecision(ticker, "hold", "no YES ask to buy back")
        exit_price = round(1.0 - book.yes_ask, 4)
        limit_price = book.yes_ask
        book_side = "bid"          # buying YES back to close a NO

    value_now = exit_price * size
    gain_pct = (value_now - cost) / cost

    # Probability this position wins, from the market's own price.
    win_prob = exit_price

    seconds_left = None
    if close_time_iso:
        import time

        seconds_left = _parse_iso8601(close_time_iso) - time.time()

    # Stop loss is checked first and has no near-expiry exemption: a losing
    # binary near expiry is the worst thing to hold, not the best.
    if gain_pct <= STOP_LOSS_PCT:
        return ExitDecision(
            ticker,
            "exit",
            f"stop loss at {gain_pct:+.1%}",
            contracts=round(size, 2),
            limit_price=limit_price,
            book_side=book_side,
            gain_pct=gain_pct,
        )

    if gain_pct < TAKE_PROFIT_PCT:
        return ExitDecision(ticker, "hold", f"gain {gain_pct:+.1%} below target", gain_pct=gain_pct)

    if (
        seconds_left is not None
        and seconds_left <= NEAR_EXPIRY_S
        and win_prob >= HOLD_TO_SETTLE_PROB
    ):
        return ExitDecision(
            ticker,
            "hold",
            f"near expiry ({seconds_left:.0f}s) and {win_prob:.0%} likely — ride to settlement",
            gain_pct=gain_pct,
        )

    return ExitDecision(
        ticker,
        "exit",
        f"take profit at {gain_pct:+.1%}",
        contracts=round(size, 2),
        limit_price=limit_price,
        book_side=book_side,
        gain_pct=gain_pct,
    )
