"""Settlement reconciliation — turning resolved markets into realized PnL.

This closes the loop that `RiskManager` depends on. A cycle opens exposure; only
settlement can retire it. Without this module `open_exposure` grows forever
until the bot jams against its own cap, and the daily loss breaker never fires
because no loss is ever recorded.

## Units warning

`GET /portfolio/settlements` mixes units in one object: `revenue` is an
**integer in cents**, while `yes_total_cost_dollars`, `no_total_cost_dollars`
and `fee_cost` are **strings in dollars**. Subtracting one from the other
directly overstates profit by 100x, which reads as a wildly profitable bot.
`settlement_pnl` normalises to dollars first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from src.kalshi_client import KalshiClient
from src.kalshi_risk import RiskManager
from src.utils import get_logger

logger = get_logger(__name__)

SETTLEMENTS_PATH = "portfolio/settlements"


def _settled_date(s: dict[str, Any]) -> date | None:
    """The day this market's outcome was determined, from its ticker.

    Deliberately **not** `settled_time`: that is when Kalshi published the
    settlement, which can lag the outcome by days. On 2026-08-31 the exchange
    released four days of backlog at once, all stamped with that day's time — so
    keying off `settled_time` would attribute Aug-27 results to Aug 31 and trip
    the daily breaker on history.

    Tickers encode the target day as `KXHIGHNY-26AUG27-T80` (YYMONDD).
    Returning None makes the caller treat the P&L as today's, which is the
    conservative direction: it can only make the breaker fire sooner.
    """
    parts = (s.get("ticker") or "").split("-")
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime(parts[1], "%y%b%d").date()
    except ValueError:
        return None


def _dollars(raw: Any) -> float:
    """Parse a dollar-denominated string field. Missing/blank reads as 0."""
    if raw in (None, ""):
        return 0.0
    return float(raw)


def settlement_cost(s: dict[str, Any]) -> float:
    """Total dollars staked on a settled market, fees included."""
    return (
        _dollars(s.get("yes_total_cost_dollars"))
        + _dollars(s.get("no_total_cost_dollars"))
        + _dollars(s.get("fee_cost"))
    )


def settlement_pnl(s: dict[str, Any]) -> float:
    """Realized profit or loss in dollars for one settled market."""
    revenue_dollars = float(s.get("revenue") or 0) / 100.0   # cents -> dollars
    return round(revenue_dollars - settlement_cost(s), 4)


def was_closed_before_settlement(s: dict[str, Any]) -> bool:
    """True if this position had already been exited when the market settled.

    Kalshi reports a closed position's settlement with **both** legs filled in —
    `yes_count_fp` and `no_count_fp` both non-zero, their costs summed, and
    `revenue` of 0, because a flat position pays nothing. Scored naively that
    reads as a total loss of the entire round trip.

    This is not a small distortion. Across this account, 41 such records scored
    -98.3% ROI on $114.57 of cost, while the 169 genuinely-held positions came
    in at -6.1%. Booking the phantom losses drove the in-memory bankroll from
    $97 down to $39 against an exchange balance that never moved, which pushed
    every Kelly size under the $1 minimum and silently stopped the bot from
    2026-09-06 to 10-03. The exit already realized this P&L when it sold; the
    settlement record is a duplicate, not a second loss.
    """
    return (
        float(s.get("yes_count_fp") or 0) > 0 and float(s.get("no_count_fp") or 0) > 0
    )


def get_settlements(
    client: KalshiClient,
    *,
    min_ts: int | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Fetch settlement records, newest-first as the API returns them."""
    params: dict[str, Any] = {"limit": limit}
    if min_ts is not None:
        params["min_ts"] = min_ts
    try:
        resp = client.get(SETTLEMENTS_PATH, params=params, authed=True)
    except Exception as exc:
        logger.warning(f"Could not fetch settlements: {exc!r}")
        return []
    return resp.get("settlements") or []


@dataclass
class SettlementReconciler:
    """Books newly-settled markets into a RiskManager exactly once.

    Kalshi keeps returning a settlement after it's been read, so the reconciler
    remembers what it has already booked. Double-booking would corrupt both the
    bankroll and the daily loss counter — in the direction that hides losses.
    """

    _seen: set[str] = field(default_factory=set)
    _primed: bool = False
    # Only book settlements whose ticker starts with this. Several bots share
    # one Kalshi account, and without a filter each one books every other bot's
    # losses as its own — tripping every daily-loss breaker at once after a
    # single trade.
    ticker_prefix: str | None = None

    def prime(self, client: KalshiClient) -> None:
        """Mark all existing settlements as already-known.

        Call once at startup. Without it, the first pass books every historical
        settlement as though it just happened, which can trip the daily loss
        breaker before the bot has placed a single trade.
        """
        for s in self._mine(get_settlements(client)):
            key = self._key(s)
            if key:
                self._seen.add(key)
        self._primed = True
        logger.info(f"Settlement reconciler primed with {len(self._seen)} existing records")

    def reconcile(self, client: KalshiClient, risk: RiskManager) -> list[dict[str, Any]]:
        """Book any settlements not seen before. Returns the newly-booked ones."""
        if not self._primed:
            logger.warning("Reconciler was not primed — priming now to avoid backfilling old PnL")
            self.prime(client)
            return []

        booked: list[dict[str, Any]] = []
        for s in self._mine(get_settlements(client)):
            key = self._key(s)
            if not key or key in self._seen:
                continue

            self._seen.add(key)
            cost = settlement_cost(s)
            if was_closed_before_settlement(s):
                # Already realized by the exit that closed it. Release the
                # exposure, but do not book the P&L a second time.
                logger.info(
                    f"Settled {s.get('ticker')}: position was exited before "
                    f"settlement — releasing ${cost:.2f}, P&L already booked"
                )
                risk.release_exposure(cost)
                booked.append(s)
                continue

            pnl = settlement_pnl(s)
            risk.record_trade_close(cost, pnl, earned_on=_settled_date(s))
            logger.info(
                f"Settled {s.get('ticker')}: result={s.get('market_result')} "
                f"cost=${cost:.2f} pnl={pnl:+.2f}"
            )
            booked.append(s)

        return booked

    def _mine(self, settlements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only settlements belonging to this bot's series."""
        if not self.ticker_prefix:
            return settlements
        return [s for s in settlements if (s.get("ticker") or "").startswith(self.ticker_prefix)]

    @staticmethod
    def _key(s: dict[str, Any]) -> str | None:
        ticker = s.get("ticker")
        if not ticker:
            return None
        return f"{ticker}@{s.get('settled_time', '')}"
