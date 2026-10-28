"""Bankroll controls: position sizing, exposure caps, and a loss circuit breaker.

Three independent layers, in the order they bind:

1. **Kelly sizing** — how much *this* trade is worth, given its edge.
2. **Exposure caps** — how much may be at risk at once, regardless of edge.
3. **Daily loss breaker** — a hard stop that ends trading for the day.

Each is a veto. Layer 1 proposes; layers 2 and 3 only ever shrink or refuse.

## On realized vs expected PnL

`record_trade_close` takes **realized** PnL — what actually settled. It is
tempting to book `size * edge` at fill time instead, since the outcome isn't
known for another 15 minutes. Don't: that records every trade as a winner equal
to its predicted edge, so the circuit breaker never trips and the one control
that limits a bad day is silently disabled. On a binary contract the result is
$1 per contract or $0, and nothing in between is real.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from src.utils import get_logger

logger = get_logger(__name__)

# Fraction of full Kelly to actually bet. Full Kelly is the growth-optimal size
# *when the edge estimate is exact*; it is violently overleveraged when the
# estimate is noisy, and a 15-minute crypto signal is noisy. Quarter-Kelly gives
# up ~25% of theoretical growth for a large reduction in drawdown.
DEFAULT_KELLY_FRACTION = 0.25

DEFAULT_MAX_TRADE_FRACTION = 0.02    # ≤2% of bankroll per trade
DEFAULT_MAX_OPEN_FRACTION = 0.06     # ≤6% of bankroll at risk at once
DEFAULT_DAILY_LOSS_FRACTION = 0.10   # stop for the day after −10%
DEFAULT_MIN_TRADE_SIZE = 1.00        # below this, fees dominate


@dataclass
class RiskManager:
    bankroll: float
    kelly_fraction: float = DEFAULT_KELLY_FRACTION
    max_trade_fraction: float = DEFAULT_MAX_TRADE_FRACTION
    max_open_fraction: float = DEFAULT_MAX_OPEN_FRACTION
    daily_loss_fraction: float = DEFAULT_DAILY_LOSS_FRACTION
    min_trade_size: float = DEFAULT_MIN_TRADE_SIZE

    open_exposure: float = 0.0
    realized_pnl_today: float = 0.0
    _pnl_date: date = field(default_factory=date.today)

    # ---- sizing -----------------------------------------------------------

    def kelly_size(self, edge: float, price: float) -> float:
        """Dollars to stake on a contract costing `price` with modelled `edge`.

        For a binary contract paying $1, the Kelly fraction reduces to
        `(p - c) / (1 - c)`, where `p - c` is the edge and `c` is the price.
        That is the whole formula — no separate odds term is needed, and
        deriving odds from the probability estimate instead of the price (a
        common error) misprices every bet.
        """
        if edge <= 0:
            return 0.0
        if not 0.0 < price < 1.0:
            logger.warning(f"kelly_size: price {price} outside (0,1) — refusing to size")
            return 0.0

        full_kelly = edge / (1.0 - price)
        staked_fraction = min(full_kelly * self.kelly_fraction, self.max_trade_fraction)

        # Floor to whole cents rather than rounding. Rounding up can push the
        # size a fraction of a cent past the cap, and `can_trade` would then
        # reject the very trades this method sized at the maximum — the bot
        # would go quiet exactly when the edge was biggest.
        return max(0.0, math.floor(self.bankroll * staked_fraction * 100) / 100)

    # ---- gating -----------------------------------------------------------

    def can_trade(self, proposed_size: float) -> tuple[bool, str]:
        """Return `(allowed, reason)`. Reason is human-readable either way."""
        self._roll_day_if_needed()

        if proposed_size < self.min_trade_size:
            return False, f"size ${proposed_size:.2f} below minimum ${self.min_trade_size:.2f}"

        loss_limit = self.bankroll * self.daily_loss_fraction
        if self.realized_pnl_today <= -loss_limit:
            return False, (
                f"daily loss limit hit: {self.realized_pnl_today:+.2f} "
                f"vs limit -{loss_limit:.2f} — trading halted for the day"
            )

        max_trade = self.bankroll * self.max_trade_fraction
        if proposed_size > max_trade:
            return False, f"size ${proposed_size:.2f} exceeds per-trade cap ${max_trade:.2f}"

        max_open = self.bankroll * self.max_open_fraction
        if self.open_exposure + proposed_size > max_open:
            return False, (
                f"would exceed open-exposure cap: "
                f"${self.open_exposure:.2f} + ${proposed_size:.2f} > ${max_open:.2f}"
            )

        return True, "ok"

    # ---- bookkeeping ------------------------------------------------------

    def record_trade_open(self, size: float) -> None:
        self.open_exposure = round(self.open_exposure + size, 2)
        logger.info(f"Trade opened: ${size:.2f} — open exposure now ${self.open_exposure:.2f}")

    def record_trade_close(
        self, size: float, realized_pnl: float, *, earned_on: date | None = None
    ) -> None:
        """Book a *settled* result. `realized_pnl` is profit or loss in dollars.

        `earned_on` is the day the market actually resolved, which is not always
        the day we read the settlement. Weather markets settle 1-3 days after
        entry and the exchange publishes them in lumps, so a backlog can land
        several days of P&L in a single tick. Counting all of it as *today's*
        loss trips the daily breaker on results that are already history — which
        is what halted this bot on 2026-08-31, when four days of Aug-27
        settlements arrived at once and exceeded the 10% limit in one pass.

        The bankroll and exposure always move; only the daily counter is
        date-aware, because that counter exists to stop a bot that is bleeding
        *now*.
        """
        self._roll_day_if_needed()
        self.open_exposure = round(max(0.0, self.open_exposure - size), 2)
        if earned_on is None or earned_on == self._pnl_date:
            self.realized_pnl_today = round(self.realized_pnl_today + realized_pnl, 2)
        self.bankroll = round(self.bankroll + realized_pnl, 2)
        logger.info(
            f"Trade closed: ${size:.2f} pnl {realized_pnl:+.2f} — "
            f"today {self.realized_pnl_today:+.2f}, bankroll ${self.bankroll:.2f}, "
            f"open ${self.open_exposure:.2f}"
        )
        if self.is_halted():
            logger.error(
                f"DAILY LOSS LIMIT REACHED ({self.realized_pnl_today:+.2f}) — "
                "no further trades will be permitted today"
            )

    def sync_bankroll(self, actual_equity: float, *, tolerance: float = 2.0) -> bool:
        """Reset the bankroll to the exchange's own figure when it has drifted.

        The in-memory bankroll is derived: a starting balance plus every booked
        settlement. Any accounting error compounds in it forever, and because
        Kelly sizes scale off it, a bankroll that drifts low quietly throttles
        the bot to nothing. That happened — double-booked exit settlements took
        it from $97 to $39 while the real balance sat at $99, and every
        subsequent trade was refused as "below minimum" for 27 days.

        The exchange balance cannot drift, so it wins. Returns True if a
        correction was applied.
        """
        if actual_equity <= 0:
            return False
        drift = actual_equity - self.bankroll
        if abs(drift) < tolerance:
            return False
        logger.warning(
            f"Bankroll drift: tracked ${self.bankroll:.2f} vs exchange "
            f"${actual_equity:.2f} ({drift:+.2f}) — resetting to the exchange"
        )
        self.bankroll = round(actual_equity, 2)
        return True

    def release_exposure(self, size: float) -> None:
        """Free exposure for an order that was cancelled without filling.

        Distinct from `record_trade_close`: no money changed hands, so the
        bankroll and the daily PnL must not move. Without this, cancelled
        orders leak exposure until the open cap jams the bot shut.
        """
        self.open_exposure = round(max(0.0, self.open_exposure - size), 2)
        logger.info(f"Released ${size:.2f} unfilled — open exposure now ${self.open_exposure:.2f}")

    def is_halted(self) -> bool:
        self._roll_day_if_needed()
        return self.realized_pnl_today <= -(self.bankroll * self.daily_loss_fraction)

    def _roll_day_if_needed(self) -> None:
        """Reset the daily counter when the calendar date changes.

        Without this the breaker latches permanently after one bad day — and a
        bot that silently stops trading looks identical to one that is broken.
        """
        today = date.today()
        if today != self._pnl_date:
            logger.info(
                f"New trading day ({today}); resetting daily PnL from "
                f"{self.realized_pnl_today:+.2f}"
            )
            self._pnl_date = today
            self.realized_pnl_today = 0.0
