"""Unit tests for bankroll controls.

The circuit breaker is the control you least want to discover is broken, and
it can only be exercised by simulating a losing day — which is exactly what a
live run won't do on demand.
"""
from __future__ import annotations

from datetime import date, timedelta

from src.kalshi_risk import RiskManager


def make_rm(**kw) -> RiskManager:
    return RiskManager(bankroll=1000.0, **kw)


class TestKellySizing:
    def test_no_edge_means_no_bet(self) -> None:
        assert make_rm().kelly_size(edge=0.0, price=0.5) == 0.0

    def test_negative_edge_means_no_bet(self) -> None:
        assert make_rm().kelly_size(edge=-0.1, price=0.5) == 0.0

    def test_bigger_edge_sizes_bigger(self) -> None:
        rm = make_rm(max_trade_fraction=1.0)
        assert rm.kelly_size(0.10, 0.5) > rm.kelly_size(0.05, 0.5)

    def test_uses_price_not_confidence_for_odds(self) -> None:
        """Kelly is (p-c)/(1-c). Same edge at a higher price stakes MORE,
        because the payoff per dollar risked is smaller."""
        rm = make_rm(max_trade_fraction=1.0)
        assert rm.kelly_size(0.05, 0.90) > rm.kelly_size(0.05, 0.10)

    def test_quarter_kelly_is_a_quarter_of_full(self) -> None:
        full = make_rm(kelly_fraction=1.0, max_trade_fraction=1.0)
        quarter = make_rm(kelly_fraction=0.25, max_trade_fraction=1.0)
        assert abs(quarter.kelly_size(0.1, 0.5) - full.kelly_size(0.1, 0.5) / 4) < 0.01

    def test_per_trade_cap_binds(self) -> None:
        """A huge edge must still respect the cap."""
        rm = make_rm(max_trade_fraction=0.02)
        assert rm.kelly_size(0.9, 0.05) == 20.0  # 2% of 1000

    def test_invalid_price_refuses_to_size(self) -> None:
        rm = make_rm()
        assert rm.kelly_size(0.1, 0.0) == 0.0
        assert rm.kelly_size(0.1, 1.0) == 0.0

    def test_max_size_is_always_accepted_by_can_trade(self) -> None:
        """Regression: rounding up past the cap made can_trade reject the very
        trades kelly_size had maxed out, so the bot went quiet on its best edges.

        The awkward bankroll matters — 200.46 * 0.02 = 4.0092, which rounds to
        4.01 and exceeds the cap. Flooring keeps it at 4.00.
        """
        for bankroll in (200.46, 1000.0, 333.33, 97.77, 1234.56):
            rm = RiskManager(bankroll=bankroll)
            size = rm.kelly_size(edge=0.9, price=0.5)   # huge edge -> pinned to the cap
            ok, reason = rm.can_trade(size)
            assert ok, f"bankroll {bankroll}: kelly_size gave {size} but can_trade said {reason}"

    def test_size_never_exceeds_the_cap(self) -> None:
        rm = RiskManager(bankroll=200.46)
        assert rm.kelly_size(0.9, 0.5) <= rm.bankroll * rm.max_trade_fraction


class TestTradeGating:
    def test_normal_trade_allowed(self) -> None:
        ok, reason = make_rm().can_trade(10.0)
        assert ok, reason

    def test_dust_trade_rejected(self) -> None:
        ok, reason = make_rm().can_trade(0.50)
        assert not ok and "minimum" in reason

    def test_oversized_trade_rejected(self) -> None:
        ok, reason = make_rm().can_trade(500.0)
        assert not ok and "per-trade cap" in reason

    def test_open_exposure_cap_binds(self) -> None:
        rm = make_rm()          # max open = 6% of 1000 = $60
        rm.record_trade_open(55.0)
        ok, reason = rm.can_trade(20.0)
        assert not ok and "open-exposure cap" in reason

    def test_exposure_frees_up_after_close(self) -> None:
        rm = make_rm()
        rm.record_trade_open(55.0)
        rm.record_trade_close(55.0, realized_pnl=0.0)
        ok, _ = rm.can_trade(20.0)
        assert ok


class TestCircuitBreaker:
    def test_breaker_trips_after_losses(self) -> None:
        rm = make_rm()                      # limit = 10% of 1000 = $100
        assert not rm.is_halted()
        rm.record_trade_open(20.0)
        rm.record_trade_close(20.0, realized_pnl=-120.0)
        assert rm.is_halted()

    def test_halted_manager_refuses_all_trades(self) -> None:
        rm = make_rm()
        rm.record_trade_close(0.0, realized_pnl=-150.0)
        ok, reason = rm.can_trade(5.0)
        assert not ok and "daily loss limit" in reason

    def test_wins_do_not_trip_the_breaker(self) -> None:
        rm = make_rm()
        for _ in range(10):
            rm.record_trade_close(10.0, realized_pnl=+15.0)
        assert not rm.is_halted()

    def test_breaker_resets_on_a_new_day(self) -> None:
        """A latched breaker that never resets looks identical to a dead bot."""
        rm = make_rm()
        rm.record_trade_close(0.0, realized_pnl=-150.0)
        assert rm.is_halted()

        rm._pnl_date = date.today() - timedelta(days=1)
        assert not rm.is_halted()
        assert rm.realized_pnl_today == 0.0

    def test_realized_losses_shrink_the_bankroll(self) -> None:
        """Sizing must follow the bankroll down, not keep betting the old one."""
        rm = make_rm()
        before = rm.kelly_size(0.1, 0.5)
        rm.record_trade_close(0.0, realized_pnl=-500.0)
        assert rm.bankroll == 500.0
        assert rm.kelly_size(0.1, 0.5) < before


class TestExpectedVsRealizedPnl:
    """Guards the flaw in the tutorial's Task 1, which books `size * edge` at
    fill time — recording every trade as a winner and disabling the breaker."""

    def test_a_losing_binary_outcome_is_a_full_loss(self) -> None:
        rm = make_rm()
        rm.record_trade_open(20.0)
        # Contract expired worthless: the whole stake is gone, not `size*edge`.
        rm.record_trade_close(20.0, realized_pnl=-20.0)
        assert rm.realized_pnl_today == -20.0
        assert rm.bankroll == 980.0

    def test_a_run_of_losses_reaches_the_breaker(self) -> None:
        """With expected-PnL bookkeeping this loop would never halt."""
        rm = make_rm()
        for _ in range(6):
            if rm.is_halted():
                break
            rm.record_trade_open(20.0)
            rm.record_trade_close(20.0, realized_pnl=-20.0)
        assert rm.is_halted()
        assert rm.realized_pnl_today <= -100.0
