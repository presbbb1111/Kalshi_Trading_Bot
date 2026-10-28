"""Unit tests for settlement PnL and reconciliation.

The mixed-units bug (`revenue` in cents, costs in dollars) and double-booking
are both silent failures that make a losing bot look profitable, so both are
pinned here.
"""
from __future__ import annotations

from src.kalshi_risk import RiskManager
from src.kalshi_settlement import SettlementReconciler, settlement_cost, settlement_pnl


def winning_settlement(ticker: str = "KXBTC15M-A") -> dict:
    """10 YES contracts bought at $0.60, resolved YES: revenue 1000c = $10."""
    return {
        "ticker": ticker,
        "market_result": "yes",
        "yes_count_fp": "10.00",
        "yes_total_cost_dollars": "6.00",
        "no_total_cost_dollars": "0",
        "revenue": 1000,
        "fee_cost": "0.20",
        "settled_time": "2026-07-26T23:30:00Z",
    }


def losing_settlement(ticker: str = "KXBTC15M-B") -> dict:
    """Same stake, resolved NO: revenue 0, the whole cost is lost."""
    return {
        "ticker": ticker,
        "market_result": "no",
        "yes_count_fp": "10.00",
        "yes_total_cost_dollars": "6.00",
        "no_total_cost_dollars": "0",
        "revenue": 0,
        "fee_cost": "0.20",
        "settled_time": "2026-07-26T23:45:00Z",
    }


class FakeClient:
    def __init__(self, settlements: list[dict]) -> None:
        self.demo = True
        self._settlements = settlements

    def get(self, path, params=None, *, authed=False):
        return {"settlements": self._settlements}


class TestSettlementMath:
    def test_revenue_is_converted_from_cents(self) -> None:
        """revenue=1000 is $10.00, not $1000. Getting this wrong inflates PnL 100x."""
        assert settlement_pnl(winning_settlement()) == 3.80   # 10.00 - 6.00 - 0.20

    def test_a_loss_is_the_full_stake_plus_fees(self) -> None:
        assert settlement_pnl(losing_settlement()) == -6.20

    def test_cost_includes_fees(self) -> None:
        assert settlement_cost(winning_settlement()) == 6.20

    def test_missing_fields_do_not_crash(self) -> None:
        assert settlement_pnl({"revenue": 0}) == 0.0

    def test_blank_strings_read_as_zero(self) -> None:
        assert settlement_cost({"yes_total_cost_dollars": "", "fee_cost": None}) == 0.0


class TestReconciler:
    def test_priming_prevents_backfilling_old_pnl(self) -> None:
        """Without priming, historical settlements would book as today's losses
        and could halt the bot before it ever trades."""
        client = FakeClient([losing_settlement() for _ in range(20)])
        risk = RiskManager(bankroll=1000.0)
        r = SettlementReconciler()
        r.prime(client)
        assert r.reconcile(client, risk) == []
        assert risk.realized_pnl_today == 0.0

    def test_new_settlement_is_booked_once(self) -> None:
        client = FakeClient([])
        risk = RiskManager(bankroll=1000.0)
        r = SettlementReconciler()
        r.prime(client)

        client._settlements = [winning_settlement()]
        assert len(r.reconcile(client, risk)) == 1
        assert risk.realized_pnl_today == 3.80

        # Kalshi keeps returning it; it must not be booked again.
        assert r.reconcile(client, risk) == []
        assert risk.realized_pnl_today == 3.80

    def test_losses_reduce_bankroll_and_can_halt(self) -> None:
        client = FakeClient([])
        risk = RiskManager(bankroll=100.0)   # limit = $10
        r = SettlementReconciler()
        r.prime(client)

        client._settlements = [losing_settlement(f"T{i}") for i in range(2)]
        r.reconcile(client, risk)
        assert risk.realized_pnl_today == -12.40
        assert risk.is_halted()

    def test_settlement_retires_open_exposure(self) -> None:
        """The whole point: exposure must come back down, or the bot jams."""
        client = FakeClient([])
        risk = RiskManager(bankroll=1000.0)
        r = SettlementReconciler()
        r.prime(client)

        risk.record_trade_open(6.20)
        assert risk.open_exposure == 6.20

        client._settlements = [winning_settlement()]
        r.reconcile(client, risk)
        assert risk.open_exposure == 0.0


class TestExitedPositionNotDoubleBooked:
    """The bug that silently stopped the bot for 27 days.

    An exited position still produces a settlement record, with both legs' cost
    and zero revenue. Booking that as a loss charges the bankroll twice for one
    trade — once when the exit sold, once when the market settled.
    """

    def _exited(self):
        return {
            "ticker": "KXHIGHNY-26SEP02-T80",
            "settled_time": "2026-09-04T19:31:45Z",
            "market_result": "yes",
            "revenue": 0,
            "yes_count_fp": "2.25",
            "no_count_fp": "2.25",
            "yes_total_cost_dollars": "1.775000",
            "no_total_cost_dollars": "1.417500",
            "fee_cost": "0.000000",
        }

    def _held(self):
        return {
            "ticker": "KXHIGHCHI-26SEP02-B78.5",
            "settled_time": "2026-09-04T19:31:45Z",
            "market_result": "no",
            "revenue": 185,
            "yes_count_fp": "0.00",
            "no_count_fp": "1.85",
            "yes_total_cost_dollars": "0.000000",
            "no_total_cost_dollars": "1.591000",
            "fee_cost": "0.000000",
        }

    def test_detects_exited_position(self):
        from src.kalshi_settlement import was_closed_before_settlement

        assert was_closed_before_settlement(self._exited()) is True
        assert was_closed_before_settlement(self._held()) is False

    def test_exited_settlement_leaves_bankroll_alone(self):
        from src.kalshi_risk import RiskManager
        from src.kalshi_settlement import SettlementReconciler

        risk = RiskManager(bankroll=100.0)
        risk.record_trade_open(3.19)
        rec = SettlementReconciler()
        rec._primed = True

        class FakeClient:
            def get(self, path, **kw):
                return {"settlements": [TestExitedPositionNotDoubleBooked()._exited()]}

        rec.reconcile(FakeClient(), risk)
        # Exposure freed, but no second loss charged.
        assert risk.bankroll == 100.0
        assert risk.open_exposure == 0.0

    def test_held_settlement_still_books_normally(self):
        from src.kalshi_risk import RiskManager
        from src.kalshi_settlement import SettlementReconciler

        risk = RiskManager(bankroll=100.0)
        rec = SettlementReconciler()
        rec._primed = True

        class FakeClient:
            def get(self, path, **kw):
                return {"settlements": [TestExitedPositionNotDoubleBooked()._held()]}

        rec.reconcile(FakeClient(), risk)
        # Won: 1.85 revenue on 1.591 cost = +0.259
        assert abs(risk.bankroll - 100.26) < 0.01


class TestBankrollSync:
    def test_resets_on_drift(self):
        from src.kalshi_risk import RiskManager

        risk = RiskManager(bankroll=39.43)
        assert risk.sync_bankroll(99.14) is True
        assert risk.bankroll == 99.14

    def test_ignores_noise(self):
        from src.kalshi_risk import RiskManager

        risk = RiskManager(bankroll=99.00)
        assert risk.sync_bankroll(99.50) is False
        assert risk.bankroll == 99.00

    def test_ignores_unreadable_balance(self):
        from src.kalshi_risk import RiskManager

        risk = RiskManager(bankroll=50.0)
        assert risk.sync_bankroll(0.0) is False
        assert risk.bankroll == 50.0
