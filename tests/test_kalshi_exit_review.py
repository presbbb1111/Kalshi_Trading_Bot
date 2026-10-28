"""Tests for exit measurement.

The point of these is the arithmetic that a fills-based reconstruction got
wrong: a NO position's cost is `count * no_price`, and it wins when the market
resolves NO. Getting either backwards flips the sign of the whole study.
"""
from __future__ import annotations

from src.kalshi_exit_review import ExitedTrade, ExitJournal, _entry


def _trade(**kw) -> ExitedTrade:
    base = dict(
        ticker="KXHIGHNY-26SEP01-T80",
        contracts=2.0,
        entry_price=0.75,
        outcome_side="no",
        realized_pnl=0.10,
        result="no",
    )
    base.update(kw)
    return ExitedTrade(**base)


class TestHeldCounterfactual:
    def test_no_position_that_would_have_won(self):
        # 2 contracts at 0.75 cost $1.50; NO wins, so holding pays $2.00.
        t = _trade(result="no")
        assert t.cost == 1.5
        assert t.held_pnl == 0.5

    def test_no_position_that_would_have_lost(self):
        t = _trade(result="yes")
        assert t.held_pnl == -1.5

    def test_yes_position_wins_on_yes(self):
        t = _trade(outcome_side="yes", entry_price=0.40, result="yes")
        assert t.held_pnl == 1.2

    def test_unresolved_market_has_no_counterfactual(self):
        assert _trade(result=None).held_pnl is None
        assert _trade(result=None).benefit is None

    def test_benefit_is_exit_minus_hold(self):
        # Exiting banked +0.10; holding would have lost 1.50. Exiting saved 1.60.
        t = _trade(realized_pnl=0.10, result="yes")
        assert t.benefit == 1.6

    def test_exit_can_be_worse_than_holding(self):
        # Stopped out for -0.30 on a position that would have won +0.50.
        t = _trade(realized_pnl=-0.30, result="no")
        assert t.benefit == -0.8


class TestEntryFromFill:
    def test_no_side_uses_no_price(self):
        side, price = _entry(
            {"outcome_side": "no", "no_price_dollars": "0.85", "yes_price_dollars": "0.15"}
        )
        assert (side, price) == ("no", 0.85)

    def test_yes_side_uses_yes_price(self):
        side, price = _entry(
            {"outcome_side": "yes", "no_price_dollars": "0.85", "yes_price_dollars": "0.15"}
        )
        assert (side, price) == ("yes", 0.15)

    def test_falls_back_to_side_when_outcome_side_absent(self):
        side, price = _entry({"side": "no", "no_price_dollars": "0.60"})
        assert (side, price) == ("no", 0.60)


class TestJournal:
    def test_roundtrip_and_resolution(self, tmp_path):
        j = ExitJournal(tmp_path / "exits.csv")
        j.record("KXHIGHNY-26SEP01-T80", "stop_loss", 2.0, 0.75, "no", 0.60, -0.20)
        rows = j.rows()
        assert len(rows) == 1
        assert rows[0]["rule"] == "stop_loss"
        assert rows[0]["result"] == ""

        class FakeClient:
            def get(self, path, **kw):
                return {"market": {"result": "no"}}

        assert j.resolve(FakeClient()) == 1
        row = j.rows()[0]
        # Held to settle: 2 * 1.00 - 1.50 = +0.50. Exited at 0.60: 2*0.60 - 1.50 = -0.30.
        assert row["held_pnl"] == "+0.5000"
        assert row["benefit"] == "-0.8000"

    def test_resolve_skips_unsettled(self, tmp_path):
        j = ExitJournal(tmp_path / "exits.csv")
        j.record("KXHIGHNY-26SEP09-T80", "take_profit", 1.0, 0.70, "no", 0.85, 0.21)

        class FakeClient:
            def get(self, path, **kw):
                return {"market": {"result": ""}}

        assert j.resolve(FakeClient()) == 0
        assert j.rows()[0]["benefit"] == ""

    def test_summary_groups_by_rule(self, tmp_path):
        j = ExitJournal(tmp_path / "exits.csv")
        assert "no resolved exits" in j.summary()
