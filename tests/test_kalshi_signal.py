"""Unit tests for the Kalshi BTC 15M signal engine.

Pure logic, no network — these run in milliseconds and are the fastest way to
check a change to the model didn't break the reasoning.
"""
from __future__ import annotations

from src.kalshi_signal import generate_signal, model_prob_yes


class TestModelProbYes:
    def test_no_drift_is_a_coin_flip(self) -> None:
        assert model_prob_yes(0.0, 450) == 0.5

    def test_positive_drift_favours_yes(self) -> None:
        assert model_prob_yes(0.5, 450) > 0.5

    def test_negative_drift_favours_no(self) -> None:
        assert model_prob_yes(-0.5, 450) < 0.5

    def test_same_drift_is_more_decisive_with_less_time_left(self) -> None:
        """A 0.3% lead matters more with 60s left than with 800s left."""
        assert model_prob_yes(0.3, 60) > model_prob_yes(0.3, 800)

    def test_expired_window_collapses_to_certainty(self) -> None:
        assert model_prob_yes(0.2, 0) == 1.0
        assert model_prob_yes(-0.2, 0) == 0.0

    def test_probability_stays_in_range(self) -> None:
        for drift in (-99.0, -1.0, 0.0, 1.0, 99.0):
            for secs in (1, 120, 450, 900):
                assert 0.0 <= model_prob_yes(drift, secs) <= 1.0


class TestFilters:
    def test_wide_spread_is_rejected(self) -> None:
        # Strong drift that would otherwise fire, but a 20-cent spread.
        assert generate_signal(0.40, 0.60, drift_pct=1.0, seconds_to_expiry=450) is None

    def test_near_expiry_is_rejected(self) -> None:
        assert generate_signal(0.50, 0.52, drift_pct=1.0, seconds_to_expiry=90) is None

    def test_spread_exactly_at_limit_is_allowed(self) -> None:
        """0.06 is the boundary — the filter rejects *above* it, not at it."""
        sig = generate_signal(0.20, 0.26, drift_pct=1.0, seconds_to_expiry=450)
        assert sig is not None


class TestSignalDirection:
    def test_strong_up_drift_buys_yes_when_underpriced(self) -> None:
        sig = generate_signal(0.50, 0.52, drift_pct=1.0, seconds_to_expiry=450)
        assert sig is not None
        assert sig.side == "YES"
        assert sig.edge >= 0.05

    def test_strong_down_drift_buys_no_when_underpriced(self) -> None:
        sig = generate_signal(0.48, 0.50, drift_pct=-1.0, seconds_to_expiry=450)
        assert sig is not None
        assert sig.side == "NO"

    def test_flat_market_produces_no_signal(self) -> None:
        assert generate_signal(0.49, 0.51, drift_pct=0.0, seconds_to_expiry=450) is None


class TestEdgeIsRealNotAnIdentity:
    """Guards against the bug in the Polymarket starter's signal_engine.py,
    where the market price cancelled out of the edge calculation entirely."""

    def test_market_price_changes_the_decision(self) -> None:
        """Same drift, same time — only the price differs. Cheap should fire."""
        cheap = generate_signal(0.20, 0.22, drift_pct=0.6, seconds_to_expiry=450)
        rich = generate_signal(0.94, 0.96, drift_pct=0.6, seconds_to_expiry=450)
        assert cheap is not None, "underpriced YES should produce a signal"
        assert rich is None, "already-priced-in YES must not produce a signal"

    def test_does_not_chase_a_market_that_already_agrees(self) -> None:
        """Big drift but the market has fully priced it — no edge left."""
        assert generate_signal(0.97, 0.99, drift_pct=2.0, seconds_to_expiry=450) is None

    def test_model_prob_is_independent_of_quotes(self) -> None:
        a = generate_signal(0.20, 0.22, drift_pct=0.6, seconds_to_expiry=450)
        b = generate_signal(0.30, 0.32, drift_pct=0.6, seconds_to_expiry=450)
        assert a is not None and b is not None
        assert a.model_prob_yes == b.model_prob_yes
        assert a.edge != b.edge
