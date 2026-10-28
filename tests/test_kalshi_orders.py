"""Unit tests for Kalshi order construction and position tracking.

No network. The translation from a Signal to a book side is the highest-risk
logic in the bot — get it backwards and every trade is the opposite of what the
signal asked for — so it's covered explicitly.
"""
from __future__ import annotations

import pytest

from src.kalshi_orders import (
    LiveOrderBlocked,
    OpenPositionTracker,
    place_limit_order,
    signal_to_order,
)
from src.kalshi_signal import Signal


def make_signal(side: str, price: float) -> Signal:
    return Signal(
        side=side,
        confidence=0.7,
        edge=0.1,
        price=price,
        model_prob_yes=0.7,
        market_prob_yes=0.6,
        drift_pct=0.1,
        spread=0.01,
        seconds_to_expiry=450,
        reason="test",
    )


class FakeClient:
    """Stands in for KalshiClient without touching the network."""

    def __init__(self, demo: bool = True) -> None:
        self.demo = demo
        self.posted: list[dict] = []

    def post(self, path, json=None, *, authed=True):
        self.posted.append(json)
        return {"order": {"order_id": "abc123", "status": "resting"}}


class TestSignalTranslation:
    def test_yes_signal_becomes_a_bid_at_the_same_price(self) -> None:
        assert signal_to_order(make_signal("YES", 0.60)) == ("bid", 0.60)

    def test_no_signal_becomes_an_ask_at_the_complement(self) -> None:
        """Buying NO at $0.30 is selling YES at $0.70 — not an ask at 0.30."""
        assert signal_to_order(make_signal("NO", 0.30)) == ("ask", 0.70)

    def test_no_at_half_is_symmetric(self) -> None:
        assert signal_to_order(make_signal("NO", 0.50)) == ("ask", 0.50)

    def test_unknown_side_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            signal_to_order(make_signal("UP", 0.60))  # Polymarket terminology


class TestOrderValidation:
    def test_price_must_be_inside_the_unit_interval(self) -> None:
        client = FakeClient()
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                place_limit_order(client, "T", "bid", bad, 10)

    def test_count_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            place_limit_order(FakeClient(), "T", "bid", 0.5, 0)

    def test_side_must_be_bid_or_ask(self) -> None:
        with pytest.raises(ValueError):
            place_limit_order(FakeClient(), "T", "YES", 0.5, 10)

    def test_payload_formats_price_and_count_as_strings(self) -> None:
        client = FakeClient()
        place_limit_order(client, "KXBTC15M-X", "bid", 0.914, 3)
        body = client.posted[0]
        assert body["price"] == "0.9140", "deci-cent precision must survive"
        assert body["count"] == "3.00"
        assert body["time_in_force"] == "immediate_or_cancel"
        assert body["client_order_id"]


class TestLiveOrderGuard:
    def test_demo_orders_are_allowed(self) -> None:
        assert place_limit_order(FakeClient(demo=True), "T", "bid", 0.5, 1) is not None

    def test_production_orders_blocked_without_optin(self, monkeypatch) -> None:
        monkeypatch.delenv("KALSHI_ALLOW_LIVE_ORDERS", raising=False)
        with pytest.raises(LiveOrderBlocked):
            place_limit_order(FakeClient(demo=False), "T", "bid", 0.5, 1)

    def test_production_orders_allowed_with_optin(self, monkeypatch) -> None:
        monkeypatch.setenv("KALSHI_ALLOW_LIVE_ORDERS", "true")
        assert place_limit_order(FakeClient(demo=False), "T", "bid", 0.5, 1) is not None


class TestPositionTracker:
    def test_add_and_query(self) -> None:
        t = OpenPositionTracker()
        assert not t.has_open_position("YES")
        t.add("o1", "YES", "KXBTC15M-X", 10, 0.6, 1000.0)
        assert t.has_open_position("YES")
        assert not t.has_open_position("NO")
        assert len(t.get_all()) == 1

    def test_remove(self) -> None:
        t = OpenPositionTracker()
        t.add("o1", "YES", "T", 10, 0.6, 1000.0)
        t.remove("o1")
        assert t.get_all() == []

    def test_remove_unknown_id_is_harmless(self) -> None:
        OpenPositionTracker().remove("nope")

    def test_failed_cancel_keeps_the_order_tracked(self, monkeypatch) -> None:
        """A cancel that failed may mean the order is still live. Forgetting it
        would leave the bot holding a position it believes it doesn't have."""
        import src.kalshi_orders as mod

        t = OpenPositionTracker()
        t.add("ok", "YES", "T", 10, 0.6, 1000.0)
        t.add("fails", "NO", "T", 10, 0.4, 1000.0)

        monkeypatch.setattr(mod, "cancel_order", lambda c, oid: oid == "ok")
        assert t.cancel_all_open(FakeClient()) == 1

        remaining = [o["order_id"] for o in t.get_all()]
        assert remaining == ["fails"]
