"""Tests for near-certainty scanning.

The empty-book case is the one that matters. A first pass found 516
"opportunities" on demo, every one of which was an untraded market quoting
bid 0.01 / ask 0.98 — both sides land inside the price band, so a price-only
filter reports two near-certainties on a market nobody is trading.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.nearcert import (
    MAX_HOURS_TO_CLOSE,
    PRICE_MAX,
    PRICE_MIN,
    NearCertQuote,
    QuotePersistence,
    _quotes_from,
)

# Close times are built relative to a fixed `NOW` rather than hardcoded. An
# earlier version pinned them to a literal date, and the whole suite began
# failing three days later when that date went into the past.
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
NOW_TS = NOW.timestamp()


def market(
    bid: float,
    ask: float,
    *,
    shard: int = 0,
    hours_out: float = 0.5,
    ticker: str = "KXNFLGAME-26OCT14X-A",
):
    return {
        "ticker": ticker,
        "yes_bid_dollars": str(bid),
        "yes_ask_dollars": str(ask),
        "exchange_index": shard,
        "close_time": (NOW + timedelta(hours=hours_out)).isoformat().replace("+00:00", "Z"),
    }


def quotes(m):
    return _quotes_from(m, funded_shard_only=False, now=NOW_TS)


class TestLeadTimeGuard:
    """The defect this exists to prevent.

    Without it the bot bought NFL spreads 5-12 days out — pre-game favourites,
    not decided outcomes — and went 8-for-8, which looked like confirmation and
    tested nothing. 95.5% of real near-certainty trades happen inside one hour.
    """

    def test_in_play_market_passes(self):
        assert len(quotes(market(0.98, 0.99, hours_out=0.2))) == 1

    def test_days_out_rejected_even_at_a_perfect_price(self):
        assert quotes(market(0.98, 0.99, hours_out=24 * 12)) == []

    def test_just_inside_the_cap_passes(self):
        assert len(quotes(market(0.98, 0.99, hours_out=MAX_HOURS_TO_CLOSE - 0.1))) == 1

    def test_just_outside_the_cap_rejected(self):
        assert quotes(market(0.98, 0.99, hours_out=MAX_HOURS_TO_CLOSE + 0.1)) == []

    def test_already_closed_rejected(self):
        assert quotes(market(0.98, 0.99, hours_out=-1)) == []

    def test_unreadable_close_time_rejected_rather_than_assumed_fine(self):
        m = market(0.98, 0.99)
        m["close_time"] = ""
        assert quotes(m) == []

    def test_lead_time_recorded_on_the_quote(self):
        q = quotes(market(0.98, 0.99, hours_out=0.25))[0]
        assert abs(q.hours_left - 0.25) < 0.01


class TestEmptyBookGuard:
    def test_wide_book_yields_nothing_even_though_both_sides_look_certain(self):
        # bid 0.01 / ask 0.98: buying YES costs 0.98, buying NO costs 0.99.
        # Price alone would report two near-certainties on a dead market.
        assert quotes(market(0.01, 0.98)) == []

    def test_tight_book_at_the_top_is_a_real_quote(self):
        found = quotes(market(0.98, 0.99))
        assert [(q.side, q.price) for q in found] == [("YES", 0.99)]

    def test_tight_book_at_the_bottom_is_a_real_NO_quote(self):
        # bid 0.01 / ask 0.02 -> buying NO costs 0.99.
        found = quotes(market(0.01, 0.02))
        assert [(q.side, q.price) for q in found] == [("NO", 0.99)]

    def test_midrange_book_is_not_near_certainty(self):
        assert quotes(market(0.50, 0.51)) == []

    def test_crossed_or_empty_book_rejected(self):
        assert quotes(market(0.0, 0.0)) == []
        assert quotes(market(0.99, 0.98)) == []


class TestShardFilter:
    def test_unfunded_shard_skipped_when_requested(self):
        assert _quotes_from(market(0.98, 0.99, shard=2), funded_shard_only=True, now=NOW_TS) == []

    def test_funded_shard_allowed(self):
        assert len(_quotes_from(market(0.98, 0.99, shard=0), funded_shard_only=True, now=NOW_TS)) == 1

    def test_shard_ignored_for_measurement_scans(self):
        assert len(quotes(market(0.98, 0.99, shard=2))) == 1


class TestOrderTranslation:
    """Buying NO is an ask at the mirror price — inverting this opens the
    opposite position, so it is pinned down here rather than at the call site."""

    def test_buying_yes_is_a_bid_at_the_price_paid(self):
        q = NearCertQuote("T", "YES", 0.99, 0.01, "", 0)
        assert (q.book_side, q.limit_price) == ("bid", 0.99)

    def test_buying_no_is_an_ask_at_one_minus_the_price_paid(self):
        q = NearCertQuote("T", "NO", 0.99, 0.01, "", 0)
        assert (q.book_side, q.limit_price) == ("ask", 0.01)


class TestPersistence:
    def test_lifetime_recorded_once_a_quote_disappears(self):
        p = QuotePersistence()
        q = NearCertQuote("T", "YES", 0.99, 0.01, "", 0)
        p.observe([q])
        assert "none expired yet" in p.summary()
        p.observe([])
        assert len(p.lifetimes) == 1
        assert "1 expired" in p.summary()

    def test_quote_still_live_is_not_counted_as_expired(self):
        p = QuotePersistence()
        q = NearCertQuote("T", "YES", 0.99, 0.01, "", 0)
        p.observe([q])
        p.observe([q])
        assert p.lifetimes == []

    def test_both_sides_of_one_market_tracked_separately(self):
        p = QuotePersistence()
        p.observe([
            NearCertQuote("T", "YES", 0.99, 0.01, "", 0),
            NearCertQuote("T", "NO", 0.99, 0.01, "", 0),
        ])
        assert len(p.first_seen) == 2


def test_band_bounds_are_inclusive():
    assert len(quotes(market(PRICE_MIN - 0.01, PRICE_MIN))) == 1
    assert len(quotes(market(PRICE_MAX - 0.01, PRICE_MAX))) == 1


class TestExpiryFieldSelection:
    """`close_time` means different things on open vs settled markets.

    On an open market it is the far-future deadline; the real one is
    `expected_expiration_time`. Reading the wrong one made the live scanner
    return zero qualifying quotes for 2,250 consecutive ticks while the
    backtest — which runs on settled markets, where `close_time` is correct —
    reported a median lead time of 8 minutes.
    """

    def test_prefers_expected_expiration_over_deadline(self):
        from src.nearcert import expiry_iso

        m = {
            "close_time": "2026-10-21T00:20:00Z",
            "expected_expiration_time": "2026-10-19T06:20:00Z",
        }
        assert expiry_iso(m) == "2026-10-19T06:20:00Z"

    def test_falls_back_to_close_time_when_absent(self):
        from src.nearcert import expiry_iso

        assert expiry_iso({"close_time": "2026-10-21T00:20:00Z"}) == "2026-10-21T00:20:00Z"

    def test_empty_when_neither_present(self):
        from src.nearcert import expiry_iso

        assert expiry_iso({}) == ""

    def test_live_game_passes_on_expected_expiration(self):
        m = market(0.98, 0.99, hours_out=24 * 7)        # deadline a week out
        m["expected_expiration_time"] = (
            (NOW + timedelta(hours=3)).isoformat().replace("+00:00", "Z")
        )
        assert len(quotes(m)) == 1
