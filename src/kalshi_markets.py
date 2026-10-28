"""Active-market lookup and top-of-book for Kalshi's BTC 15-minute series.

The series is `KXBTC15M` ("BTC price up in next 15 mins?"). A new market opens
every 15 minutes, so — exactly as on Polymarket — the active ticker must be
re-resolved every cycle rather than hardcoded.

Outcomes are YES / NO here, not UP / DOWN. YES means the price went up.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from src.kalshi_client import KalshiClient
from src.utils import get_logger

logger = get_logger(__name__)

SERIES_TICKER = "KXBTC15M"


@dataclass(frozen=True)
class KalshiMarket:
    ticker: str
    title: str
    close_time_iso: str

    @property
    def seconds_to_expiry(self) -> int:
        return max(0, int(_parse_iso8601(self.close_time_iso) - time.time()))


@dataclass(frozen=True)
class TopOfBook:
    """Best prices in dollars. `None` means that side of the book is empty.

    Kalshi returns bids only — never asks. A YES ask is derived from the NO bid,
    because a NO bid at $0.79 is the same offer as a YES ask at $0.21.
    See docs/kalshi/_llms-full.txt:7079.
    """

    yes_bid: float | None
    yes_ask: float | None
    no_bid: float | None
    no_ask: float | None

    @property
    def yes_mid(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return round((self.yes_bid + self.yes_ask) / 2, 4)

    @property
    def spread(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return round(self.yes_ask - self.yes_bid, 4)


def get_active_btc_15m_market(
    client: KalshiClient, series: str = SERIES_TICKER
) -> KalshiMarket:
    """Return the open KXBTC15M market closing soonest, or raise RuntimeError.

    Kalshi usually has the next cycle listed alongside the live one, so sorting
    by close time and taking the first is what selects the *current* window.
    """
    body = client.get(
        "markets",
        params={"series_ticker": series, "status": "open", "limit": 20},
    )
    markets = body.get("markets") or []

    now = time.time()
    live = [
        m
        for m in markets
        if m.get("close_time") and _parse_iso8601(m["close_time"]) > now
    ]
    if not live:
        raise RuntimeError(
            f"No open {series} market found. Markets briefly disappear as a "
            "cycle rolls; retry in 5–10 seconds before giving up on the run."
        )

    live.sort(key=lambda m: _parse_iso8601(m["close_time"]))
    pick = live[0]
    market = KalshiMarket(
        ticker=pick["ticker"],
        title=pick.get("title", ""),
        close_time_iso=pick["close_time"],
    )
    logger.info(
        f"Active market: {market.ticker} expires_in={market.seconds_to_expiry}s"
    )
    return market


def get_top_of_book(client: KalshiClient, ticker: str) -> TopOfBook:
    """Return best YES/NO bid and ask for `ticker`."""
    body = client.get(f"markets/{ticker}/orderbook", params={"depth": 1})
    book = body.get("orderbook_fp") or {}

    yes_bid = _best_bid(book.get("yes_dollars"))
    no_bid = _best_bid(book.get("no_dollars"))

    return TopOfBook(
        yes_bid=yes_bid,
        yes_ask=None if no_bid is None else round(1.0 - no_bid, 4),
        no_bid=no_bid,
        no_ask=None if yes_bid is None else round(1.0 - yes_bid, 4),
    )


def _best_bid(levels: list[list[str]] | None) -> float | None:
    """Highest price in a `[[price, count], ...]` array, or None if empty.

    Kalshi documents these as sorted best-to-worst, but the observed ordering is
    ascending, so take the max rather than trusting either end.
    """
    if not levels:
        return None
    return max(float(price) for price, _count in levels)


def _parse_iso8601(iso: str) -> float:
    """Lenient ISO-8601 parse that tolerates a trailing Z."""
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()
