"""Near-certainty scanning: find contracts priced 0.97-0.99 worth buying.

## The strategy

Buy an outcome the market has already decided but not yet priced at 1.00, and
collect the last cent or two. Measured over 1,919 settled markets across 71
dates and 7 sports, taker buys in this band returned +1.52% with a 99.87% win
rate, profitable on 69 of 71 dates (see `nearcert_backtest.py`).

## Why the band is 0.97-0.99 and the spread cap matters more than the price

Kalshi's taker fee is `0.07 * count * price * (1 - price)`, which collapses at
the extremes — at 0.99 it is ~0.07% of stake, which is why a 1c gross gain
survives it at all. But price alone is a trap. A market quoting bid 0.01 /
ask 0.98 has *both* sides inside the band:

    KXINXU-26OCT12H1000-T7894.9999   buy YES @ 0.98
    KXINXU-26OCT12H1000-T7894.9999   buy NO  @ 0.99

That is not a decided outcome, it is an empty book with token quotes at each
end. 516 of 516 "opportunities" found on demo in a first pass were this. The
spread cap, not the price filter, is what separates a real near-certainty from
an untraded market.

## What is still unproven

The backtest measures trades *other people chose to make*. They watched a game,
judged it locked, and hit the ask — and also declined plenty of 0.99 offers that
looked wrong to them. A bot taking every 0.99 offer gets both groups, and the
declined ones are disproportionately the losers. Nothing in the historical data
can size that gap, which is why `QuotePersistence` below measures how long these
quotes actually survive: if they vanish in milliseconds, the question is moot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.kalshi_client import KalshiClient
from src.kalshi_markets import _parse_iso8601
from src.utils import get_logger

logger = get_logger(__name__)

# Series the backtest cleared, by measured ROI. KXNCAAF1H (-0.17%) and
# KXNCAAF3Q (-1.80%) are deliberately absent: they were the only two of fifteen
# that lost, and both are college-football part-game markets where the "decided"
# read is weakest — a 3rd-quarter total can be overturned by one drive.
SERIES = (
    "KXWTAMATCH",      # +1.64%
    "KXATPMATCH",      # +1.62%
    "KXNBAGAME",       # +1.60%
    "KXMLBGAME",       # +1.48%
    "KXNCAAFGAME",     # +1.39%
    "KXNFLGAME",       # +1.31%
    "KXNFLSPREAD",     # +1.44% (as KXNCAAFSPREAD proxy; NFL variant same shape)
    "KXNFLTOTAL",
    "KXNHLTOTAL",      # +1.05%
    "KXNHLGAME",
)

PRICE_MIN = 0.97
PRICE_MAX = 0.99
MAX_SPREAD = 0.02       # the empty-book guard; see module docstring
FUNDED_SHARD = 0        # demo balance lives only on exchange_index 0

# How close to settlement a market must be. This is the difference between the
# strategy and a superficially identical mistake.
#
# A contract at 0.97 with a minute left is an outcome that has happened. The
# same 0.97 twelve days before kickoff is a heavy favourite, which loses
# regularly. Both pass a price filter; only the first is what the backtest
# measured. Omitting this, the bot bought NFL spreads 5-12 days out and went
# 8-for-8 on pre-game favourites — a result that looked like confirmation and
# tested nothing.
#
# Measured over 32,664 real near-certainty trades: median lead time **0.14h**
# (8 minutes), p90 **0.78h** (47 minutes). 95.5% fall inside one hour.
#
#   lead time   trades    win%      ROI
#   <1h          31185  99.97%   +1.64%
#   1-6h          1454 100.00%   +1.52%
#   6-24h           21 100.00%   +2.15%
#   >3d              4 100.00%   +2.88%
#
# The long-dated rows are not evidence of an edge out there — they are 21 and 4
# observations, which is what "nobody does this" looks like in a table. The cap
# is set at the 1-6h band's outer edge rather than 1h, to allow for polling lag
# and pre-game markets that lock early, but anything beyond it is a different
# bet with no support behind it.
MAX_HOURS_TO_CLOSE = 6.0


@dataclass(frozen=True)
class NearCertQuote:
    """A contract currently offered at a near-certainty price."""

    ticker: str
    side: str               # "YES" or "NO" — the outcome we would be buying
    price: float            # what we would pay per contract
    spread: float
    close_time: str
    exchange_index: int
    hours_left: float = 0.0

    @property
    def book_side(self) -> str:
        """Kalshi's single-book side for this buy.

        Buying YES is a bid; buying NO is an ask at the mirror price. Getting
        this backwards opens exactly the opposite position.
        """
        return "bid" if self.side == "YES" else "ask"

    @property
    def limit_price(self) -> float:
        """Order price in YES terms, which is what the API expects."""
        return self.price if self.side == "YES" else round(1.0 - self.price, 4)

    @property
    def key(self) -> str:
        return f"{self.ticker}:{self.side}"


def expiry_iso(market: dict) -> str:
    """When this market's outcome is actually expected to be known.

    `close_time` means two different things depending on status, which is a
    trap worth stating plainly:

      * **settled** market -> `close_time` is rewritten to the real close
        (`2026-10-18T20:31`), so it is correct to measure history against.
      * **open** market -> `close_time` is the far-future *deadline*
        (`2026-10-21T00:20`, equal to `latest_expiration_time`), while
        `expected_expiration_time` (`2026-10-19T06:20`) is the real one.

    The backtest ran on settled markets and so measured lead times correctly
    (median 8 minutes). The live scanner ran on open markets and read the same
    field, concluded every NFL game was days away, and returned zero qualifying
    quotes for 2,250 consecutive ticks on an NFL Sunday.

    `expected_expiration_time` first, falling back to `close_time` for markets
    that do not publish it.
    """
    return market.get("expected_expiration_time") or market.get("close_time") or ""


def hours_to_close(close_time_iso: str, *, now: float | None = None) -> float | None:
    """Hours until settlement, or None if the close time can't be read."""
    if not close_time_iso:
        return None
    try:
        closes = _parse_iso8601(close_time_iso)
    except Exception:
        return None
    if not closes:
        return None
    return (closes - (time.time() if now is None else now)) / 3600.0


def _quotes_from(
    market: dict, *, funded_shard_only: bool, now: float | None = None
) -> list[NearCertQuote]:
    if funded_shard_only and market.get("exchange_index") != FUNDED_SHARD:
        return []

    # Reject anything too far from settlement before looking at price at all.
    # An unreadable close time is also rejected: trading blind on *when* is the
    # exact failure this guard exists to stop.
    lead = hours_to_close(expiry_iso(market), now=now)
    if lead is None or not 0 <= lead <= MAX_HOURS_TO_CLOSE:
        return []
    try:
        bid = float(market.get("yes_bid_dollars") or 0)
        ask = float(market.get("yes_ask_dollars") or 0)
    except (TypeError, ValueError):
        return []
    if not 0 < bid < ask < 1:
        return []
    spread = round(ask - bid, 4)
    if spread > MAX_SPREAD:
        return []

    out = []
    # Buying YES costs the ask; buying NO costs 1 - bid.
    for side, price in (("YES", ask), ("NO", round(1.0 - bid, 4))):
        if PRICE_MIN <= price <= PRICE_MAX:
            out.append(
                NearCertQuote(
                    ticker=market.get("ticker", ""),
                    side=side,
                    price=price,
                    spread=spread,
                    close_time=expiry_iso(market),
                    exchange_index=market.get("exchange_index", -1),
                    hours_left=round(lead, 3),
                )
            )
    return out


def scan(
    client: KalshiClient,
    *,
    series: tuple[str, ...] = SERIES,
    funded_shard_only: bool = False,
) -> list[NearCertQuote]:
    """Every near-certainty quote currently on offer in `series`.

    Scans by series rather than listing all open markets: there are more than
    30,000 of those, and a paged sweep both misses the tail and costs far more
    requests than asking for the ten series that matter.
    """
    out: list[NearCertQuote] = []
    for ticker_series in series:
        try:
            markets = client.get(
                "markets",
                params={"series_ticker": ticker_series, "status": "open", "limit": 200},
            ).get("markets") or []
        except Exception as exc:
            logger.debug(f"{ticker_series}: {exc!r}")
            continue
        for market in markets:
            out.extend(_quotes_from(market, funded_shard_only=funded_shard_only))
    out.sort(key=lambda q: -q.price)
    return out


@dataclass
class QuotePersistence:
    """How long near-certainty quotes stay on the book.

    This is the measurement the backtest structurally cannot make. The history
    says these trades win; it cannot say whether *we* could have got them. If a
    0.99 ask survives for seconds, a polling bot can take it. If it is gone
    between ticks, the edge is real and unreachable, and the honest answer is to
    stop — the same gap that made the crypto bot read +$124 on paper and -$156
    on real fills.
    """

    first_seen: dict[str, float] = field(default_factory=dict)
    lifetimes: list[float] = field(default_factory=list)
    observations: int = 0

    def observe(self, quotes: list[NearCertQuote]) -> None:
        now = time.time()
        live = {q.key for q in quotes}
        for key in live:
            self.first_seen.setdefault(key, now)
            self.observations += 1
        for key in [k for k in self.first_seen if k not in live]:
            self.lifetimes.append(now - self.first_seen.pop(key))

    def summary(self) -> str:
        if not self.lifetimes:
            return (
                f"persistence: {len(self.first_seen)} quotes still live, "
                "none expired yet"
            )
        ordered = sorted(self.lifetimes)
        median = ordered[len(ordered) // 2]
        return (
            f"persistence: {len(self.lifetimes)} expired | "
            f"median {median:.0f}s | min {ordered[0]:.0f}s | max {ordered[-1]:.0f}s | "
            f"{len(self.first_seen)} still live"
        )
