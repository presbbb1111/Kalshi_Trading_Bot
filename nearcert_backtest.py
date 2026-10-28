"""Backtest buying near-certainty contracts (0.97-0.99) on in-play sports.

    python nearcert_backtest.py                 # default series, 40 markets each
    python nearcert_backtest.py --limit 100     # deeper history

## Where this came from

Reading the one fully public profile on Kalshi's social leaderboard
(`weatherman.allday`, +$113,927 over 515,891 trades) showed a single repeated
action: *Bought Yes at 99c*, in 1-175 contract clips, on live in-play sports
micro-markets. 90% of their profit is Sports; weather is 0.5% and crypto loses.

This script tests that strategy directly against settled markets, using the
public (anonymous) trade feed. Identity is not needed: every trade carries
ticker, price, size and taker side, and the settled market carries the result.

## The two traps this is built to avoid

1. **Clustering.** A single market can contain 100 near-certainty trades, which
   all win or all lose together. Treating them as 100 independent observations
   inflates significance enormously — the same error that invented a +31c tennis
   "edge" earlier in this project. `--by-market` reports the per-market
   distribution and a t-statistic over markets, which is the honest unit.

2. **Fees at the extremes.** Kalshi's taker fee is `0.07 * count * price *
   (1 - price)`, which collapses near 0 and 1. That is *why* 99c works at all: a
   1c gross gain only survives because the fee there is ~0.07% of stake. Any
   backtest that ignores fees will find an edge at every price.

## What it does NOT establish

It assumes you get the fill at the observed price. These are trades that
actually happened, so somebody got them — but whether *you* can, polling from a
laptop against participants doing 515,891 trades, is the open question. That gap
is exactly what made the crypto bot report +$124 on paper and -$156 on real
fills.
"""
from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass, field

from dotenv import load_dotenv

from src.kalshi_client import KalshiClient
from src.utils import get_logger

load_dotenv()
logger = get_logger(__name__)

# In-play micro-markets: the series where the near-certainty volume actually is,
# measured off the live trade feed (218 trades >=0.97 in a 96-second window, with
# college-football half/quarter markets the most frequent by trade count).
DEFAULT_SERIES = (
    "KXNCAAF1H",
    "KXNCAAF1HSPREAD",
    "KXNCAAF2QSPREAD",
    "KXNCAAF1HTOTAL",
    "KXNCAAFSPREAD",
    "KXNHLTOTAL",
    "KXNCAAFGAME",
)

PRICE_MIN = 0.97
PRICE_MAX = 0.995


def taker_fee(count: float, price: float) -> float:
    """Kalshi's taker fee, rounded up to the next $0.0001."""
    return math.ceil(0.07 * count * price * (1.0 - price) * 10000) / 10000


@dataclass
class MarketResult:
    ticker: str
    day: str = ""
    trades: int = 0
    wins: int = 0
    pnl: float = 0.0
    stake: float = 0.0
    prices: list[float] = field(default_factory=list)


def market_day(ticker: str) -> str:
    """The `26OCT03` date segment of a ticker, or '' if absent.

    Sampling and reporting are both keyed on this. The markets endpoint returns
    newest-first, so taking the first N without regard to date collects one
    weekend and calls it history — which is what the first run of this script
    did, and the reason the result needed re-checking.
    """
    parts = ticker.split("-")
    return parts[1][:7] if len(parts) > 1 else ""


def _taker_price(trade: dict) -> tuple[str, float] | None:
    """(side the taker bought, price they paid). None if unreadable.

    The feed quotes everything in YES terms, so a taker buying NO paid
    `1 - yes_price`. Reading `yes_price_dollars` unconditionally would score
    every NO trade at its mirror price and invert the result.
    """
    try:
        yes_price = float(trade.get("yes_price_dollars") or 0)
    except (TypeError, ValueError):
        return None
    side = trade.get("taker_outcome_side") or ""
    if side not in ("yes", "no"):
        return None
    return side, (yes_price if side == "yes" else round(1.0 - yes_price, 4))


def all_trades(client: KalshiClient, ticker: str, max_pages: int = 40) -> list[dict]:
    """Every trade for a market, not just the most recent page.

    `markets/trades` returns newest-first, so a bare `limit=1000` on a busy
    market returns only its final minutes — the window where the outcome is
    already decided and a near-certainty bet is almost never wrong. Measured on
    32 settled markets, 30 were truncated by the cap and full pagination found
    39,575 near-certainty trades against 4,164 capped, nearly 10x.

    (The truncation turned out to flatter the strategy *less*, not more: the
    fully-paginated ROI came in at +1.83% against +1.39% capped. It is fixed
    anyway, because a sample that silently drops 90% of its observations is not
    one to reason from in either direction.)
    """
    out: list[dict] = []
    cursor: str | None = None
    for _ in range(max_pages):
        params: dict = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        try:
            body = client.get("markets/trades", params=params)
        except Exception:
            break
        rows = body.get("trades") or []
        if not rows:
            break
        out.extend(rows)
        cursor = body.get("cursor")
        if not cursor:
            break
    return out


def settled_markets(
    client: KalshiClient, series: str, pages: int, per_day: int
) -> list[dict]:
    """Settled markets for `series`, spread across calendar days.

    Pages through the whole history, then keeps at most `per_day` markets from
    each date. Without the per-day cap the sample is whatever settled most
    recently, which collapses to a single weekend of one sport.
    """
    found: list[dict] = []
    cursor: str | None = None
    for _ in range(pages):
        params: dict = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            body = client.get("markets", params=params)
        except Exception as exc:
            logger.warning(f"{series}: {exc!r}")
            break
        rows = body.get("markets") or []
        if not rows:
            break
        found.extend(rows)
        cursor = body.get("cursor")
        if not cursor:
            break

    by_day: dict[str, list[dict]] = {}
    for market in found:
        by_day.setdefault(market_day(market.get("ticker", "")), []).append(market)
    out: list[dict] = []
    for day in sorted(by_day):
        out.extend(by_day[day][:per_day])
    return out


def scan(
    client: KalshiClient,
    series: tuple[str, ...],
    per_day: int = 8,
    pages: int = 8,
) -> list[MarketResult]:
    out: list[MarketResult] = []
    for ticker_series in series:
        markets = settled_markets(client, ticker_series, pages, per_day)
        logger.info(f"{ticker_series}: {len(markets)} settled markets sampled")

        for market in markets:
            result = (market.get("result") or "").lower()
            if result not in ("yes", "no"):
                continue
            trades = all_trades(client, market["ticker"])

            row = MarketResult(ticker=market["ticker"], day=market_day(market["ticker"]))
            for trade in trades:
                parsed = _taker_price(trade)
                if parsed is None:
                    continue
                side, price = parsed
                try:
                    count = float(trade.get("count_fp") or 0)
                except (TypeError, ValueError):
                    continue
                if count <= 0 or not PRICE_MIN <= price <= PRICE_MAX:
                    continue

                won = result == side
                row.trades += 1
                row.wins += won
                row.stake += count * price
                row.prices.append(price)
                row.pnl += (
                    count * (1.0 - price) if won else -count * price
                ) - taker_fee(count, price)

            if row.trades:
                out.append(row)
    return out


def report(rows: list[MarketResult]) -> None:
    if not rows:
        print("No near-certainty trades found.")
        return

    trades = sum(r.trades for r in rows)
    wins = sum(r.wins for r in rows)
    pnl = sum(r.pnl for r in rows)
    stake = sum(r.stake for r in rows)

    print(f"markets {len(rows)}   trades {trades}   staked ${stake:,.2f}")
    print(f"win rate {100 * wins / trades:.2f}%   P&L ${pnl:+,.2f}   ROI {100 * pnl / stake:+.3f}%")

    # Per-market is the honest unit: trades inside one market share an outcome.
    per_market = [r.pnl for r in rows]
    mean = statistics.mean(per_market)
    sd = statistics.pstdev(per_market)
    t_stat = mean / (sd / math.sqrt(len(rows))) if sd else float("inf")
    profitable = sum(1 for p in per_market if p > 0)

    print()
    print("clustered by market (trades within a market are not independent):")
    print(f"  profitable   {profitable}/{len(rows)} ({100 * profitable / len(rows):.0f}%)")
    print(f"  mean         ${mean:+.2f}   sd ${sd:.2f}")
    print(f"  t-stat       {t_stat:.2f}  (>2 to be meaningful)")
    ordered = sorted(per_market)
    print(
        f"  min ${ordered[0]:+.2f} | p10 ${ordered[len(ordered) // 10]:+.2f} | "
        f"median ${statistics.median(per_market):+.2f} | "
        f"p90 ${ordered[9 * len(ordered) // 10]:+.2f} | max ${ordered[-1]:+.2f}"
    )
    print()
    print("  The tail is the whole risk: most markets earn pennies, a few lose a lot.")

    _by_day(rows)
    _by_series(rows)


def _by_day(rows: list[MarketResult]) -> None:
    """Per-date results — the test of whether this is one lucky weekend.

    A real edge shows up on most days. One that lives in a handful of dates is a
    story about those dates, not about the strategy.
    """
    days: dict[str, list[MarketResult]] = {}
    for row in rows:
        days.setdefault(row.day, []).append(row)
    if len(days) < 2:
        print("\nonly one date in sample — cannot test stability over time")
        return

    print(f"\nby date ({len(days)} dates):")
    print(f"  {'date':>9} {'mkts':>5} {'trades':>7} {'win%':>7} {'P&L':>11} {'ROI':>8}")
    positive = 0
    for day in sorted(days, key=lambda d: (d[:2], _MONTHS.get(d[2:5], 0), d[5:])):
        group = days[day]
        trades = sum(r.trades for r in group)
        stake = sum(r.stake for r in group)
        pnl = sum(r.pnl for r in group)
        wins = sum(r.wins for r in group)
        positive += pnl > 0
        print(
            f"  {day:>9} {len(group):5d} {trades:7d} {100 * wins / trades:6.2f}% "
            f"${pnl:+10.2f} {100 * pnl / stake if stake else 0:+7.2f}%"
        )
    print(f"  profitable on {positive}/{len(days)} dates")


def _by_series(rows: list[MarketResult]) -> None:
    """Per-series results — whether the edge is one sport or the mechanism."""
    series: dict[str, list[MarketResult]] = {}
    for row in rows:
        series.setdefault(row.ticker.split("-")[0], []).append(row)
    print(f"\nby series ({len(series)}):")
    print(f"  {'series':<22} {'mkts':>5} {'trades':>7} {'win%':>7} {'P&L':>11} {'ROI':>8}")
    for name in sorted(series, key=lambda s: -sum(r.pnl for r in series[s])):
        group = series[name]
        trades = sum(r.trades for r in group)
        stake = sum(r.stake for r in group)
        pnl = sum(r.pnl for r in group)
        wins = sum(r.wins for r in group)
        print(
            f"  {name:<22} {len(group):5d} {trades:7d} {100 * wins / trades:6.2f}% "
            f"${pnl:+10.2f} {100 * pnl / stake if stake else 0:+7.2f}%"
        )


_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-day", type=int, default=8,
        help="max settled markets sampled per calendar date, per series",
    )
    parser.add_argument(
        "--pages", type=int, default=8, help="pages of settled-market history to page through"
    )
    parser.add_argument("--series", nargs="*", default=list(DEFAULT_SERIES))
    args = parser.parse_args()

    client = KalshiClient(demo=False)
    report(scan(client, tuple(args.series), args.per_day, args.pages))


if __name__ == "__main__":
    main()
