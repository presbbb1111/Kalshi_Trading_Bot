"""Does the market overreact to a break of serve more in WTA than in ATP?

    python tennis_break_test.py                   # full run
    python tennis_break_test.py --markets 200     # quicker

## The hypothesis, and why it is worth testing

Breaks of serve are far more common in women's tennis than men's. A single
break is therefore weaker evidence in a WTA match than in an ATP one. If the
market prices a WTA break with roughly the same severity it prices an ATP
break, it is overreacting in WTA — and the favourite's marked-down price would
win more often than that price implies.

That is a mechanism, not a pattern someone noticed in a chart, which is why it
is worth a run. ATP is the control: the same rule, same code, same window. An
edge that shows up in both tours is not this effect, it is a bug.

## Guards, each against a specific error this project has already made

* **First touch only.** One observation per market, at the first qualifying
  dip. Sampling every tick weights long matches more heavily and invented a
  +31c "dip edge" here once already.
* **Symmetry.** A match has two markets, one per player, and they resolve
  oppositely. A real overreaction to a fall must be anti-symmetric: if buying
  the faller is +X, buying the riser must be about -X. Both sides showing the
  same sign means the measurement is broken — exactly how an apparent uniform
  +7c tennis edge turned out to be API side-ordering bias.
* **Out of sample.** Parameters are chosen on the older half of the dates and
  tested once on the newer half. The in-sample number is reported too, so the
  gap between them is visible.
* **Fees on entry.** `0.07 * n * p * (1-p)`, rounded up, charged on every
  simulated trade.
* **Edge is measured against the entry price**, not against a win rate. A 60%
  win rate at 60c is nothing. Only `win_rate - entry_price` is edge.
"""
from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

from dotenv import load_dotenv

from src.kalshi_client import KalshiClient
from src.utils import get_logger

load_dotenv()
logger = get_logger(__name__)

TOURS = {"WTA": "KXWTAMATCH", "ATP": "KXATPMATCH"}
FAV_LEVELS = (0.60, 0.65, 0.70)        # price the side must have reached first
DROPS = (0.10, 0.15, 0.20)             # absolute fall from that running high
MIN_TRADES = 20


def fee(count: float, price: float) -> float:
    return math.ceil(0.07 * count * price * (1.0 - price) * 10000) / 10000


@dataclass
class Obs:
    """One market's first qualifying trigger."""
    ticker: str
    day: str
    entry: float
    won: bool
    direction: str          # "fall" (bought the faller) or "rise" (bought the riser)

    @property
    def pnl(self) -> float:
        n = 1.0
        return (n * (1.0 - self.entry) if self.won else -n * self.entry) - fee(n, self.entry)


def price_path(client: KalshiClient, ticker: str) -> list[float]:
    out: list[tuple[int, float]] = []
    cursor = None
    for _ in range(8):
        params = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        try:
            body = client.get("markets/trades", params=params)
        except Exception:
            break
        rows = body.get("trades") or []
        if not rows:
            break
        for r in rows:
            try:
                out.append((int(r.get("ts") or 0), float(r.get("yes_price_dollars") or 0)))
            except (TypeError, ValueError):
                continue
        cursor = body.get("cursor")
        if not cursor:
            break
    out.sort(key=lambda x: x[0])
    return [p for _, p in out if 0 < p < 1]


def first_trigger(path: list[float], fav: float, drop: float, *, rising: bool) -> float | None:
    """Entry price at the first qualifying move, or None.

    `rising=False` is the hypothesis: a side that reached `fav` and then fell
    `drop`. `rising=True` is its mirror, used only as the symmetry check.
    """
    if rising:
        running = min(path)
        low = path[0]
        for price in path[1:]:
            low = min(low, price)
            # a side that was cheap and has climbed by `drop`
            if low <= 1.0 - fav and price - low >= drop:
                return price
        return None

    high = path[0]
    for price in path[1:]:
        high = max(high, price)
        if high >= fav and high - price >= drop:
            return price
    return None


def collect(client: KalshiClient, series: str, limit: int) -> list[tuple[str, str, list[float], bool]]:
    markets: list[dict] = []
    cursor = None
    while len(markets) < limit:
        params = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        body = client.get("markets", params=params)
        rows = body.get("markets") or []
        if not rows:
            break
        markets.extend(rows)
        cursor = body.get("cursor")
        if not cursor:
            break

    out = []
    for m in markets[:limit]:
        result = (m.get("result") or "").lower()
        if result not in ("yes", "no"):
            continue
        parts = m["ticker"].split("-")
        day = parts[1][:7] if len(parts) > 1 else ""
        path = price_path(client, m["ticker"])
        if len(path) < MIN_TRADES:
            continue
        out.append((m["ticker"], day, path, result == "yes"))
    return out


def score(obs: list[Obs]) -> tuple[int, float, float, float, float]:
    """(n, win_rate, mean_entry, edge, pnl_per_trade)."""
    if not obs:
        return 0, 0.0, 0.0, 0.0, 0.0
    n = len(obs)
    wins = sum(1 for o in obs if o.won) / n
    entry = statistics.mean(o.entry for o in obs)
    pnl = statistics.mean(o.pnl for o in obs)
    return n, wins, entry, wins - entry, pnl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markets", type=int, default=600, help="settled markets per tour")
    args = ap.parse_args()

    client = KalshiClient(demo=False)
    data = {}
    for tour, series in TOURS.items():
        data[tour] = collect(client, series, args.markets)
        logger.info(f"{tour}: {len(data[tour])} usable markets")

    # Split by date so the out-of-sample test is on later matches, not a
    # random subset of the same ones.
    for tour in TOURS:
        days = sorted({d for _, d, _, _ in data[tour]})
        cut = days[len(days) // 2] if days else ""
        data[tour] = [(t, d, p, w, d < cut) for t, d, p, w in data[tour]]

    print(f"\n{'tour':5} {'fav':>5} {'drop':>5} {'half':>6} {'n':>5} {'win%':>7} {'entry':>7} "
          f"{'EDGE':>8} {'pnl/trade':>10}")
    results = defaultdict(dict)
    for tour in TOURS:
        for fav in FAV_LEVELS:
            for drop in DROPS:
                for in_sample in (True, False):
                    obs = []
                    for ticker, day, path, won, is_early in data[tour]:
                        if is_early != in_sample:
                            continue
                        entry = first_trigger(path, fav, drop, rising=False)
                        if entry is None:
                            continue
                        obs.append(Obs(ticker, day, entry, won, "fall"))
                    n, wr, en, edge, pnl = score(obs)
                    if n < 10:
                        continue
                    half = "IN" if in_sample else "OUT"
                    results[(tour, fav, drop)][half] = (n, edge, pnl)
                    print(f"{tour:5} {fav:5.2f} {drop:5.2f} {half:>6} {n:5d} {100*wr:6.1f}% "
                          f"{en:7.3f} {edge:+8.3f} {pnl:+10.4f}")

    # --- symmetry check -------------------------------------------------
    print("\nSYMMETRY CHECK (buying the riser instead — must be ~the mirror):")
    print(f"{'tour':5} {'fav':>5} {'drop':>5} {'n':>5} {'EDGE(fall)':>11} {'EDGE(rise)':>11} {'sum':>8}")
    for tour in TOURS:
        for fav in FAV_LEVELS:
            for drop in DROPS:
                fall, rise = [], []
                for ticker, day, path, won, _ in data[tour]:
                    e = first_trigger(path, fav, drop, rising=False)
                    if e is not None:
                        fall.append(Obs(ticker, day, e, won, "fall"))
                    e2 = first_trigger(path, fav, drop, rising=True)
                    if e2 is not None:
                        rise.append(Obs(ticker, day, e2, won, "rise"))
                if len(fall) < 10 or len(rise) < 10:
                    continue
                _, _, _, ef, _ = score(fall)
                _, _, _, er, _ = score(rise)
                print(f"{tour:5} {fav:5.2f} {drop:5.2f} {len(fall):5d} {ef:+11.3f} {er:+11.3f} "
                      f"{ef + er:+8.3f}")
    print("\nA real overreaction is anti-symmetric: the two edges should roughly")
    print("cancel. Both the same sign means the measurement is wrong, not the market.")


if __name__ == "__main__":
    main()
