"""Does a stop-loss / take-profit beat holding on short-dated Kalshi markets?

    python shortdated_sltp.py                      # default series
    python shortdated_sltp.py --series KXBTC15M

## The question, stated so it can be falsified

A stop-loss cannot manufacture an edge in a fairly-priced market. If a contract
is worth what it costs, every exit rule has the same expected value, minus the
fees each extra trade adds. Exits reshape the *distribution* — smaller losses,
smaller wins — they do not move the mean.

So this does not ask "do stops help?". It asks whether short-dated markets show
**exploitable price dynamics**: does a position that moves in your favour keep
moving (momentum), or snap back (mean reversion)? Only that can make an exit
rule pay. If the answer is neither, the grid below comes out flat-to-negative
everywhere and the idea is dead regardless of how the levels are tuned.

## Method

For each settled market, reconstruct the price path from its public trades,
then simulate buying at each observed price and walking forward:

  * exit at **+TP** if the price rises that far,
  * exit at **-SL** if it falls that far,
  * otherwise settle against the real outcome.

Both are measured against holding the same entry to settlement, so the number
reported is the *difference the exit rule made*, not the raw return.

Fees are charged on entry and on any exit trade — `0.07 * n * p * (1-p)`,
rounded up. Charging only the entry is how a round-trip strategy gets scored as
free money.
"""
from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

from dotenv import load_dotenv

from src.kalshi_client import KalshiClient
from src.utils import get_logger

load_dotenv()
logger = get_logger(__name__)

DEFAULT_SERIES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXGOLDH", "KXSILVERH")
TP_GRID = (0.05, 0.10, 0.15, 0.25)
SL_GRID = (0.05, 0.10, 0.15, 0.25)
# Entries outside this band are ignored: at 0.02 a 10% move is noise on a
# fraction of a cent, and the fee term dominates everything.
ENTRY_MIN, ENTRY_MAX = 0.15, 0.85


def fee(count: float, price: float) -> float:
    return math.ceil(0.07 * count * price * (1.0 - price) * 10000) / 10000


@dataclass
class Sim:
    hold: float = 0.0
    exited: float = 0.0
    n: int = 0
    tp_hits: int = 0
    sl_hits: int = 0


def price_path(client: KalshiClient, ticker: str) -> list[float]:
    """YES prices over time, oldest first."""
    out: list[tuple[int, float]] = []
    cursor = None
    for _ in range(6):
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


def simulate(path: list[float], won: bool, tp: float, sl: float, sim: Sim) -> None:
    """Buy YES at each point on the path; compare exit-rule P&L against holding."""
    n = 1.0
    for i, entry in enumerate(path[:-1]):
        if not ENTRY_MIN <= entry <= ENTRY_MAX:
            continue
        entry_fee = fee(n, entry)
        hold_pnl = (n * (1.0 - entry) if won else -n * entry) - entry_fee

        exit_pnl = None
        for future in path[i + 1:]:
            move = (future - entry) / entry
            if move >= tp:
                exit_pnl = n * (future - entry) - entry_fee - fee(n, future)
                sim.tp_hits += 1
                break
            if move <= -sl:
                exit_pnl = n * (future - entry) - entry_fee - fee(n, future)
                sim.sl_hits += 1
                break
        if exit_pnl is None:
            exit_pnl = hold_pnl

        sim.hold += hold_pnl
        sim.exited += exit_pnl
        sim.n += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", nargs="*", default=list(DEFAULT_SERIES))
    ap.add_argument("--markets", type=int, default=40, help="settled markets per series")
    args = ap.parse_args()

    client = KalshiClient(demo=False)
    grids: dict[tuple[float, float], Sim] = {
        (tp, sl): Sim() for tp in TP_GRID for sl in SL_GRID
    }
    scanned = 0
    for series in args.series:
        try:
            markets = client.get(
                "markets",
                params={"series_ticker": series, "status": "settled", "limit": args.markets},
            ).get("markets") or []
        except Exception as exc:
            logger.warning(f"{series}: {exc!r}")
            continue
        for m in markets:
            result = (m.get("result") or "").lower()
            if result not in ("yes", "no"):
                continue
            path = price_path(client, m["ticker"])
            if len(path) < 10:
                continue
            scanned += 1
            for (tp, sl), sim in grids.items():
                simulate(path, result == "yes", tp, sl, sim)
        logger.info(f"{series}: done ({scanned} markets so far)")

    print(f"\nmarkets simulated: {scanned}")
    if not scanned:
        return
    print("\nDifference the exit rule made vs holding (per simulated entry):")
    print(f"{'TP':>6} {'SL':>6} {'entries':>9} {'hold':>10} {'exited':>10} {'diff':>10} {'TP/SL hits':>12}")
    for (tp, sl), s in sorted(grids.items()):
        if not s.n:
            continue
        print(
            f"{tp:6.0%} {sl:6.0%} {s.n:9d} "
            f"{s.hold / s.n:+10.4f} {s.exited / s.n:+10.4f} "
            f"{(s.exited - s.hold) / s.n:+10.4f} "
            f"{s.tp_hits:>5d}/{s.sl_hits:<6d}"
        )
    best = max(grids.items(), key=lambda kv: (kv[1].exited - kv[1].hold) / max(kv[1].n, 1))
    (tp, sl), s = best
    print(f"\nbest cell: TP {tp:.0%} / SL {sl:.0%} -> {(s.exited - s.hold) / s.n:+.4f} per entry")
    print("A positive 'diff' column means the market has dynamics an exit rule can")
    print("capture. Flat or negative everywhere means it does not, and no amount of")
    print("tuning the levels will change that.")


if __name__ == "__main__":
    main()
