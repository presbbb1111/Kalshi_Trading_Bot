"""Kalshi near-certainty bot — buys contracts at 0.97-0.99 on the demo exchange.

    python nearcert_main.py --ticks 5 --tick-seconds 20   # bounded
    python nearcert_main.py                               # until Ctrl-C

## It does two jobs, deliberately

1. **Trades demo.** Real orders on the funded shard, visible in the account.
   Demo books rarely tighten to a genuine near-certainty — one qualifying quote
   against production's 28, when this was built — so expect it to fire rarely.
2. **Measures production.** Every tick it also records how long production's
   near-certainty quotes survive. That is the question the backtest cannot
   answer: the history says these trades win, not that a polling bot could have
   got them. No orders are sent to production.

Job 2 is the research output. Job 1 is what you can watch.

## Sizing: why not Kelly

Kelly on a 0.99 contract with the measured 99.87% win rate asks for ~87% of
bankroll, because it treats that win rate as known. It is not — it is an
estimate from other traders' chosen entries, and the loss is 99x the gain. One
bad fill at full size ends the account. So size is a hard cap in contracts, and
the open-exposure cap bounds the rest.
"""
from __future__ import annotations

import argparse
import math
import signal
import threading
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.kalshi_client import KalshiClient
from src.kalshi_journal import DecisionJournal
from src.kalshi_orders import get_resting_orders, place_limit_order
from src.nearcert import MAX_HOURS_TO_CLOSE, QuotePersistence, SERIES, scan
from src.utils import get_logger

load_dotenv()
logger = get_logger(__name__)

_stop = threading.Event()


def _handle_signal(signum: int, _frame: object) -> None:
    logger.info(f"Received signal {signum} — shutting down after this tick")
    _stop.set()


def taker_fee(count: float, price: float) -> float:
    return math.ceil(0.07 * count * price * (1.0 - price) * 10000) / 10000


def held(client: KalshiClient) -> set[str]:
    """Tickers already held or with a working order — never double up."""
    out: set[str] = set()
    try:
        body = client.get("portfolio/positions", authed=True)
        out |= {
            p["ticker"]
            for p in (body.get("market_positions") or [])
            if p.get("ticker") and float(p.get("position_fp") or 0) != 0
        }
    except Exception as exc:
        logger.warning(f"Could not read positions: {exc!r}")
    out |= {o["ticker"] for o in get_resting_orders(client) if o.get("ticker")}
    return out


def run_tick(
    order_client: KalshiClient,
    data_client: KalshiClient,
    persistence: QuotePersistence,
    journal: DecisionJournal,
    contracts: float,
    max_open: int,
) -> None:
    # --- job 2: measure production, place nothing ------------------------
    live = scan(data_client, funded_shard_only=False)
    persistence.observe(live)
    logger.info(f"production: {len(live)} near-certainty quotes | {persistence.summary()}")

    # --- job 1: trade whatever demo actually offers ----------------------
    offers = scan(order_client, funded_shard_only=True)
    if not offers:
        logger.info("demo: nothing qualifying this tick")
        return

    open_tickers = held(order_client)
    if len(open_tickers) >= max_open:
        logger.info(f"demo: at open-position cap ({max_open}) — not adding")
        return

    for quote in offers:
        if quote.ticker in open_tickers:
            continue
        cost = contracts * quote.price
        fee = taker_fee(contracts, quote.price)
        logger.info(
            f"BUY {quote.side} {quote.ticker} @ {quote.price:.2f} "
            f"(spread {quote.spread:.2f}, closes in {quote.hours_left:.2f}h) — {contracts} contracts, "
            f"${cost:.2f} at risk to win ${contracts * (1 - quote.price) - fee:.2f}"
        )
        # Immediate-or-cancel: the whole point is taking a quote that exists
        # right now. A resting order here would sit until the price moved
        # against us, which is the one direction we cannot afford.
        order = place_limit_order(
            order_client,
            quote.ticker,
            quote.book_side,
            quote.limit_price,
            contracts,
            time_in_force="immediate_or_cancel",
        )
        if order is None:
            logger.warning(f"{quote.ticker}: order rejected")
            continue
        journal.write(
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "market": quote.ticker,
                "close_time": quote.close_time,
                "action": "buy",
                "side": quote.side,
                "size": f"{cost:.2f}",
                "spread": f"{quote.spread:.4f}",
                # No model here: the price *is* the estimate. The only claim is
                # that the market under-prices decided outcomes by ~1.5%.
                "market_prob_yes": f"{quote.price:.4f}",
                "reason": f"near-certainty @ {quote.price:.2f}, spread {quote.spread:.2f}",
            }
        )
        return   # one entry per tick


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi near-certainty bot (demo orders)")
    parser.add_argument("--ticks", type=int, default=None)
    parser.add_argument("--tick-seconds", type=float, default=20.0)
    parser.add_argument(
        "--contracts", type=float, default=5.0,
        help="contracts per entry — a hard cap, not a Kelly fraction (see module docs)",
    )
    parser.add_argument("--max-open", type=int, default=6)
    parser.add_argument("--journal", default="nearcert_trades.csv")
    args = parser.parse_args()

    order_client = KalshiClient(demo=True)    # orders here, visible
    data_client = KalshiClient(demo=False)    # measurement only, never ordered against

    persistence = QuotePersistence()
    journal = DecisionJournal(args.journal)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    logger.info("=" * 68)
    logger.info("Kalshi near-certainty bot — buying 0.97-0.99")
    logger.info(f"  series   : {len(SERIES)} validated (KXNCAAF1H/3Q excluded — both lost)")
    logger.info(f"  lead time: must settle within {MAX_HOURS_TO_CLOSE:.0f}h (95.5% of real trades are <1h)")
    logger.info(f"  orders   : DEMO shard 0 only   measurement: production")
    logger.info(f"  size     : {args.contracts} contracts/entry, max {args.max_open} open")
    logger.info(f"  journal  : {args.journal}")
    logger.info("  NOTE     : backtest +1.52% over 1,919 markets, but fill-")
    logger.info("             feasibility is UNPROVEN — that is what this measures")
    logger.info("=" * 68)

    ticks = 0
    while not _stop.is_set():
        if args.ticks is not None and ticks >= args.ticks:
            break
        try:
            run_tick(
                order_client, data_client, persistence, journal,
                args.contracts, args.max_open,
            )
        except Exception as exc:
            logger.exception(f"Tick failed: {exc!r}")
        ticks += 1
        if _stop.wait(args.tick_seconds):
            break

    logger.info(f"Stopped after {ticks} tick(s). {persistence.summary()}")


if __name__ == "__main__":
    main()
