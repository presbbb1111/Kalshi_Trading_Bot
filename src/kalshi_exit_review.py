"""Measure whether exiting early beat holding to settlement.

## Why this module exists

The bot's stop-loss and take-profit rules were added on evidence borrowed from
BTC 15-minute markets, and then ran for a week with no way to check them. An
exited position never reaches `GET /portfolio/settlements`, so every P&L report
built on that feed silently excluded exactly the trades under test.

## Why not reconstruct from fills

The obvious fix — add up the fills — is a trap. Kalshi's fill records describe a
short YES position as `action="sell", side="no"`, so a naive pass reads an
opening trade as proceeds and reports large fictional profits. A first attempt
at this produced +$289 on an account that had lost money.

`GET /portfolio/positions` carries `realized_pnl_dollars`, which is the
exchange's own accounting for the round trip. That is used instead: it needs no
reconstruction and cannot get the sign backwards.

## The counterfactual

Realized P&L alone doesn't answer the question. "The stop lost $0.56" is only
bad if holding would have lost less. So for each exited position this recovers
the entry from the opening fill, looks up how the market actually resolved, and
computes what the same position would have returned held to settlement.

## The journal

`GET /portfolio/positions` returns only *current* positions — 11 rows on an
account with 252 settlements. The 21 exits taken between 2026-08-25 and 08-27
are simply gone from it, and cannot be recovered. So exits are also written to
an append-only CSV as they happen; `collect_exits` covers the live window, the
journal covers everything after 2026-09-03.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.kalshi_client import KalshiClient
from src.utils import get_logger

logger = get_logger(__name__)

EXIT_CSV = "weather_exits.csv"
EXIT_FIELDS = [
    "exited_utc", "ticker", "rule", "contracts", "entry_price",
    "outcome_side", "exit_price", "gain_pct", "result", "held_pnl", "benefit",
]


@dataclass(frozen=True)
class ExitedTrade:
    """One position that was closed before the market settled."""

    ticker: str
    contracts: float
    entry_price: float          # cost per contract, in the direction held
    outcome_side: str           # "yes" or "no" — the side we were long
    realized_pnl: float         # exchange's own number, fees included
    result: str | None          # "yes" / "no" / None if not yet resolved

    @property
    def cost(self) -> float:
        return self.contracts * self.entry_price

    @property
    def held_pnl(self) -> float | None:
        """What this position would have returned held to settlement."""
        if self.result not in ("yes", "no"):
            return None
        won = self.result == self.outcome_side
        return round(self.contracts * (1.0 if won else 0.0) - self.cost, 4)

    @property
    def benefit(self) -> float | None:
        """How much the early exit gained (positive) or cost (negative)."""
        held = self.held_pnl
        return None if held is None else round(self.realized_pnl - held, 4)


def _paged(client: KalshiClient, path: str, key: str) -> Iterator[dict[str, Any]]:
    """Walk every page of a portfolio endpoint."""
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        body = client.get(path, params=params, authed=True)
        rows = body.get(key) or []
        if not rows:
            return
        yield from rows
        cursor = body.get("cursor")
        if not cursor:
            return


def _opening_fill(fills: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The earliest fill for a ticker — the one that established the position."""
    if not fills:
        return None
    return min(fills, key=lambda f: f.get("ts") or 0)


def _entry(fill: dict[str, Any]) -> tuple[str, float]:
    """(side we went long, price paid per contract) from an opening fill.

    `outcome_side` names the outcome the position pays on, and the matching
    price field is what it cost. Reading `yes_price_dollars` unconditionally is
    the sign error this module exists to avoid.
    """
    side = (fill.get("outcome_side") or fill.get("side") or "").lower()
    field = "no_price_dollars" if side == "no" else "yes_price_dollars"
    return side, float(fill.get(field) or 0)


def collect_exits(client: KalshiClient, prefix: str = "KXHIGH") -> list[ExitedTrade]:
    """Every position closed by an order rather than by settlement."""
    settled = {
        s.get("ticker")
        for s in _paged(client, "portfolio/settlements", "settlements")
        if (s.get("ticker") or "").startswith(prefix)
    }

    fills_by_ticker: dict[str, list[dict[str, Any]]] = {}
    for f in _paged(client, "portfolio/fills", "fills"):
        ticker = f.get("ticker") or ""
        if ticker.startswith(prefix):
            fills_by_ticker.setdefault(ticker, []).append(f)

    out: list[ExitedTrade] = []
    for pos in _paged(client, "portfolio/positions", "market_positions"):
        ticker = pos.get("ticker") or ""
        if not ticker.startswith(prefix):
            continue
        # Flat, with money changed hands, and never settled => closed by us.
        if float(pos.get("position_fp") or 0) != 0:
            continue
        realized = float(pos.get("realized_pnl_dollars") or 0)
        if realized == 0:
            continue
        if ticker in settled:
            continue

        opening = _opening_fill(fills_by_ticker.get(ticker, []))
        if opening is None:
            logger.debug(f"{ticker}: closed with no fills on record — skipped")
            continue
        side, price = _entry(opening)
        contracts = float(opening.get("count_fp") or opening.get("count") or 0)
        if contracts <= 0 or not 0 < price < 1:
            continue

        out.append(
            ExitedTrade(
                ticker=ticker,
                contracts=contracts,
                entry_price=price,
                outcome_side=side,
                realized_pnl=round(realized - float(pos.get("fees_paid_dollars") or 0), 4),
                result=_market_result(client, ticker),
            )
        )
    return out


_RESULT_CACHE: dict[str, str | None] = {}


def _market_result(client: KalshiClient, ticker: str) -> str | None:
    """How the market resolved, or None if it hasn't yet.

    Needed because an exited position is absent from the settlement feed, so the
    outcome has to come from the market itself.
    """
    if ticker in _RESULT_CACHE:
        return _RESULT_CACHE[ticker]
    try:
        market = client.get(f"markets/{ticker}").get("market") or {}
        result = (market.get("result") or "").lower() or None
    except Exception as exc:
        logger.debug(f"{ticker}: could not read result ({exc!r})")
        return None
    # Only a real outcome is cached. Caching the None from a market that simply
    # hasn't resolved yet would pin it as unresolvable for the life of the
    # process, and this runs for days at a time.
    if result in ("yes", "no"):
        _RESULT_CACHE[ticker] = result
        return result
    return None


class ExitJournal:
    """Append-only record of every exit, written at the moment it is taken.

    Exists because the exchange forgets: a position that was closed and whose
    market later settled disappears from `portfolio/positions` entirely, taking
    its realized P&L with it. Anything not written down here is unmeasurable
    later, no matter how the API is queried.
    """

    def __init__(self, path: str | Path = EXIT_CSV) -> None:
        self.path = Path(path)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=EXIT_FIELDS).writeheader()

    def rows(self) -> list[dict[str, str]]:
        with self.path.open(newline="") as fh:
            return list(csv.DictReader(fh))

    def record(
        self,
        ticker: str,
        rule: str,
        contracts: float,
        entry_price: float,
        outcome_side: str,
        exit_price: float,
        gain_pct: float,
    ) -> None:
        """Write one exit. `result` and the counterfactual are filled in later."""
        with self.path.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=EXIT_FIELDS).writerow(
                {
                    "exited_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "ticker": ticker,
                    "rule": rule,
                    "contracts": f"{contracts:.2f}",
                    "entry_price": f"{entry_price:.4f}",
                    "outcome_side": outcome_side,
                    "exit_price": f"{exit_price:.4f}",
                    "gain_pct": f"{gain_pct:.4f}",
                    "result": "",
                    "held_pnl": "",
                    "benefit": "",
                }
            )

    def resolve(self, client: KalshiClient) -> int:
        """Fill in outcomes for exits whose markets have since settled."""
        rows = self.rows()
        pending = [r for r in rows if not r["result"]]
        if not pending:
            return 0

        filled = 0
        for row in pending:
            result = _market_result(client, row["ticker"])
            if result not in ("yes", "no"):
                continue
            contracts = float(row["contracts"])
            cost = contracts * float(row["entry_price"])
            won = result == row["outcome_side"]
            held = round(contracts * (1.0 if won else 0.0) - cost, 4)
            # What the exit actually banked, from the price it was filled at.
            exited = round(contracts * float(row["exit_price"]) - cost, 4)
            row["result"] = result
            row["held_pnl"] = f"{held:+.4f}"
            row["benefit"] = f"{exited - held:+.4f}"
            filled += 1

        if filled:
            with self.path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=EXIT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"Exit journal: resolved {filled} exit(s)")
        return filled

    def summary(self) -> str:
        """Per-rule verdict: did this exit rule beat holding?"""
        done = [r for r in self.rows() if r["benefit"]]
        if not done:
            return "Exit journal: no resolved exits yet"

        by_rule: dict[str, list[float]] = {}
        for r in done:
            by_rule.setdefault(r["rule"], []).append(float(r["benefit"]))
        lines = [f"Exit journal: {len(done)} resolved"]
        for rule, benefits in sorted(by_rule.items()):
            total = sum(benefits)
            better = sum(1 for b in benefits if b > 0)
            lines.append(
                f"  {rule:12s} n={len(benefits):3d}  net ${total:+7.2f}  "
                f"beat holding {better}/{len(benefits)}"
            )
        return "\n".join(lines)


def summarize(trades: list[ExitedTrade]) -> str:
    """Plain-language verdict on whether the exit rules paid for themselves."""
    scored = [t for t in trades if t.benefit is not None]
    if not scored:
        return "No exited positions have resolved yet — nothing to compare."

    realized = sum(t.realized_pnl for t in scored)
    held = sum(t.held_pnl or 0 for t in scored)
    cost = sum(t.cost for t in scored)
    better = sum(1 for t in scored if (t.benefit or 0) > 0)

    lines = [
        f"{len(scored)} exited positions have resolved (cost ${cost:.2f})",
        f"  exited early : ${realized:+.2f}  ({100 * realized / cost:+.1f}% ROI)",
        f"  held instead : ${held:+.2f}  ({100 * held / cost:+.1f}% ROI)",
        f"  exiting was better on {better}/{len(scored)} "
        f"({100 * better / len(scored):.0f}%)",
        f"  net effect of exiting: ${realized - held:+.2f}",
    ]
    unresolved = len(trades) - len(scored)
    if unresolved:
        lines.append(f"  ({unresolved} exited position(s) not yet resolved)")
    return "\n".join(lines)
