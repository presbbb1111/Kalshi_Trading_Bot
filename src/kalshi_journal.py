"""Append-only CSV record of every decision the bot makes.

The point of a shadow run is the dataset it produces. A log line scrolling past
is not analysable; this is. One row per tick, whether or not it traded, so the
denominator is visible — "how often did we see an edge at all" matters as much
as "how did the trades do".

Rows are written with the market's close time, so a later pass can join each
decision against how that market actually resolved and score the model.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_JOURNAL = "decisions.csv"

FIELDS = [
    "ts_utc",
    "market",
    "close_time",
    "seconds_to_expiry",
    "action",
    "side",
    "size",
    "yes_bid",
    "yes_ask",
    "spread",
    "market_prob_yes",
    "model_prob_yes",
    "edge",
    "drift_pct",
    "sigma_15m_pct",
    "reason",
]


class DecisionJournal:
    """Writes one row per decision, creating the file with a header if needed."""

    def __init__(self, path: str | Path = DEFAULT_JOURNAL) -> None:
        self.path = Path(path)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()
        logger.info(f"Decision journal created at {self.path}")

    def write(self, row: dict[str, Any]) -> None:
        """Append one decision. Unknown keys are dropped, missing ones blank."""
        clean = {k: row.get(k, "") for k in FIELDS}
        try:
            with self.path.open("a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writerow(clean)
        except OSError as exc:
            # A journal failure must not take down a running bot, but it must
            # be visible — a silent gap in the dataset is worse than no dataset.
            logger.error(f"Could not write decision journal row: {exc!r}")
