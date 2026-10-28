"""Signal generation for Kalshi's BTC 15-minute market (`KXBTC15M`).

The market resolves YES when the average BRTI over the 60 seconds before close
is at least the average over the 60 seconds before open. So the question at any
moment is: *given BTC has drifted `d` since this window opened, and `t` seconds
remain, how likely is it to still be above the open at close?*

That is modelled here as a random walk: the remaining move is treated as
roughly normal with a standard deviation that shrinks as the window runs out.
Late in a window a small lead is nearly decisive; early on the same lead means
little. The estimate is deliberately computed **without reference to the market
price**, so that comparing the two produces a real edge rather than an
identity — see the note in `generate_signal`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.utils import get_logger

logger = get_logger(__name__)

CYCLE_SECONDS = 900

# Standard deviation of BTC's move over a full 15-minute window, in percent.
#
# This is the most important parameter in the model and it should be MEASURED,
# not assumed — pass `sigma_15m_pct` from `kalshi_brti.estimate_sigma_15m_pct`
# on every cycle. The value below is only a fallback for when the BRTI series is
# unavailable, and it is regime-dependent: 0.09% was measured on 2026-07-26, but
# a volatile session can be several times that.
#
# Getting this wrong is not a small error. An earlier 0.35% guess here — 4x the
# realized value — made the model rate a losing position at 26% when the market
# said 6%, which reads as "free money" and is really just a broken assumption.
DEFAULT_SIGMA_15M_PCT = 0.09

# Minimum modelled advantage over the price you'd actually pay, before trading.
EDGE_THRESHOLD = 0.05

# Filters
MAX_SPREAD = 0.06
MIN_SECONDS_TO_EXPIRY = 120


@dataclass(frozen=True)
class Signal:
    side: str                 # 'YES' (BTC ends up) or 'NO' (BTC ends down)
    confidence: float         # modelled probability this side wins, 0..1
    edge: float               # confidence minus the price paid
    price: float              # the ask you'd pay for this side
    model_prob_yes: float     # modelled P(YES), before choosing a side
    market_prob_yes: float    # market's implied P(YES) (the YES mid)
    drift_pct: float          # BTC % change since this window opened
    spread: float
    seconds_to_expiry: int
    reason: str


def _norm_cdf(z: float) -> float:
    """Standard normal CDF. No SciPy needed — `math.erf` is exact enough."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def model_prob_yes(
    drift_pct: float,
    seconds_to_expiry: int,
    sigma_15m_pct: float = DEFAULT_SIGMA_15M_PCT,
) -> float:
    """Probability BTC closes at or above this window's open price.

    `drift_pct` is the move so far, in percent, measured from the window open.
    With `t` seconds left the remaining move has standard deviation
    `sigma_15m_pct * sqrt(t / 900)`; the current lead survives if the remaining
    move doesn't erase it.
    """
    if seconds_to_expiry <= 0:
        return 1.0 if drift_pct >= 0 else 0.0

    sigma_remaining = sigma_15m_pct * math.sqrt(seconds_to_expiry / CYCLE_SECONDS)
    if sigma_remaining <= 0:
        return 1.0 if drift_pct >= 0 else 0.0

    return _norm_cdf(drift_pct / sigma_remaining)


def generate_signal(
    yes_bid: float,
    yes_ask: float,
    drift_pct: float,
    seconds_to_expiry: int,
    *,
    sigma_15m_pct: float = DEFAULT_SIGMA_15M_PCT,
    edge_threshold: float = EDGE_THRESHOLD,
    max_spread: float = MAX_SPREAD,
    min_seconds_to_expiry: int = MIN_SECONDS_TO_EXPIRY,
) -> Signal | None:
    """Return a Signal, or None meaning 'no edge — sit this cycle out'.

    Two pre-signal filters run first:

    **Spread filter.** A wide book means you pay the spread on entry and again
    on exit, which can exceed any modelled edge. Above `max_spread` the trade is
    skipped regardless of how good the signal looks.

    **Time-to-expiry filter.** Inside the last couple of minutes, anyone still
    quoting is likely better informed than you are, and there is no time to be
    proven right. Late entries on a 15-minute market are adverse selection.

    The modelled probability is computed from price drift alone and never reads
    the market price. That matters: if the estimate were derived *from* the
    market price, the two would cancel and the "edge" would be a constant
    dressed up as a comparison.
    """
    spread = round(yes_ask - yes_bid, 4)

    if spread > max_spread:
        logger.info(f"Spread too wide ({spread:.3f}) — skipping to avoid slippage")
        return None

    if seconds_to_expiry < min_seconds_to_expiry:
        logger.info(f"Too close to expiry ({seconds_to_expiry}s) — skipping")
        return None

    prob_yes = model_prob_yes(drift_pct, seconds_to_expiry, sigma_15m_pct)
    market_prob_yes = round((yes_bid + yes_ask) / 2, 4)

    # You buy at the ask, so the ask is what the edge must clear. Comparing
    # against the mid would overstate every edge by half the spread.
    no_ask = round(1.0 - yes_bid, 4)
    yes_edge = prob_yes - yes_ask
    no_edge = (1.0 - prob_yes) - no_ask

    if yes_edge >= edge_threshold and yes_edge >= no_edge:
        return Signal(
            side="YES",
            confidence=round(prob_yes, 4),
            edge=round(yes_edge, 4),
            price=yes_ask,
            model_prob_yes=round(prob_yes, 4),
            market_prob_yes=market_prob_yes,
            drift_pct=drift_pct,
            spread=spread,
            seconds_to_expiry=seconds_to_expiry,
            reason=(
                f"model P(YES)={prob_yes:.3f} vs ask {yes_ask:.2f} "
                f"(drift {drift_pct:+.3f}%, {seconds_to_expiry}s left)"
            ),
        )

    if no_edge >= edge_threshold:
        return Signal(
            side="NO",
            confidence=round(1.0 - prob_yes, 4),
            edge=round(no_edge, 4),
            price=no_ask,
            model_prob_yes=round(prob_yes, 4),
            market_prob_yes=market_prob_yes,
            drift_pct=drift_pct,
            spread=spread,
            seconds_to_expiry=seconds_to_expiry,
            reason=(
                f"model P(NO)={1 - prob_yes:.3f} vs ask {no_ask:.2f} "
                f"(drift {drift_pct:+.3f}%, {seconds_to_expiry}s left)"
            ),
        )

    logger.info(
        f"No edge. model P(YES)={prob_yes:.3f} market={market_prob_yes:.3f} "
        f"yes_edge={yes_edge:+.3f} no_edge={no_edge:+.3f}"
    )
    return None
