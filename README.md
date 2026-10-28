# Kalshi strategy research

Testing trading strategies against [Kalshi](https://kalshi.com), a US prediction
market, over six months.

This repository is mostly **negative results** — strategies that looked sound
and didn't survive measurement. That's deliberate. Finding something that works
is partly luck; reliably proving to yourself that something *doesn't* is a
skill, and it's where almost all the engineering here went.

Every number below has fees subtracted and comes from real market data.

---

## The problem with backtesting your own ideas

You will find an edge. The question is whether it's real.

Over this project, six separate "edges" turned out to be bugs in how I was
measuring. Each one was convincing enough that I'd started building on it:

| What it looked like | What it actually was |
|---|---|
| A +31¢ edge buying dips in tennis | Sampling every price tick, which weights long matches more heavily |
| A uniform +7¢ edge across all price bands | API side-ordering bias — side A read +0.075, side B −0.052 |
| A 100% win rate over 58,448 trades | `limit=1000` returns *newest* trades first; on a busy market that's only the final minutes, where outcomes are already decided |
| A strategy going 8-for-8 live | No time filter, so it was buying pre-game favourites days out — a market regime the research never covered |
| Paper trading up $124 | The same trades filled for real returned −$156. Adverse selection |
| A profitable 15-min crypto model | The market's own prices beat it on Brier score for all six assets |

The guards in this repo exist because each of them caught one of those:

**First-touch sampling.** One observation per market, at the first qualifying
trigger. Sampling every tick is how the tennis edge appeared.

**Cluster by market.** Trades inside one market share an outcome, so 58,448
trades in one market is *one* observation. Reporting a t-statistic over trades
rather than markets inflates significance enormously.

**Symmetry check.** In a two-sided market, buying the faller and buying the
riser must roughly cancel. When both come back the same sign, the measurement
is broken — not the market. `tennis_break_test.py` prints this check alongside
every result.

**Out-of-sample by date.** Parameters chosen on the older half of the dates,
tested once on the newer half, with both halves printed so the gap is visible
rather than hidden.

**Count your tests.** Testing 18 parameter cells and reporting the best one at
1.7σ is meaningless when 0.8 cells clearing that bar is what chance predicts.

**Fees on every leg.** Charging only the entry scores a round trip as free
money.

**Edge is `win_rate − entry_price`.** A 60% win rate at 60¢ is nothing. This
one sounds obvious and is the single most common error in retail trading
writeups.

---

## Experiment 1: stop-loss / take-profit on short-dated markets

**Claim under test:** 15-minute crypto markets move fast, so a stop-loss and
take-profit should beat holding to settlement.

**Why it's falsifiable:** a stop cannot create an edge in a fairly-priced
market. Exits reshape the distribution, not the mean, and each one adds a
second fee. So the test isn't "do stops help?" — it's whether these markets
have *exploitable dynamics*: does a position that moves your way keep moving,
or snap back? If neither, the grid comes out flat everywhere.

```bash
python shortdated_sltp.py
```

Price paths rebuilt from public trades across 90 settled markets (BTC/ETH/SOL),
buying simulated at every observed price in 0.15–0.85. **158,786 simulated
entries per grid cell**, fees on entry *and* exit:

| TP / SL | hold | with exits | difference |
|---|---|---|---|
| 5% / 5% | −0.0212 | −0.0341 | **−0.0129** |
| 10% / 10% | −0.0212 | −0.0422 | −0.0210 |
| 15% / 15% | −0.0212 | −0.0476 | −0.0264 |
| 25% / 25% | −0.0212 | −0.0567 | −0.0355 |

**Result: negative in all 16 cells**, degrading monotonically as the levels
widen — the signature of fee drag with nothing underneath it. Holding alone
loses 2.1¢ per contract, almost exactly the taker fee, which says the market is
fairly priced.

---

## Experiment 2: a WTA/ATP asymmetry in tennis

**Claim under test:** breaks of serve are far more common in women's tennis, so
one break is weaker evidence there. If the market prices a WTA break with the
same severity as an ATP break, it's overreacting.

**Why this one was worth running:** it proposes a *mechanism*, not a pattern
spotted in a chart — and ATP gives a natural control group. Same rule, same
code, same window. An edge appearing in both tours isn't this effect.

```bash
python tennis_break_test.py --markets 500
```

**Result: not supported.** ATP — the control — showed nearly the same edge as
WTA, so whatever is there isn't tour-specific. The best cell reached **1.72σ
out of 18 tested**, against 0.8 expected by chance alone.

The symmetry check came back clean, which is what makes the negative result
trustworthy:

```
tour   fav  drop    n   EDGE(fall)  EDGE(rise)     sum
WTA   0.60  0.10  341      -0.087      +0.088   +0.001
ATP   0.70  0.20  304      +0.042      -0.051   -0.009
```

Every sum within ±0.014 of zero. No one-sided artifact this time.

What the data *does* show, incidentally: buying a favourite after a shallow 10¢
dip is clearly bad — you pay ~78¢ for something that wins ~69%. Deep dips claw
back to roughly fair. Shallow dips are a trap.

---

## Kalshi API gotchas

Things that cost real debugging time, documented so the next person loses less:

**Exchange sharding.** Markets live on numbered shards and balances are
per-shard. An order to an unfunded shard fails as `404 not_found` with
`insufficient_shard_balance / Exchange user not found` — misleading, because
the endpoint, ticker, payload and auth are all fine. Compare the market's
`exchange_index` against `balance_breakdown`.

**`close_time` means two different things.** On a *settled* market it's the
real close. On an *open* market it's a far-future deadline — the real value is
`expected_expiration_time`. Reading the wrong one made a live scanner return
zero results for 2,250 consecutive ticks while the backtest, which runs on
settled markets, looked perfect.

**An exited position still produces a settlement record**, with both
`yes_count_fp` and `no_count_fp` set and `revenue: 0`. Scored naively that's a
total loss of the entire round trip. It also corrupted a running bankroll
badly enough to throttle a bot to zero for 27 days while logging nothing worse
than "size below minimum" — see `src/kalshi_settlement.py`.

**Mixed units in one object.** `revenue` is an integer in cents;
`*_cost_dollars` are strings in dollars. Subtracting one from the other
overstates profit 100×.

**Fills can't be summed naively.** A short YES position is reported as
`action="sell", side="no"`, so an opening trade reads as proceeds. A first
attempt at this reported +$289 on an account that had lost money. Use
`realized_pnl_dollars` from `portfolio/positions`.

**Never retry a POST that creates an order.** A timeout doesn't tell you
whether the exchange received it. `src/kalshi_client.py` deliberately excludes
writes from its retry wrapper.

---

## Layout

```
src/kalshi_client.py       REST client; RSA-PSS request signing, retries on reads only
src/kalshi_auth.py         Key loading and signature generation
src/kalshi_orders.py       Order placement; the YES/NO → bid/ask translation, isolated
src/kalshi_risk.py         Position sizing, exposure caps, daily loss breaker
src/kalshi_settlement.py   Settlement reconciliation and its double-booking guard
src/kalshi_exit.py         Stop-loss / take-profit evaluation
src/kalshi_exit_review.py  Scoring exits against the counterfactual of holding
src/nearcert.py            Scanner for contracts priced near certainty
shortdated_sltp.py         Experiment 1
tennis_break_test.py       Experiment 2
nearcert_backtest.py       Pricing study on near-certainty contracts
nearcert_main.py           Demo-exchange bot; also measures quote persistence
tests/                     103 tests
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # only needed for nearcert_main.py
pytest -q
```

The experiments and backtests use only public market data — no API key, no
account needed. `nearcert_main.py` is the one component that trades; it places
orders on Kalshi's **demo** exchange only.

---

## What I'd tell you about prediction markets

They're mostly efficient. Published research on Kalshi finds the average
contract returns **−20% before fees**
([Bürgi, Deng & Whelan, 2026](https://cepr.org/voxeu/columns/economics-kalshi-prediction-market),
300,000+ contracts), driven by a persistent favourite–longshot bias: cheap
contracts win less often than their price implies.

If you find an edge, the most likely explanation is a bug in how you measured
it. That was true six times here. The discipline that matters isn't finding
signals — it's building the apparatus that tells you when you've fooled
yourself.

Not financial advice. This is a research project.
