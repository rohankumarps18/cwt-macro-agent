# CWT Macro Agent: macro-aware strategy-permutation model

Pipeline for the CrowdWisdomTrading Quantitative Data Scientist assessment: scrape the macro calendar (Apify) and
news (Tavily), join them to a trading log in **Postgres**, engineer causal features, train a LightGBM model with
**walk-forward validation**, and report Sharpe / Sortino / max drawdown plus sentiment charts.

```
Apify actor ──► macro_events ─┐
Tavily news ──► news_articles ├─► Postgres ─► trade_macro_context (SQL LATERAL join view)
trading log CSV ──► trades ───┘                     │
                                                    ▼
                         features.py ─► model.py (expanding walk-forward, embargo) ─► metrics / charts / report
```

## Quick start (real data)
```bash
docker compose up -d                      # local Postgres (or point DATABASE_URL at any Postgres)
pip install -e .[dev]
cp .env.example .env                      # fill APIFY_TOKEN, TAVILY_API_KEY, PARAM_COLS (never commit .env)
python -m cwt init-db
python -m cwt ingest-calendar --dump data/apify_raw.json     # Apify actor, lookback + warm-up window
python -m cwt ingest-news                                    # Tavily, weekly windows x 3 queries
python -m cwt load-trades --csv path/to/trading_log.csv      # your real log, if you have one (see below if not)
python -m cwt run --out outputs
```

### No trading log? Build one from free public prices
The assessment PDF does not ship a log. `build-trade-log` backtests the strategy permutations on free Yahoo
hourly bars (no key) and writes a CSV the loader accepts:
```bash
python -m cwt build-trade-log --symbol SPY --out data/backtest_trades.csv --cost-bps 1 --slippage-bps 1
python -m cwt load-trades --csv data/backtest_trades.csv --source backtest
```
These are **simulated** trades, not live records; `--source backtest` makes the evaluation report say so. Assumptions
(all in `ingest/backtest_log.py`): decide at bar close, fill at next bar open; 12 permutations (lookback x stop_atr x
momentum/meanrev); one position per permutation, flat by end of the New-York day; stop = stop_atr x ATR(14), gaps fill
at the open; per-side cost + slippage; fixed notional. Still ask Gilad whether he has a real log, and state this
clearly in your submission.
Outputs: `evaluation_report.md`, `macro_outlook.md` (1-page summary), 5 PNG charts, `strategy_metrics.csv`,
`fold_metrics.csv`, `oos_predictions.csv`, `latest_permutation_ranking.csv`.

**Trading log CSV:** one row per trade with a timestamp column (`timestamp`/`ts`/`datetime`), `pnl`, and the strategy
parameter columns named in `PARAM_COLS` (optional `strategy_id`). Naive timestamps are read in `TRADE_LOG_TZ`.

## Synthetic demo (no keys, no log needed)
```bash
python -m cwt demo-data --out data/demo --strength 4
python -m cwt reset-db --yes
python -m cwt ingest-calendar --csv data/demo/calendar_demo.csv --source synthetic
python -m cwt ingest-news --csv data/demo/news_demo.csv --source synthetic
python -m cwt load-trades --csv data/demo/trades_demo.csv --source synthetic
python -m cwt run --out outputs
```
Anything loaded with `--source synthetic` makes every chart/report carry a SYNTHETIC banner. Do not mix synthetic and
real rows in one database (`reset-db --yes` truncates).

## Design choices that matter
- **Timezones:** every timestamp is stored as UTC (`timestamptz`); naive calendar times are localised with
  `CALENDAR_SOURCE_TZ` (DST-aware); US-session features use America/New_York.
- **No look-ahead:** a trade sees only (a) outcomes of events released at/before its timestamp and (b) the *schedule*
  of the next event. Surprise z-scores use only earlier prints of the same event. News features are rolling windows
  over articles published up to that moment; undated articles are placed at the *end* of their search window.
- **Warm-up history:** monthly events give ~6 prints per 180 days and a z-score needs 3 prior prints, so the calendar
  is fetched for `LOOKBACK_DAYS + WARMUP_DAYS` to seed the statistics; charts show only the trading window.
- **Walk-forward:** expanding window, 1-day embargo, model refit every fold, fixed (untuned) hyper-parameters.
- **Policies compared on identical out-of-sample days:** `all_trades`, `model_trade_filter` (fully causal),
  `model_top_k_permutations`, and a naive `trailing_top_k_permutations` baseline.

## What has and has not been verified
Verified by actually running it (see `tests/`, 27 tests passing, incl. a real-Postgres test of the join view and
hand-computed backtest cases: next-open fills, stop and gap fills, costs, no look-ahead):
- Parsing/timezone/DST handling, causal z-scores, hand-checked Sharpe/Sortino/drawdown, fold non-overlap, SQL join
  semantics + idempotent upserts, full CLI pipeline end to end on a local Postgres 16.
- Synthetic controls (17 folds): **no planted signal** gave mean IC -0.011 (7/17 folds > 0), i.e. no skill found in
  noise; **strong planted signal** gave mean IC 0.070 (15/17 folds > 0) and the model's top features were the planted
  drivers. So the pipeline neither leaks on noise nor misses an obvious signal.

**Not verified (needs your keys/data):**
- A live Apify run. Defaults target `xtracto/forexfactory-calendar`; its input/output field names were taken from the
  actor's Apify docs page, not from a run. Per those docs, without a Residential proxy it returns forecast-only data
  with no actuals; `ingest-calendar` refuses to continue in that case. I have not checked your plan covers it.
- Live Tavily calls. `start_date`/`end_date` with `topic="news"` exist in the installed client's signature, but server
  behaviour and free-tier credit use (~78 calls for 26 weeks x 3 queries) are untested.
- `build-trade-log` against live Yahoo data (the sandbox could not reach it). The backtest engine and CSV hand-off were
  tested on hand-built bars and a random-walk price file, which is not market data.
- Anything on real market data or a real trading log. Until one is loaded, no real performance numbers exist.

## Known limitations
- `model_top_k_permutations` ranks permutations by mean predicted score over the test window, which is optimistic
  versus deciding ex ante; `model_trade_filter` is the strictly causal result.
- Few folds => wide uncertainty on Sharpe/Sortino (no confidence intervals computed). No extra cost/slippage model.
- Sentiment is a signed-surprise index plus VADER on news text: descriptive, not predictive.
- The macro summary is template-generated from the data (no LLM).

## Submission checklist
- Repo link; actor ID (`xtracto/forexfactory-calendar` or whichever you actually run) and Tavily usage noted.
- **Do not commit or email live keys.** Send revocable free-tier keys separately and rotate them afterwards.
- Attach `outputs/evaluation_report.md` from a run on the *real* log, not the synthetic example.
