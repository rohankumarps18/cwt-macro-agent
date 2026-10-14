# CWT Macro Agent: Macro-Aware Strategy-Permutation Model

A quantitative research pipeline built for the CrowdWisdomTrading Quantitative Data Scientist assessment. It integrates macroeconomic calendar events from Apify and financial news from Tavily, stores data in PostgreSQL, engineers time-aware features, trains a LightGBM model using walk-forward validation, and generates performance metrics, sentiment visualizations, and evaluation reports.

The project emphasizes causal feature engineering, reproducibility, and transparent reporting of limitations.

## Architecture

```text
Apify macro calendar ──► macro_events ──┐
Tavily financial news ─► news_articles ─┼──► PostgreSQL
Trading log CSV ───────► trades ────────┘         │
                                                  ▼
                                      trade_macro_context
                                          (SQL join view)
                                                  │
                                                  ▼
                                         Feature engineering
                                                  │
                                                  ▼
                                      LightGBM walk-forward model
                                                  │
                                                  ▼
                                    Metrics, charts, and reports
```

## Quick Start

### 1. Set up the environment

Requirements: Python, Docker with Docker Compose, and PostgreSQL. External data ingestion requires the appropriate API credentials.

```bash
docker compose up -d
pip install -e ".[dev]"
cp .env.example .env
```

Configure `DATABASE_URL`, `APIFY_TOKEN`, `TAVILY_API_KEY`, and `PARAM_COLS` in `.env`. Never commit your actual `.env` file or share live credentials in a public repository.

### 2. Initialize the database

```bash
python -m cwt init-db
```

### 3. Ingest data

```bash
python -m cwt ingest-calendar --dump data/apify_raw.json
python -m cwt ingest-news
```

Live ingestion depends on the configured providers, their current API behavior, account permissions, and available usage credits.

### 4. Load a trading log

If you have a real trading log:

```bash
python -m cwt load-trades --csv path/to/trading_log.csv
```

The CSV should contain one row per trade, a timestamp column such as `timestamp`, `ts`, or `datetime`, a `pnl` column, and the strategy parameter columns specified in `PARAM_COLS`. Naive timestamps are interpreted using `TRADE_LOG_TZ`.

### 5. Run the pipeline

```bash
python -m cwt run --out outputs
```

## No Trading Log? Use the Simulated Backtest

The assessment materials did not provide a trading log. The project therefore supports generating a trade log by backtesting strategy permutations on historical SPY hourly price bars.

```bash
python -m cwt build-trade-log --symbol SPY --out data/backtest_trades.csv --cost-bps 1 --slippage-bps 1
python -m cwt load-trades --csv data/backtest_trades.csv --source backtest
python -m cwt run --out outputs
```

**Important: these are simulated trades, not live executions or a record of actual trading performance.**

The backtest uses the following assumptions:

- Decisions are made at bar close, with entries filled at the next bar's open.
- Twelve strategy permutations are evaluated across lookback, stop distance, and momentum/mean-reversion settings.
- Each permutation holds at most one position and is flat by the end of the New York trading day.
- Stops use ATR(14); gaps through a stop are handled using the bar's open.
- Commission and slippage are modeled at 1 basis point each per side.
- P&L includes these modeled costs.

The saved SPY hourly data and reproducibility audit script are included in the repository. The audit reproduced all 3,482 simulated trades with zero P&L difference. This verifies reproducibility against the saved data, not profitability or generalization to live markets.

Ask the assessment contact whether a real trading log is available, and disclose which dataset you used in your submission.

## Generated Outputs

The `outputs/` directory contains:

- `evaluation_report.md` — strategy metrics, walk-forward results, and limitations.
- `macro_outlook.md` — concise macroeconomic summary.
- PNG visualizations, including macro sentiment, equity drawdown, and feature importance.
- `strategy_metrics.csv` — performance metrics for the evaluated policies.
- `fold_metrics.csv` — per-fold out-of-sample metrics.
- `oos_predictions.csv` — out-of-sample model predictions.
- `latest_permutation_ranking.csv` — latest-window strategy-permutation ranking.

## Synthetic Demo

The project also provides a synthetic-data workflow that does not require live API credentials or a real trading log.

```bash
python -m cwt demo-data --out data/demo --strength 4
python -m cwt reset-db --yes
python -m cwt ingest-calendar --csv data/demo/calendar_demo.csv --source synthetic
python -m cwt ingest-news --csv data/demo/news_demo.csv --source synthetic
python -m cwt load-trades --csv data/demo/trades_demo.csv --source synthetic
python -m cwt run --out outputs
```

**Warning:** `reset-db --yes` truncates database data. Do not run it against a database containing data you need to retain.

Rows loaded with `--source synthetic` are marked as synthetic in the generated reports and visualizations. Keep synthetic and real datasets separate when evaluating results.

## Design Choices

### Timezones

Timestamps are stored in UTC using PostgreSQL `timestamptz`. Naive calendar timestamps are localized using `CALENDAR_SOURCE_TZ`, with timezone-aware conversion. US market-session features use `America/New_York`.

### Causal Feature Engineering

Features are designed to use information available at the relevant trade timestamp:

- Previously released macroeconomic event outcomes.
- The published schedule of upcoming events.
- Surprise z-scores based on earlier releases of the same event.
- News sentiment calculated from articles available by the relevant timestamp.

Date-only news timestamps are shifted to the end of the same UTC day to avoid treating them as available at midnight. This is a conservative approximation, not verification of the exact publication time.

### Warm-Up History

Historical macro events are fetched beyond the primary evaluation window to provide prior observations for surprise-score calculations. The configured lookback and warm-up windows are controlled by `LOOKBACK_DAYS` and `WARMUP_DAYS`.

### Walk-Forward Validation

The model uses expanding training windows, a one-day embargo, fixed hyperparameters, and periodic refitting. Performance is evaluated on out-of-sample periods.

The policies compared are:

- `all_trades` — baseline using all eligible trades.
- `model_trade_filter` — per-trade filtering using model predictions.
- `model_top_k_permutations` — selects permutations based on their predicted scores across the test window.
- `trailing_top_k_permutations` — a trailing-history baseline.

The top-k test-window policy is optimistic relative to a decision made entirely before the test window. Treat the per-trade filter as the more strictly causal comparison.

## Verification Status

The current project test run completed with **30 passed and 1 skipped**.

Additional checks performed include:

- End-to-end execution of the research pipeline.
- A reproducibility audit comparing 3,482 simulated trades against trades rebuilt from saved SPY hourly bars.
- Exact agreement in the rebuilt trade P&L.
- Regeneration of the evaluation report, CSV outputs, and charts.

The repository includes tests for core data handling, feature engineering, backtesting, and evaluation behavior. Test results establish software behavior under the tested conditions; they do not establish that the strategy is profitable.

### External Integrations

Live-provider verification should be distinguished from local pipeline verification:

- **Apify:** the configured actor is `xtracto/forexfactory-calendar`. Confirm its current input/output schema, actual-value availability, proxy requirements, and account permissions before relying on live ingestion.
- **Tavily:** live search behavior, date filtering, and credit consumption should be confirmed using the configured account.
- **Market data:** the saved SPY bars and trade reconstruction were verified for reproducibility. This is not evidence of future market performance.
- **Real trading data:** no claims about performance on a real trading log should be made unless such a log has actually been loaded and evaluated.

## Known Limitations

- The evaluated strategy results are simulated and negative; the model has not demonstrated a profitable trading edge.
- The evaluation covers a limited historical window and a limited number of out-of-sample folds.
- Sharpe and Sortino estimates can be unstable over short periods. Confidence intervals should not be inferred unless they are explicitly calculated and reported.
- The latest-window top-k policy uses predictions across the test window and is optimistic relative to selecting a strategy in advance.
- Sentiment indicators are descriptive features; their predictive value is not established.
- All FOMC decisions in the current dataset matched consensus, leaving the FOMC surprise feature uninformative in this sample.
- Date-only news timestamps require conservative handling because exact publication times are unavailable.
- The macro outlook is template-generated, not generated by an LLM.
- Results depend on the selected period, market data, transaction-cost assumptions, and model configuration.

## Submission Checklist

- Provide the repository link and instructions to reproduce the pipeline.
- Include the evaluation report and macro outlook generated by the final run.
- Identify the data used: real trading log, public-market backtest, or synthetic demonstration.
- Document which external integrations were actually tested.
- Never commit or email live API keys, database passwords, or `.env` files.
- Report simulated performance honestly; do not present backtest results as live trading returns.
