# Evaluation Report: Macro-aware strategy-permutation model

> **NOTE: the trades are SIMULATED.** They come from backtesting strategy permutations on public price bars (fills at next-bar open, per-side cost and slippage assumptions from `build-trade-log`), not from live trading records. Results show how the model ranks simulated strategies; they are not evidence of live profitability.

## Setup
- Trades: 3,482 from 2026-04-17 to 2026-10-13; data sources: {'trades': ['backtest'], 'macro_events': ['apify'], 'news_articles': ['tavily']}
- Model: LightGBM (regression on trade P&L), refit every fold
- Walk-forward: expanding window, min train 60d, test 7d, step 7d, embargo 1d, top-k permutations = 3
- Returns = daily P&L / $100,000 initial capital (non-compounded); annualisation = 252 days; risk-free rate = 0; Sortino target = 0

## Out-of-sample strategy metrics
All policies are evaluated on the same out-of-sample days. `all_trades` is the do-nothing-clever baseline.

|  | n_trades | total_pnl | avg_pnl_per_trade | hit_rate | sharpe | sortino | max_drawdown_pct | max_drawdown_abs |
|---|---|---|---|---|---|---|---|---|
| all_trades | 2305.000 | -10728.976 | -4.655 | 0.414 | -23.543 | -13.260 | -0.107 | -10728.976 |
| model_trade_filter | 709.000 | -4389.645 | -6.191 | 0.417 | -4.804 | -5.138 | -0.044 | -4452.520 |
| model_top_k_permutations | 591.000 | -3830.499 | -6.481 | 0.396 | -6.907 | -7.185 | -0.040 | -4005.117 |
| trailing_top_k_permutations | 555.000 | -2336.578 | -4.210 | 0.429 | -4.485 | -5.703 | -0.028 | -2790.683 |

## Per-fold model quality (out-of-sample)
| fold | test_start | n_train | n_test | ic_spearman | rmse | mae | directional_acc |
|---|---|---|---|---|---|---|---|
| 0 | 2026-06-17 | 1153 | 120 | 0.142 | 41.930 | 32.620 | 0.625 |
| 1 | 2026-06-24 | 1264 | 132 | -0.141 | 53.565 | 42.587 | 0.530 |
| 2 | 2026-07-01 | 1405 | 113 | -0.286 | 36.172 | 31.519 | 0.354 |
| 3 | 2026-07-08 | 1513 | 139 | 0.059 | 29.533 | 22.173 | 0.604 |
| 4 | 2026-07-15 | 1651 | 139 | 0.109 | 31.963 | 26.217 | 0.640 |
| 5 | 2026-07-22 | 1796 | 153 | 0.031 | 38.627 | 30.312 | 0.490 |
| 6 | 2026-07-29 | 1945 | 144 | -0.213 | 59.487 | 46.395 | 0.458 |
| 7 | 2026-08-05 | 2084 | 134 | -0.331 | 32.294 | 24.339 | 0.455 |
| 8 | 2026-08-12 | 2222 | 142 | 0.045 | 22.584 | 17.272 | 0.549 |
| 9 | 2026-08-19 | 2363 | 138 | 0.177 | 15.589 | 13.278 | 0.638 |
| 10 | 2026-08-26 | 2504 | 146 | 0.310 | 21.502 | 17.471 | 0.603 |
| 11 | 2026-09-02 | 2647 | 114 | 0.077 | 25.829 | 20.337 | 0.544 |
| 12 | 2026-09-09 | 2761 | 129 | -0.064 | 24.954 | 20.340 | 0.504 |
| 13 | 2026-09-16 | 2893 | 138 | 0.208 | 31.220 | 22.458 | 0.638 |
| 14 | 2026-09-23 | 3034 | 145 | -0.013 | 30.368 | 24.326 | 0.545 |
| 15 | 2026-09-30 | 3173 | 141 | -0.190 | 31.020 | 24.093 | 0.496 |
| 16 | 2026-10-07 | 3314 | 138 | -0.312 | 22.654 | 17.828 | 0.406 |

Mean out-of-sample rank correlation (IC): **-0.023** (folds > 0: 9/17). An IC near 0 means the model has no usable ranking skill.

## Top features (mean share of gain)
|  | gain |
|---|---|
| news_sent_7d | 0.108 |
| news_sent_3d | 0.104 |
| days_since_fomc | 0.094 |
| param_mode | 0.085 |
| days_since_employment | 0.084 |
| minutes_to_event | 0.074 |
| days_since_cpi | 0.070 |
| hour_est | 0.068 |
| minutes_since_event | 0.058 |
| prev_signed_z | 0.051 |

## Permutation ranking for the latest window (not validated out-of-sample)
| strategy_id | param_lookback | param_stop_atr | param_mode | score |
|---|---|---|---|---|
| lookback=12|stop_atr=2.0|mode=meanrev | 12 | 2.000 | meanrev | -4.772 |
| lookback=24|stop_atr=2.0|mode=meanrev | 24 | 2.000 | meanrev | -4.893 |
| lookback=6|stop_atr=2.0|mode=meanrev | 6 | 2.000 | meanrev | -5.056 |
| lookback=24|stop_atr=1.0|mode=meanrev | 24 | 1.000 | meanrev | -5.237 |
| lookback=12|stop_atr=1.0|mode=meanrev | 12 | 1.000 | meanrev | -5.484 |
| lookback=6|stop_atr=1.0|mode=meanrev | 6 | 1.000 | meanrev | -5.550 |
| lookback=12|stop_atr=2.0|mode=momentum | 12 | 2.000 | momentum | -6.502 |
| lookback=24|stop_atr=2.0|mode=momentum | 24 | 2.000 | momentum | -6.557 |
| lookback=6|stop_atr=2.0|mode=momentum | 6 | 2.000 | momentum | -7.057 |
| lookback=24|stop_atr=1.0|mode=momentum | 24 | 1.000 | momentum | -7.643 |

## Uncertainty
Moving-block bootstrap (5-day blocks, 2000 resamples) 95% intervals for the Sharpe ratio:

| policy | Sharpe | 95% interval |
|---|---|---|
| all_trades | -23.54 | [-29.30, -18.94] |
| model_trade_filter | -4.80 | [-7.80, -1.15] |
| model_top_k_permutations | -6.91 | [-10.33, -3.69] |
| trailing_top_k_permutations | -4.48 | [-9.56, -2.64] |

**Taken vs skipped trades (model_trade_filter):** mean P&L taken -6.19, skipped -3.97, difference -2.22 (95% day-clustered bootstrap interval [-8.59, 3.67]); the interval includes 0, so there is no statistical evidence of skill.

## Limitations (read before trusting any number)
- Only 17 test folds; Sharpe/Sortino over a few months carry wide uncertainty (see the intervals above).
- `model_top_k_permutations` ranks permutations by mean predicted score over the test window, which uses features of trades later in that window (event *schedule* and as-of-trade-time data only, never future outcomes), but a live system would have to decide ex ante. Treat it as optimistic relative to `model_trade_filter`, which is fully causal per trade.
- Trades are SIMULATED on public SPY 1h bars (decision at bar close, fill at next-bar open), not live executions. P&L is net of the `build-trade-log` defaults: 1 bp commission + 1 bp slippage per side (about 4 bps, roughly $4, round trip on a $10,000 notional). Costs are already inside every P&L figure above; they are not missing.
- Gross of costs the strategies show no demonstrable edge (mean about -$0.38 per trade; momentum +$20.65/day, t = 0.97; mean-reversion -$31.36/day, t = -1.61, over 124 days). The net losses are mostly cost drag. `all_trades` holds near-mirror-image momentum and mean-reversion books, so its very negative Sharpe reflects steady small losses with little variance, not a calculation error (P&L recomputed from prices to 1e-14).
- Results use one fixed seed (42). Across seeds 0-4 the filter's mean P&L per taken trade ranged from about -5.8 to -3.2, so a gap of about $1 per trade between policies is within noise. About 82 out-of-sample days and roughly 340 distinct entry times are too few to detect a modest effect.
- All 12 FOMC decisions in the data matched the consensus forecast, so FOMC surprise z-scores are undefined (filled with 0) and carry no information.
- About 22% of news articles carry date-only (midnight UTC) timestamps; these are shifted to 23:59:59 UTC of the same day so they are never visible before publication.
- Event surprise z-scores need 3 prior prints, so the first prints of each event carry no signal.
- Hyper-parameters were not tuned; fixed defaults avoid selecting them on the test folds.
