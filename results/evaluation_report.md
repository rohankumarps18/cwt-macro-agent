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
| model_trade_filter | 717.000 | -2610.117 | -3.640 | 0.462 | -3.150 | -3.960 | -0.031 | -3072.516 |
| model_top_k_permutations | 578.000 | -3381.414 | -5.850 | 0.410 | -5.985 | -6.441 | -0.036 | -3597.096 |
| trailing_top_k_permutations | 555.000 | -2336.578 | -4.210 | 0.429 | -4.485 | -5.703 | -0.028 | -2790.683 |

## Per-fold model quality (out-of-sample)
| fold | test_start | n_train | n_test | ic_spearman | rmse | mae | directional_acc |
|---|---|---|---|---|---|---|---|
| 0 | 2026-06-17 | 1153 | 120 | 0.110 | 42.118 | 32.585 | 0.625 |
| 1 | 2026-06-24 | 1264 | 132 | 0.075 | 51.841 | 40.886 | 0.682 |
| 2 | 2026-07-01 | 1405 | 113 | -0.304 | 35.745 | 31.585 | 0.354 |
| 3 | 2026-07-08 | 1513 | 139 | -0.150 | 30.008 | 23.609 | 0.532 |
| 4 | 2026-07-15 | 1651 | 139 | 0.063 | 31.978 | 26.376 | 0.647 |
| 5 | 2026-07-22 | 1796 | 153 | 0.268 | 37.829 | 29.659 | 0.542 |
| 6 | 2026-07-29 | 1945 | 144 | -0.352 | 59.846 | 47.370 | 0.549 |
| 7 | 2026-08-05 | 2084 | 134 | -0.317 | 31.049 | 23.422 | 0.403 |
| 8 | 2026-08-12 | 2222 | 142 | -0.002 | 21.625 | 16.362 | 0.585 |
| 9 | 2026-08-19 | 2363 | 138 | 0.292 | 14.962 | 12.175 | 0.717 |
| 10 | 2026-08-26 | 2504 | 146 | 0.205 | 22.951 | 18.186 | 0.637 |
| 11 | 2026-09-02 | 2647 | 114 | 0.332 | 22.769 | 17.471 | 0.632 |
| 12 | 2026-09-09 | 2761 | 129 | -0.096 | 24.785 | 20.061 | 0.465 |
| 13 | 2026-09-16 | 2893 | 138 | 0.165 | 31.099 | 22.579 | 0.616 |
| 14 | 2026-09-23 | 3034 | 145 | 0.076 | 30.663 | 24.687 | 0.517 |
| 15 | 2026-09-30 | 3173 | 141 | -0.146 | 31.645 | 24.486 | 0.539 |
| 16 | 2026-10-07 | 3314 | 138 | -0.339 | 23.289 | 18.219 | 0.493 |

Mean out-of-sample rank correlation (IC): **-0.007** (folds > 0: 9/17). An IC near 0 means the model has no usable ranking skill.

## Top features (mean share of gain)
|  | gain |
|---|---|
| news_sent_3d | 0.124 |
| news_sent_7d | 0.102 |
| days_since_fomc | 0.098 |
| days_since_employment | 0.089 |
| param_mode | 0.082 |
| minutes_to_event | 0.076 |
| minutes_since_event | 0.069 |
| hour_est | 0.063 |
| days_since_cpi | 0.061 |
| news_n_7d | 0.056 |

## Permutation ranking for the latest window (not validated out-of-sample)
| strategy_id | param_lookback | param_stop_atr | param_mode | score |
|---|---|---|---|---|
| lookback=12|stop_atr=2.0|mode=meanrev | 12 | 2.000 | meanrev | -4.475 |
| lookback=24|stop_atr=2.0|mode=meanrev | 24 | 2.000 | meanrev | -4.541 |
| lookback=12|stop_atr=1.0|mode=meanrev | 12 | 1.000 | meanrev | -5.001 |
| lookback=6|stop_atr=2.0|mode=meanrev | 6 | 2.000 | meanrev | -5.019 |
| lookback=24|stop_atr=1.0|mode=meanrev | 24 | 1.000 | meanrev | -5.058 |
| lookback=6|stop_atr=1.0|mode=meanrev | 6 | 1.000 | meanrev | -5.248 |
| lookback=12|stop_atr=2.0|mode=momentum | 12 | 2.000 | momentum | -7.178 |
| lookback=24|stop_atr=2.0|mode=momentum | 24 | 2.000 | momentum | -7.429 |
| lookback=12|stop_atr=1.0|mode=momentum | 12 | 1.000 | momentum | -7.821 |
| lookback=6|stop_atr=2.0|mode=momentum | 6 | 2.000 | momentum | -7.839 |

## Uncertainty
Moving-block bootstrap (5-day blocks, 2000 resamples) 95% intervals for the Sharpe ratio:

| policy | Sharpe | 95% interval |
|---|---|---|
| all_trades | -23.54 | [-29.30, -18.94] |
| model_trade_filter | -3.15 | [-6.13, 0.39] |
| model_top_k_permutations | -5.98 | [-9.07, -3.19] |
| trailing_top_k_permutations | -4.48 | [-9.56, -2.64] |

**Taken vs skipped trades (model_trade_filter):** mean P&L taken -3.64, skipped -5.11, difference 1.47 (95% day-clustered bootstrap interval [-4.11, 7.03]); the interval includes 0, so there is no statistical evidence of skill.

## Limitations (read before trusting any number)
- Only 17 test folds; Sharpe/Sortino over a few months carry wide uncertainty (see the intervals above).
- `model_top_k_permutations` ranks permutations by mean predicted score over the test window, which uses features of trades later in that window (event *schedule* and as-of-trade-time data only, never future outcomes), but a live system would have to decide ex ante. Treat it as optimistic relative to `model_trade_filter`, which is fully causal per trade.
- No transaction costs or slippage beyond what is already in the logged P&L.
- Event surprise z-scores need 3 prior prints, so the first prints of each event carry no signal.
- Hyper-parameters were not tuned; fixed defaults avoid selecting them on the test folds.
