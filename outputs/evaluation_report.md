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
| model_trade_filter | 678.000 | -2466.958 | -3.639 | 0.459 | -3.092 | -3.757 | -0.029 | -2913.690 |
| model_top_k_permutations | 573.000 | -2697.940 | -4.708 | 0.426 | -4.518 | -5.457 | -0.031 | -3139.661 |
| trailing_top_k_permutations | 555.000 | -2336.578 | -4.210 | 0.429 | -4.485 | -5.703 | -0.028 | -2790.683 |

## Per-fold model quality (out-of-sample)
| fold | test_start | n_train | n_test | ic_spearman | rmse | mae | directional_acc |
|---|---|---|---|---|---|---|---|
| 0 | 2026-06-17 | 1153 | 120 | 0.146 | 42.159 | 32.184 | 0.642 |
| 1 | 2026-06-24 | 1264 | 132 | 0.097 | 51.670 | 40.678 | 0.682 |
| 2 | 2026-07-01 | 1405 | 113 | -0.278 | 36.458 | 31.539 | 0.327 |
| 3 | 2026-07-08 | 1513 | 139 | -0.097 | 29.770 | 23.015 | 0.540 |
| 4 | 2026-07-15 | 1651 | 139 | 0.110 | 32.012 | 26.506 | 0.619 |
| 5 | 2026-07-22 | 1796 | 153 | 0.225 | 38.042 | 29.652 | 0.542 |
| 6 | 2026-07-29 | 1945 | 144 | -0.320 | 59.852 | 47.407 | 0.535 |
| 7 | 2026-08-05 | 2084 | 134 | -0.202 | 30.050 | 22.874 | 0.396 |
| 8 | 2026-08-12 | 2222 | 142 | 0.000 | 21.669 | 16.422 | 0.592 |
| 9 | 2026-08-19 | 2363 | 138 | 0.255 | 15.219 | 12.918 | 0.703 |
| 10 | 2026-08-26 | 2504 | 146 | 0.245 | 22.194 | 17.868 | 0.623 |
| 11 | 2026-09-02 | 2647 | 114 | 0.373 | 21.304 | 16.889 | 0.640 |
| 12 | 2026-09-09 | 2761 | 129 | -0.044 | 24.311 | 19.664 | 0.481 |
| 13 | 2026-09-16 | 2893 | 138 | 0.130 | 31.311 | 22.395 | 0.652 |
| 14 | 2026-09-23 | 3034 | 145 | 0.039 | 30.949 | 25.132 | 0.490 |
| 15 | 2026-09-30 | 3173 | 141 | -0.187 | 31.676 | 24.105 | 0.532 |
| 16 | 2026-10-07 | 3314 | 138 | -0.335 | 22.769 | 17.369 | 0.536 |

Mean out-of-sample rank correlation (IC): **0.009** (folds > 0: 10/17). An IC near 0 means the model has no usable ranking skill.

## Top features (mean share of gain)
|  | gain |
|---|---|
| news_sent_3d | 0.124 |
| news_sent_7d | 0.099 |
| days_since_fomc | 0.090 |
| days_since_employment | 0.082 |
| param_mode | 0.079 |
| days_since_cpi | 0.071 |
| minutes_to_event | 0.070 |
| hour_est | 0.069 |
| minutes_since_event | 0.067 |
| news_n_7d | 0.054 |

## Permutation ranking for the latest window (not validated out-of-sample)
| strategy_id | param_lookback | param_stop_atr | param_mode | score |
|---|---|---|---|---|
| lookback=12|stop_atr=2.0|mode=meanrev | 12 | 2.000 | meanrev | -3.838 |
| lookback=24|stop_atr=2.0|mode=meanrev | 24 | 2.000 | meanrev | -4.188 |
| lookback=6|stop_atr=2.0|mode=meanrev | 6 | 2.000 | meanrev | -4.469 |
| lookback=12|stop_atr=1.0|mode=meanrev | 12 | 1.000 | meanrev | -5.127 |
| lookback=24|stop_atr=1.0|mode=meanrev | 24 | 1.000 | meanrev | -5.318 |
| lookback=6|stop_atr=1.0|mode=meanrev | 6 | 1.000 | meanrev | -5.661 |
| lookback=12|stop_atr=2.0|mode=momentum | 12 | 2.000 | momentum | -6.783 |
| lookback=24|stop_atr=2.0|mode=momentum | 24 | 2.000 | momentum | -6.930 |
| lookback=12|stop_atr=1.0|mode=momentum | 12 | 1.000 | momentum | -7.287 |
| lookback=6|stop_atr=2.0|mode=momentum | 6 | 2.000 | momentum | -7.398 |

## Uncertainty
Moving-block bootstrap (5-day blocks, 2000 resamples) 95% intervals for the Sharpe ratio:

| policy | Sharpe | 95% interval |
|---|---|---|
| all_trades | -23.54 | [-29.30, -18.94] |
| model_trade_filter | -3.09 | [-6.20, 0.34] |
| model_top_k_permutations | -4.52 | [-8.28, -2.65] |
| trailing_top_k_permutations | -4.48 | [-9.56, -2.64] |

**Taken vs skipped trades (model_trade_filter):** mean P&L taken -3.64, skipped -5.08, difference 1.44 (95% day-clustered bootstrap interval [-3.92, 7.13]); the interval includes 0, so there is no statistical evidence of skill.

## Limitations (read before trusting any number)
- Only 17 test folds; Sharpe/Sortino over a few months carry wide uncertainty (see the intervals above).
- `model_top_k_permutations` ranks permutations by mean predicted score over the test window, which uses features of trades later in that window (event *schedule* and as-of-trade-time data only, never future outcomes), but a live system would have to decide ex ante. Treat it as optimistic relative to `model_trade_filter`, which is fully causal per trade.
- No transaction costs or slippage beyond what is already in the logged P&L.
- Event surprise z-scores need 3 prior prints, so the first prints of each event carry no signal.
- Hyper-parameters were not tuned; fixed defaults avoid selecting them on the test folds.
