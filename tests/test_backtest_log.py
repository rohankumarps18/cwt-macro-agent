import numpy as np
import pandas as pd
import pytest

from cwt.ingest.backtest_log import backtest_permutation, build_trade_log


def _bars():
    # HAND-BUILT deterministic bars (not market data): 12 hourly bars from 09:30 New York on one winter day.
    ts = pd.date_range("2026-03-02 14:30", periods=12, freq="h", tz="UTC")
    o = np.arange(100.0, 112.0)
    return pd.DataFrame({"ts_utc": ts, "open": o, "high": o + 0.5 + 0.2, "low": o - 0.2, "close": o + 0.5})


KW = dict(lookback=2, stop_atr=1.0, mode="momentum", hold_bars=100, notional=10_000.0, cost_bps=0.0,
          slippage_bps=0.0, atr_n=2)


def test_entry_at_next_open_and_exit_at_day_end_close():
    tr = backtest_permutation(_bars(), **KW)
    assert len(tr) == 1
    assert tr[0]["entry_price"] == 103.0                    # signal at close of bar 2 -> fill at OPEN of bar 3
    assert tr[0]["exit_price"] == 111.5                     # flat at last bar's close
    assert tr[0]["pnl"] == pytest.approx((111.5 - 103.0) / 103.0 * 10_000)


def test_stop_fill_level_and_gap_through_stop():
    b = _bars()
    b.loc[4, "low"] = 100.0                                 # pierces stop (103 - 1.2 = 101.8) intrabar
    t1 = backtest_permutation(b, **KW)[0]
    assert t1["exit_price"] == pytest.approx(101.8)
    b.loc[4, "open"] = 95.0                                 # gaps THROUGH the stop: fills at the open
    t2 = backtest_permutation(b, **KW)[0]
    assert t2["exit_price"] == pytest.approx(95.0)


def test_costs_and_slippage_are_deducted():
    base = backtest_permutation(_bars(), **KW)[0]["pnl"]
    costed = backtest_permutation(_bars(), **{**KW, "cost_bps": 5.0})[0]["pnl"]
    assert costed == pytest.approx(base - 2 * 5e-4 * 10_000)
    slipped = backtest_permutation(_bars(), **{**KW, "slippage_bps": 10.0})[0]
    assert slipped["entry_price"] > 103.0 and slipped["exit_price"] < 111.5   # always against the trade


def test_meanrev_takes_the_opposite_side():
    assert backtest_permutation(_bars(), **{**KW, "mode": "meanrev"})[0]["side"] == -1


def test_no_lookahead_changing_future_bars_keeps_earlier_trades():
    b = _bars()
    base = backtest_permutation(b, **{**KW, "hold_bars": 2})
    b2 = b.copy()
    b2.loc[9:, ["open", "high", "low", "close"]] *= 1.5     # alter only the last bars
    alt = backtest_permutation(b2, **{**KW, "hold_bars": 2})
    first = [t for t in base if t["exit_timestamp"] <= b.loc[8, "ts_utc"]]
    assert first and all(a == c for a, c in zip(first, alt))


def test_build_trade_log_schema_matches_loader():
    log = build_trade_log(_bars(), grid={"lookback": [2], "stop_atr": [1.0], "mode": ["momentum", "meanrev"]},
                          symbol="TEST", hold_bars=100, atr_n=2)
    assert {"timestamp", "pnl", "lookback", "stop_atr", "mode", "exit_timestamp"} <= set(log.columns)
