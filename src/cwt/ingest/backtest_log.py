"""Build a trade log by BACKTESTING strategy permutations on public OHLC bars.

This is *simulated* trading on real historical prices, not a record of live trades. Everything below is a
stated assumption, so say so in any report built from it:
  - decision at the CLOSE of bar t, fill at the OPEN of bar t+1 (no look-ahead)
  - one open position per permutation; position is flat by the last bar of the New-York trading day
  - stop at stop_atr x ATR(atr_n) measured at decision time; gaps through the stop fill at the open
  - costs: `cost_bps` per side (commission/spread) plus `slippage_bps` per side against the trade
  - fixed notional per trade
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

DEFAULT_GRID = {"lookback": [6, 12, 24], "stop_atr": [1.0, 2.0], "mode": ["momentum", "meanrev"]}


def fetch_prices_yfinance(symbol: str, start: str, end: str, interval: str = "1h") -> pd.DataFrame:
    """Free, no API key. Yahoo serves ~730 days of 1h bars."""
    import yfinance as yf

    raw = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"No {interval} bars returned for {symbol} between {start} and {end}.")
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close"]].copy()
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df.rename_axis("ts_utc").reset_index()


def load_prices_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    ts_col = next((c for c in ("ts_utc", "timestamp", "datetime", "date") if c in raw.columns), None)
    missing = [c for c in ("open", "high", "low", "close") if c not in raw.columns]
    if ts_col is None or missing:
        raise ValueError(f"Price CSV needs a timestamp column and open/high/low/close; found {list(raw.columns)}")
    ts = pd.to_datetime(raw[ts_col])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz)
    out = raw[["open", "high", "low", "close"]].astype(float).copy()
    out.insert(0, "ts_utc", ts.dt.tz_convert("UTC"))
    return out.sort_values("ts_utc").reset_index(drop=True)


def _atr(bars: pd.DataFrame, n: int) -> np.ndarray:
    pc = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - pc).abs(), (bars["low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy()


def backtest_permutation(bars: pd.DataFrame, lookback: int, stop_atr: float, mode: str, hold_bars: int = 4,
                         notional: float = 10_000.0, cost_bps: float = 1.0, slippage_bps: float = 1.0,
                         atr_n: int = 14) -> list[dict]:
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ts = bars["ts_utc"]
    et_day = ts.dt.tz_convert("America/New_York").dt.date.to_numpy()
    bar_len = ts.diff().median()
    n = len(bars)
    ret = pd.Series(c).pct_change(lookback).to_numpy()
    atr = _atr(bars, atr_n)
    slip = slippage_bps / 1e4
    trades, t = [], max(lookback, atr_n)
    while t < n - 1:
        if np.isnan(ret[t]) or np.isnan(atr[t]) or ret[t] == 0:
            t += 1
            continue
        side = float(np.sign(ret[t])) * (1.0 if mode == "momentum" else -1.0)
        i = t + 1                                                  # fill at next bar's open
        entry = o[i] * (1 + side * slip)
        stop = entry - side * stop_atr * atr[t]
        last = i
        while last + 1 < n and last + 1 <= i + hold_bars - 1 and et_day[last + 1] == et_day[i]:
            last += 1                                              # flat by end of the NY day / hold limit
        exit_idx, exit_raw = last, c[last]
        exit_ts = ts.iloc[last] + bar_len
        for j in range(i, last + 1):
            if side > 0 and l[j] <= stop:
                exit_idx, exit_raw, exit_ts = j, (o[j] if o[j] <= stop else stop), ts.iloc[j]
                break
            if side < 0 and h[j] >= stop:
                exit_idx, exit_raw, exit_ts = j, (o[j] if o[j] >= stop else stop), ts.iloc[j]
                break
        exit_fill = exit_raw * (1 - side * slip)
        gross = side * (exit_fill - entry) / entry * notional
        pnl = gross - 2 * cost_bps / 1e4 * notional
        trades.append({"timestamp": ts.iloc[i], "exit_timestamp": exit_ts, "lookback": lookback,
                       "stop_atr": stop_atr, "mode": mode, "side": int(side), "entry_price": entry,
                       "exit_price": exit_fill, "pnl": pnl})
        t = exit_idx                                               # next decision at that bar's close
    return trades


def build_trade_log(bars: pd.DataFrame, grid: dict | None = None, symbol: str = "", since: pd.Timestamp | None = None,
                    **kwargs) -> pd.DataFrame:
    grid = grid or DEFAULT_GRID
    rows = []
    for lb, st, md in itertools.product(grid["lookback"], grid["stop_atr"], grid["mode"]):
        rows += backtest_permutation(bars, lb, st, md, **kwargs)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Backtest produced no trades; check the price data and parameters.")
    if since is not None:
        df = df[df["timestamp"] >= since]
    df.insert(2, "symbol", symbol)
    return df.sort_values("timestamp").reset_index(drop=True)
