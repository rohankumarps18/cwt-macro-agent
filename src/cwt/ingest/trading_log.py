"""Load a historical trading log (CSV) into the canonical schema."""
from __future__ import annotations

import json

import pandas as pd

from cwt.config import Settings
from cwt.db import stable_id


def load_trading_log(path: str, settings: Settings, source: str = "csv") -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    ts_col = next((c for c in ("timestamp", "ts", "datetime", "time", "entry_time", "exit_time")
                   if c in raw.columns), None)
    pnl_col = next((c for c in ("pnl", "profit", "net_pnl", "p&l", "profit_loss") if c in raw.columns), None)
    if ts_col is None or pnl_col is None:
        raise ValueError(f"Trading log needs a timestamp column and a pnl column; found {list(raw.columns)}")
    params = [p.lower() for p in settings.param_cols]
    missing = [p for p in params if p not in raw.columns]
    if missing:
        raise ValueError(f"PARAM_COLS {missing} not found in the trading log columns {list(raw.columns)}. "
                         "Set PARAM_COLS to the strategy-parameter columns of your log.")
    ts = pd.to_datetime(raw[ts_col], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(settings.trade_log_tz, ambiguous="NaT", nonexistent="NaT")
    df = pd.DataFrame({"ts_utc": ts.dt.tz_convert("UTC"), "pnl": pd.to_numeric(raw[pnl_col], errors="coerce")})
    bad = df["ts_utc"].isna() | df["pnl"].isna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have an unparseable timestamp or pnl; fix the log first.")
    if "strategy_id" in raw.columns:
        df["strategy_id"] = raw["strategy_id"].astype(str)
    else:
        df["strategy_id"] = raw[params].astype(str).apply(
            lambda r: "|".join(f"{k}={v}" for k, v in zip(params, r)), axis=1)
    df["params"] = json.loads(raw[params].to_json(orient="records"))  # JSON-safe python types
    df["trade_id"] = [stable_id(i, t.isoformat(), s, p) for i, (t, s, p) in
                      enumerate(zip(df["ts_utc"], df["strategy_id"], df["pnl"]))]
    df["source"] = source
    return df.sort_values("ts_utc").reset_index(drop=True)
