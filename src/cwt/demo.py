"""SYNTHETIC data generator, used only to exercise the pipeline when real inputs are unavailable.

Everything it writes is loaded with --source synthetic, which makes every chart/report carry a
"SYNTHETIC" banner. With signal=True a mild, known relationship between macro events and P&L is planted
(a sanity check that the pipeline *can* recover signal); with signal=False P&L is pure noise
(a leakage check: a correct pipeline must then find nothing).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/New_York"
def _utc(d) -> pd.Timestamp:
    ts = pd.Timestamp(d)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


SD = {"CPI": 0.1, "NFP": 60_000, "UNEMP": 0.1, "AHE": 0.1, "RATE": 0.1}


def _events(end: pd.Timestamp, days: int, rng: np.random.Generator) -> pd.DataFrame:
    start = end - pd.Timedelta(days=days)
    rows = []
    months = pd.date_range(start.normalize() - pd.offsets.MonthBegin(1), end + pd.Timedelta(days=45), freq="MS")
    for m in months:
        cpi = m + pd.Timedelta(days=11)
        while cpi.dayofweek >= 5:
            cpi += pd.Timedelta(days=1)
        fri = m + pd.Timedelta(days=(4 - m.dayofweek) % 7)
        rows += [(cpi, "08:30", "CPI m/m", "US", 0.3, SD["CPI"], "high"),
                 (cpi, "08:30", "Core CPI m/m", "US", 0.3, SD["CPI"], "high"),
                 (fri, "08:30", "Non-Farm Employment Change", "US", 180_000, SD["NFP"], "high"),
                 (fri, "08:30", "Unemployment Rate", "US", 4.0, SD["UNEMP"], "high"),
                 (fri, "08:30", "Average Hourly Earnings m/m", "US", 0.3, SD["AHE"], "medium")]
    for d in pd.date_range(start + pd.Timedelta(days=20), end + pd.Timedelta(days=45), freq="42D"):
        while d.dayofweek >= 5:
            d += pd.Timedelta(days=1)
        rows.append((d, "14:00", "Fed Interest Rate Decision", "US", 5.25, SD["RATE"], "high"))
    out = []
    for d, t, name, c, fc, sd, imp in rows:
        if not (start <= _utc(d) <= end + pd.Timedelta(days=45)):
            continue
        actual = fc + rng.normal(0, sd)
        if "Rate" in name:
            actual = fc + 0.25 * rng.choice([-1, 0, 0, 0, 1])
        out.append({"date": d.strftime("%Y-%m-%d"), "time": t, "event": name, "country": c,
                    "forecast": round(fc, 3), "actual": round(actual, 3), "previous": round(fc, 3),
                    "impact": imp, "_sd": sd})
    df = pd.DataFrame(out).sort_values(["date", "time"]).reset_index(drop=True)
    now_local = end.tz_convert(TZ)
    future = pd.to_datetime(df["date"] + " " + df["time"]).dt.tz_localize(TZ) > now_local
    df.loc[future, "actual"] = np.nan  # not released yet
    return df


def make_demo(out_dir: str, days: int = 180, signal: bool = True, seed: int = 7, strength: float = 1.0) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").normalize()
    ev = _events(end, days + 365, rng)  # +365d warm-up history for surprise z-scores
    ev["ts_utc"] = pd.to_datetime(ev["date"] + " " + ev["time"]).dt.tz_localize(TZ).dt.tz_convert("UTC")
    ev["z_true"] = ((ev["actual"] - ev["forecast"]) / ev["_sd"]) * np.where(ev["event"].str.contains("Unemployment"), -1, 1)

    # --- trades ---
    grid = list(itertools.product([10, 20, 50], [1.0, 2.0], ["momentum", "meanrev"]))
    rel = ev[ev["actual"].notna()].sort_values("ts_utc")
    ev_ts = rel["ts_utc"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    ev_z = rel["z_true"].to_numpy()
    recs = []
    for d in pd.bdate_range(end - pd.Timedelta(days=days), end - pd.Timedelta(days=1)):
        for lookback, stop, mode in grid:
            for _ in range(6):
                minute = rng.integers(570, 960)
                ts = (pd.Timestamp(d.date(), tz=TZ) + pd.Timedelta(minutes=int(minute))).tz_convert("UTC")
                pnl = rng.normal(0, 60)
                if signal:
                    i = np.searchsorted(ev_ts, np.datetime64(ts.tz_localize(None)), side="right") - 1
                    sgn = 1.0 if mode == "momentum" else -1.0
                    if i >= 0:
                        mins = (ts - pd.Timestamp(ev_ts[i], tz="UTC")).total_seconds() / 60
                        if 0 <= mins <= 120:
                            pnl += strength * (25 * sgn if mode == "momentum" else -15)
                        pnl += strength * 8 * sgn * np.tanh(ev_z[i])
                recs.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "lookback": lookback,
                             "stop_atr": stop, "mode": mode, "pnl": round(pnl, 2)})
    trades = pd.DataFrame(recs)

    # --- news (sentiment loosely follows latest surprise sign) ---
    pos = ["Markets rally as data beats expectations", "Strong economy lifts confidence",
           "Solid growth reassures investors"]
    neg = ["Hot inflation fuels rate worries", "Stocks slide on hawkish Fed fears",
           "Weak outlook weighs on sentiment"]
    news = []
    for i, d in enumerate(pd.date_range(end - pd.Timedelta(days=days), end - pd.Timedelta(days=1), freq="12h")):
        k = np.searchsorted(ev_ts, np.datetime64(d.tz_convert("UTC").tz_localize(None)), side="right") - 1
        regime = np.tanh(ev_z[k]) if k >= 0 else 0.0
        positive = rng.random() < 0.5 + 0.3 * regime
        title = rng.choice(pos if positive else neg)
        news.append({"url": f"https://example.invalid/synthetic/{i}", "title": title,
                     "content": f"{title}. Analysts discuss CPI, jobs and the Fed.",
                     "published_date": d.tz_convert("UTC").isoformat(), "score": 0.5})
    news = pd.DataFrame(news)

    paths = {"calendar": out / "calendar_demo.csv", "news": out / "news_demo.csv", "trades": out / "trades_demo.csv"}
    ev.drop(columns=["ts_utc", "z_true", "_sd"]).to_csv(paths["calendar"], index=False)
    news.to_csv(paths["news"], index=False)
    trades.to_csv(paths["trades"], index=False)
    return {k: str(v) for k, v in paths.items()}
