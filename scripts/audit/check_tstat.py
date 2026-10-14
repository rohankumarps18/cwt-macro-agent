import numpy as np, pandas as pd

df = pd.read_csv(r"E:\cwt-macro-agent\data\backtest_trades.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
N, slip = 10_000.0, 1e-4
s = df["side"]
entry_raw = df["entry_price"] / (1 + s * slip)
exit_raw = df["exit_price"] / (1 - s * slip)
df["gross"] = s * (exit_raw - entry_raw) / entry_raw * N
df["day"] = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
all_days = pd.DatetimeIndex(sorted(df["day"].unique()))

for m, g in df.groupby("mode"):
    for col in ("gross", "pnl"):
        d = g.groupby("day")[col].sum().reindex(all_days, fill_value=0.0)
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        print(f"{m:9s} {col:5s} days={len(d)} mean/day={d.mean():8.2f} t-stat={t:6.2f}")
