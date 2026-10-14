import pandas as pd

df = pd.read_csv(r"E:\cwt-macro-agent\data\backtest_trades.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"], utc=True)
N, slip = 10_000.0, 1e-4
s = df["side"]
entry_raw = df["entry_price"] / (1 + s * slip)
exit_raw = df["exit_price"] / (1 - s * slip)
df["gross"] = s * (exit_raw - entry_raw) / entry_raw * N

print("symbols:", df["symbol"].unique(), "| range:", df["timestamp"].min(), "->", df["timestamp"].max())
print("mean net pnl:", round(df["pnl"].mean(), 3), "| mean gross pnl:", round(df["gross"].mean(), 3))
print("mean gross in bps of notional:", round(df["gross"].mean() / N * 1e4, 3))
print()
print(df.groupby(["lookback", "stop_atr", "mode"])[["pnl", "gross"]].agg(["count", "mean"]).round(2).to_string())
print()
print("trades per entry timestamp:\n", df.groupby("timestamp").size().describe().round(2))
ev = pd.DataFrame({"t": pd.concat([df["timestamp"], df["exit_timestamp"]]),
                   "v": [1] * len(df) + [-1] * len(df)}).sort_values(["t", "v"])
print("max simultaneous positions:", ev["v"].cumsum().max())
