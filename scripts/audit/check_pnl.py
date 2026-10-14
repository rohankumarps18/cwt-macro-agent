import pandas as pd

df = pd.read_csv(r"E:\cwt-macro-agent\data\backtest_trades.csv")
notional, cost_bps = 10_000.0, 1.0
recalc = (df["side"] * (df["exit_price"] - df["entry_price"]) / df["entry_price"] * notional
          - 2 * cost_bps / 1e4 * notional)
print("rows:", len(df))
print("columns:", list(df.columns))
print("max abs recompute error:", (recalc - df["pnl"]).abs().max())
print("mean pnl:", df["pnl"].mean(), "| approx pre-cost:", df["pnl"].mean() + 4.0)
