import subprocess, sys
import pandas as pd
from cwt.pipeline import get_settings
from cwt.ingest import backtest_log

s = get_settings()
start, end = s.window()
warm = pd.Timestamp(start) - pd.Timedelta(days=10)
bars = backtest_log.fetch_prices_yfinance("SPY", warm.date().isoformat(), end.isoformat(), "1h")
bars.to_csv(r"data\spy_1h.csv", index=False)
print("saved bars:", len(bars), "|", bars["ts_utc"].min(), "->", bars["ts_utc"].max())

subprocess.run([sys.executable, "-m", "cwt", "build-trade-log", "--symbol", "SPY",
                "--prices-csv", r"data\spy_1h.csv", "--out", r"data\backtest_trades_rebuilt.csv"], check=True)

old = pd.read_csv(r"data\backtest_trades.csv")
new = pd.read_csv(r"data\backtest_trades_rebuilt.csv")
print("old: %d trades, mean pnl %.3f | rebuilt: %d trades, mean pnl %.3f" % (
    len(old), old["pnl"].mean(), len(new), new["pnl"].mean()))
key = ["timestamp", "lookback", "stop_atr", "mode"]
m = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
print(m["_merge"].value_counts().to_string())
both = m[m["_merge"] == "both"]
print("max |pnl diff| on shared trades:", (both["pnl_old"] - both["pnl_new"]).abs().max())
