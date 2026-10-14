import sys
import pandas as pd

p = pd.read_csv(sys.argv[1])
p["ts_utc"] = pd.to_datetime(p["ts_utc"], utc=True)
p["day"] = p["ts_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
p["taken"] = p["pred"] > p["threshold"]
p["mom"] = p["strategy_id"].str.contains("mode=momentum")
p["stop"] = p["strategy_id"].str.extract(r"stop_atr=([\d.]+)")[0]
p["lb"] = p["strategy_id"].str.extract(r"lookback=(\d+)")[0]

def expected_null(keys):
    c = p.groupby(keys).agg(k=("taken", "sum"), m=("pnl", "mean"))
    return (c["k"] * c["m"]).sum() / c["k"].sum()

print("analytic null, day:      %.3f (sim was -4.480)" % expected_null(["day"]))
print("analytic null, day+mode: %.3f (sim was -5.033)" % expected_null(["day", "mom"]))
print()
for name, g in p.groupby("mom"):
    print("%-9s n=%4d taken=%4d | all %.2f | taken %.2f | skipped %.2f" % (
        "momentum" if name else "meanrev", len(g), g["taken"].sum(), g["pnl"].mean(),
        g.loc[g["taken"], "pnl"].mean(), g.loc[~g["taken"], "pnl"].mean()))
print()
print("taken share by stop_atr:\n", p.groupby("stop")["taken"].mean().round(2).to_string())
print("taken share by lookback:\n", p.groupby("lb")["taken"].mean().round(2).to_string())
print()
d = p.groupby("day").agg(n=("taken", "sum"), m=("pnl", "mean"))
print("corr(taken count per day, mean pnl that day): %.3f" % d["n"].corr(d["m"]))
