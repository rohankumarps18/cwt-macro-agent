import sys
import numpy as np
import pandas as pd

p = pd.read_csv(sys.argv[1])
p["ts_utc"] = pd.to_datetime(p["ts_utc"], utc=True)
et = p["ts_utc"].dt.tz_convert("America/New_York")
p["day"] = et.dt.tz_localize(None).dt.normalize()
p["hour"] = et.dt.hour
p["taken"] = p["pred"] > p["threshold"]
p["mom"] = p["strategy_id"].str.contains("mode=momentum")

def cell_excess(keys):
    m = p.groupby(keys)["pnl"].transform("mean")
    return np.where(p["taken"], p["pnl"] - m, 0.0)

rng = np.random.default_rng(0)
for label, keys in [("day", ["day"]), ("day+mode", ["day", "mom"])]:
    p["e"] = cell_excess(keys)
    d = p.groupby("day").agg(e=("e", "sum"), k=("taken", "sum"))
    e, k = d["e"].to_numpy(), d["k"].to_numpy()
    idx = rng.integers(0, len(d), (5000, len(d)))
    boot = e[idx].sum(axis=1) / k[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("%-9s excess per taken trade vs cell mean: %.3f | day-clustered 95%% CI [%.3f, %.3f] | days positive: %.2f" % (
        label, e.sum() / k.sum(), lo, hi, (e[k > 0] > 0).mean()))

p["e"] = cell_excess(["day", "mom"])
t = p[p["taken"]]
print()
print("excess vs day+mode cell mean, taken trades, by mode:")
print(t.groupby("mom")["e"].agg(["count", "mean"]).round(3).to_string())
print()
print("by ET hour:")
h = p.groupby("hour").agg(n=("taken", "size"), taken_share=("taken", "mean"), mean_pnl=("pnl", "mean"))
h["excess_taken"] = t.groupby("hour")["e"].mean()
print(h.round(3).to_string())
print()
g = t.groupby("ts_utc")["mom"].agg(["sum", "count"])
both = ((g["sum"] > 0) & (g["sum"] < g["count"])).mean()
print("taken timestamps holding both momentum and meanrev trades: %.2f" % both)
