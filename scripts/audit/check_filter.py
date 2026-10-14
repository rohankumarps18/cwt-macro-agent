import sys
import numpy as np
import pandas as pd

p = pd.read_csv(sys.argv[1])
p["ts_utc"] = pd.to_datetime(p["ts_utc"], utc=True)
p["day"] = p["ts_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
p["taken"] = p["pred"] > p["threshold"]
p["mom"] = p["strategy_id"].str.contains("mode=momentum")

print("example strategy_ids:", list(p["strategy_id"].unique()[:3]), "| distinct:", p["strategy_id"].nunique())
print("OOS trades:", len(p), "| OOS days:", p["day"].nunique(), "| taken:", int(p["taken"].sum()))
print("momentum share: overall %.2f, among taken %.2f" % (p["mom"].mean(), p.loc[p["taken"], "mom"].mean()))
print("mean pnl  all: %.3f | taken: %.3f | skipped: %.3f" % (
    p["pnl"].mean(), p.loc[p["taken"], "pnl"].mean(), p.loc[~p["taken"], "pnl"].mean()))
print("mean pnl  always-momentum: %.3f | always-meanrev: %.3f" % (
    p.loc[p["mom"], "pnl"].mean(), p.loc[~p["mom"], "pnl"].mean()))

pnl = p["pnl"].to_numpy()
taken = p["taken"].to_numpy()
obs = pnl[taken].mean()
rng = np.random.default_rng(0)

def perm_test(keys, n=2000):
    groups = list(p.groupby(keys).indices.values())
    null = np.empty(n)
    for b in range(n):
        sel = np.zeros(len(p), dtype=bool)
        for ix in groups:
            k = int(taken[ix].sum())
            if k:
                sel[rng.choice(ix, k, replace=False)] = True
        null[b] = pnl[sel].mean()
    return null.mean(), (null >= obs).mean()

for label, keys in [("random, same count per day", ["day"]),
                    ("random, same count per day+mode", ["day", "mom"])]:
    m, pv = perm_test(keys)
    print("%-34s observed %.3f | null mean %.3f | p(null>=obs) = %.3f" % (label, obs, m, pv))
