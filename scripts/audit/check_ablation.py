import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from cwt.pipeline import _engine
from cwt.features import build_feature_frame
from cwt.model import WFConfig, walk_forward

s, eng = _engine()
df, feats, param_names = build_feature_frame(eng, s)
cfg = WFConfig()
TIME = ["hour_est", "dow", "day_of_month", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_cash_session", "minutes_into_session"]
time_f = [c for c in feats if c in TIME]
param_f = [c for c in feats if c in param_names]
macro_f = [c for c in feats if c not in TIME and c not in param_names]
sets = {"all": feats, "param_only": param_f, "param+time": param_f + time_f,
        "time+macro (no params)": time_f + macro_f, "macro_only": macro_f}

def ic(a, b):
    return float(spearmanr(a, b).statistic) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else float("nan")

for name, fs in sets.items():
    if not fs:
        continue
    p, folds, _ = walk_forward(df, fs, cfg)
    p["day"] = p["ts_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    p["mom"] = p["strategy_id"].str.contains("mode=momentum")
    t = p["pred"] > p["threshold"]
    g = p.groupby(["day", "mom"])
    dp = p["pred"] - g["pred"].transform("mean")
    dy = p["pnl"] - g["pnl"].transform("mean")
    print("%-24s folds=%d taken=%4d meanPnL(taken)=%7.3f mom_share=%.2f pooledIC=%6.3f withinDayModeIC=%6.3f" % (
        name, len(folds), t.sum(), p.loc[t, "pnl"].mean(), p.loc[t, "mom"].mean(),
        ic(p["pred"], p["pnl"]), ic(dp, dy)))
