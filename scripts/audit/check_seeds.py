import numpy as np
from scipy.stats import spearmanr
from cwt.pipeline import _engine
from cwt.features import build_feature_frame
from cwt.model import WFConfig, walk_forward

s, eng = _engine()
df, feats, param_names = build_feature_frame(eng, s)
TIME = ["hour_est","dow","day_of_month","hour_sin","hour_cos","dow_sin","dow_cos","is_cash_session","minutes_into_session"]
sets = {"all": feats, "param+time": [c for c in feats if c in param_names or c in TIME]}
for name, fs in sets.items():
    for seed in range(5):
        p, folds, _ = walk_forward(df, fs, WFConfig(seed=seed))
        t = p["pred"] > p["threshold"]
        print("%-11s seed=%d taken=%4d meanPnL(taken)=%7.3f pooledIC=%6.3f" % (
            name, seed, t.sum(), p.loc[t, "pnl"].mean(), spearmanr(p["pred"], p["pnl"]).statistic))
