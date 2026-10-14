"""Walk-forward (expanding window) validation with an embargo gap, and policy back-tests.

Fold k: train on trades with ts < train_end, test on trades in [test_start, test_end) where
train_end = test_start - embargo. Models are refit on every fold and never see test-window data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from cwt.metrics import fold_stats, summarize

log = logging.getLogger(__name__)


@dataclass
class WFConfig:
    train_min_days: int = 60
    test_days: int = 7
    step_days: int = 7
    embargo_days: int = 1
    min_train_rows: int = 300
    target: str = "pnl"          # "pnl" (regression) or "win" (classification)
    top_k: int = 3
    trailing_days: int = 30
    seed: int = 42
    lgbm: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1))


@dataclass
class Fold:
    k: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(ts: pd.Series, cfg: WFConfig) -> list[Fold]:
    t0 = ts.min().normalize()
    tmax = ts.max()
    test_start = t0 + pd.Timedelta(days=cfg.train_min_days + cfg.embargo_days)
    folds, k = [], 0
    while test_start <= tmax:
        folds.append(Fold(k, test_start - pd.Timedelta(days=cfg.embargo_days), test_start,
                          test_start + pd.Timedelta(days=cfg.test_days)))
        test_start += pd.Timedelta(days=cfg.step_days)
        k += 1
    return folds


def _new_model(cfg: WFConfig):
    cls = LGBMRegressor if cfg.target == "pnl" else LGBMClassifier
    return cls(random_state=cfg.seed, **cfg.lgbm)


def _predict(model, X: pd.DataFrame, cfg: WFConfig) -> np.ndarray:
    return model.predict(X) if cfg.target == "pnl" else model.predict_proba(X)[:, 1]


def walk_forward(df: pd.DataFrame, features: list[str], cfg: WFConfig):
    """Returns (predictions, fold_table, mean_feature_gain)."""
    y_col = "pnl" if cfg.target == "pnl" else "win"
    folds = make_folds(df["ts_utc"], cfg)
    preds, rows, gains = [], [], []
    for f in folds:
        tr = df[df["ts_utc"] < f.train_end]
        te = df[(df["ts_utc"] >= f.test_start) & (df["ts_utc"] < f.test_end)]
        if len(tr) < cfg.min_train_rows or te.empty:
            log.info("fold %d skipped (train=%d, test=%d)", f.k, len(tr), len(te))
            continue
        if cfg.target == "win" and tr[y_col].nunique() < 2:
            continue
        assert tr["ts_utc"].max() < f.test_start, "train/test overlap"  # leakage guard
        model = _new_model(cfg).fit(tr[features], tr[y_col])
        p = _predict(model, te[features], cfg)
        thr = 0.0 if cfg.target == "pnl" else float(tr[y_col].mean())
        out = te[["trade_id", "ts_utc", "strategy_id", "pnl", "win"]].copy()
        out["pred"], out["threshold"], out["fold"] = p, thr, f.k
        preds.append(out)
        rows.append({"fold": f.k, "train_end": f.train_end, "test_start": f.test_start,
                     "test_end": f.test_end, "n_train": len(tr), "n_test": len(te),
                     **fold_stats(te[y_col].to_numpy(), p, cfg.target)})
        gains.append(pd.Series(model.booster_.feature_importance("gain"), index=features))
    if not preds:
        raise RuntimeError("No usable walk-forward folds: not enough history for the configured windows. "
                           "Lower train_min_days/min_train_rows or supply a longer trading log.")
    gain = pd.concat(gains, axis=1).mean(axis=1)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(rows), (gain / gain.sum()).sort_values(ascending=False)


def build_policies(df: pd.DataFrame, preds: pd.DataFrame, cfg: WFConfig) -> dict[str, pd.DataFrame]:
    """Trades each policy would have taken in the out-of-sample windows."""
    pol: dict[str, list[pd.DataFrame]] = {k: [] for k in
                                           ("all_trades", "model_trade_filter", "model_top_k_permutations",
                                            "trailing_top_k_permutations")}
    folds = {f.k: f for f in make_folds(df["ts_utc"], cfg)}
    for k, g in preds.groupby("fold"):
        f = folds[k]
        base = g[["ts_utc", "strategy_id", "pnl"]]
        pol["all_trades"].append(base)
        pol["model_trade_filter"].append(base[g["pred"] > g["threshold"]])
        # permutation ranking by mean predicted score over the test window (see README caveat)
        top = g.groupby("strategy_id")["pred"].mean().nlargest(cfg.top_k).index
        pol["model_top_k_permutations"].append(base[base["strategy_id"].isin(top)])
        # naive baseline: best permutations by trailing realised P&L before train_end
        hist = df[(df["ts_utc"] < f.train_end) &
                  (df["ts_utc"] >= f.train_end - pd.Timedelta(days=cfg.trailing_days))]
        if not hist.empty:
            ttop = hist.groupby("strategy_id")["pnl"].mean().nlargest(cfg.top_k).index
            pol["trailing_top_k_permutations"].append(base[base["strategy_id"].isin(ttop)])
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=["ts_utc", "strategy_id", "pnl"]))
            for k, v in pol.items()}


def evaluate_policies(policies: dict[str, pd.DataFrame], preds: pd.DataFrame,
                      capital: float, ppy: int) -> pd.DataFrame:
    days = preds["ts_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    all_days = pd.DatetimeIndex(sorted(days.unique()))
    rows = {name: summarize(t, all_days, capital, ppy) for name, t in policies.items()}
    return pd.DataFrame(rows).T


def latest_ranking(df: pd.DataFrame, features: list[str], cfg: WFConfig, param_cols: list[str]) -> pd.DataFrame:
    """Refit on ALL history and rank permutations for the most recent window (informational, not validated)."""
    y_col = "pnl" if cfg.target == "pnl" else "win"
    model = _new_model(cfg).fit(df[features], df[y_col])
    last = df[df["ts_utc"] >= df["ts_utc"].max() - pd.Timedelta(days=cfg.test_days)].copy()
    last["pred"] = _predict(model, last[features], cfg)
    keep = ["strategy_id"] + param_cols
    return (last.groupby(keep, observed=True)["pred"].mean().sort_values(ascending=False)
            .reset_index().rename(columns={"pred": "score"}))
