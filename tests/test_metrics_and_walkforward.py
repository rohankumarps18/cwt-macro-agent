import numpy as np
import pandas as pd
import pytest

from cwt.metrics import max_drawdown, sharpe, sortino
from cwt.model import WFConfig, make_folds, walk_forward


def test_sharpe_sortino_hand_values():
    r = pd.Series([0.01, -0.02, 0.03, -0.01])
    assert sharpe(r, 252) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    downside = np.sqrt((0.02 ** 2 + 0.01 ** 2) / 4)
    assert sortino(r, 252) == pytest.approx(r.mean() / downside * np.sqrt(252))


def test_max_drawdown_hand_value():
    pnl = pd.Series([100, 100, -50, -100, 50.0])        # equity: 1100,1200,1150,1050,1100 on capital 1000
    pct, absolute = max_drawdown(pnl, 1000.0)
    assert absolute == pytest.approx(-150.0)
    assert pct == pytest.approx(1050 / 1200 - 1)


def _noise_frame(n=6000, days=150, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, days, n)), unit="D")
    df = pd.DataFrame({"trade_id": range(n), "ts_utc": ts, "strategy_id": rng.choice(list("abcd"), n),
                       "pnl": rng.normal(0, 50, n)})
    df["win"] = (df["pnl"] > 0).astype(int)
    for i in range(5):
        df[f"f{i}"] = rng.normal(size=n)
    return df


def test_folds_have_no_train_test_overlap():
    df = _noise_frame()
    cfg = WFConfig()
    for f in make_folds(df["ts_utc"], cfg):
        assert f.train_end + pd.Timedelta(days=cfg.embargo_days) == f.test_start
        assert f.test_end - f.test_start == pd.Timedelta(days=cfg.test_days)


def test_walk_forward_predictions_are_strictly_out_of_sample_and_find_nothing_in_noise():
    df = _noise_frame()
    cfg = WFConfig(lgbm=dict(n_estimators=60, learning_rate=0.05, num_leaves=7, min_child_samples=50, verbose=-1))
    preds, folds, _ = walk_forward(df, [f"f{i}" for i in range(5)], cfg)
    ts_test_start = folds.set_index("fold")["test_start"]
    assert (preds["ts_utc"].to_numpy() >= preds["fold"].map(ts_test_start).to_numpy()).all()
    assert abs(folds["ic_spearman"].mean()) < 0.08      # pure noise -> no skill
