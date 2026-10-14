import numpy as np
import pandas as pd

from cwt.metrics import block_bootstrap_sharpe, taken_vs_skipped


def test_sharpe_interval_brackets_zero_for_noise_and_excludes_it_for_strong_edge():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-01-01", periods=120)
    noise = pd.Series(rng.normal(0, 100, 120), index=idx)
    lo, hi = block_bootstrap_sharpe(noise, 100_000)
    assert lo < 0 < hi
    edge = pd.Series(rng.normal(80, 100, 120), index=idx)
    assert block_bootstrap_sharpe(edge, 100_000)[0] > 0


def _preds(skill: float, seed=2, n=4000):
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(rng.uniform(0, 100, n), unit="D")
    good = rng.random(n) < 0.5
    pnl = rng.normal(0, 50, n) + skill * good
    return pd.DataFrame({"ts_utc": ts, "pnl": pnl, "pred": np.where(good, 1.0, -1.0), "threshold": 0.0})


def test_taken_vs_skipped_detects_skill_and_not_noise():
    assert taken_vs_skipped(_preds(0.0))["lo"] < 0 < taken_vs_skipped(_preds(0.0))["hi"]
    s = taken_vs_skipped(_preds(20.0))
    assert s["lo"] > 0 and abs(s["diff"] - 20) < 6
