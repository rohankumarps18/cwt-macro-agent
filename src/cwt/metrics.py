"""Performance metrics. Daily P&L is divided by initial capital to give (non-compounded) daily returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sharpe(r: pd.Series, ppy: int = 252) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if len(r) > 1 and sd > 1e-12 else float("nan")


def sortino(r: pd.Series, ppy: int = 252) -> float:
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    return float(r.mean() / downside * np.sqrt(ppy)) if len(r) > 1 and downside > 1e-12 else float("nan")


def max_drawdown(daily_pnl: pd.Series, capital: float) -> tuple[float, float]:
    """Returns (max drawdown as a fraction of peak equity, max drawdown in currency)."""
    equity = capital + daily_pnl.cumsum()
    peak = equity.cummax().clip(lower=capital)
    dd = equity / peak - 1.0
    return float(dd.min()), float((equity - peak).min())


def daily_pnl(taken: pd.DataFrame, all_days: pd.DatetimeIndex) -> pd.Series:
    """Sum P&L per New-York calendar day; days with no taken trades are 0 (same index for every policy)."""
    if taken.empty:
        return pd.Series(0.0, index=all_days)
    d = taken["ts_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return taken.groupby(d)["pnl"].sum().reindex(all_days, fill_value=0.0)


def summarize(taken: pd.DataFrame, all_days: pd.DatetimeIndex, capital: float, ppy: int) -> dict:
    dp = daily_pnl(taken, all_days)
    r = dp / capital
    mdd_pct, mdd_abs = max_drawdown(dp, capital)
    wins, losses = taken.loc[taken["pnl"] > 0, "pnl"].sum(), -taken.loc[taken["pnl"] < 0, "pnl"].sum()
    return {
        "n_trades": int(len(taken)),
        "total_pnl": float(taken["pnl"].sum()) if len(taken) else 0.0,
        "avg_pnl_per_trade": float(taken["pnl"].mean()) if len(taken) else float("nan"),
        "hit_rate": float((taken["pnl"] > 0).mean()) if len(taken) else float("nan"),
        "profit_factor": float(wins / losses) if losses > 0 else float("nan"),
        "sharpe": sharpe(r, ppy), "sortino": sortino(r, ppy),
        "max_drawdown_pct": mdd_pct, "max_drawdown_abs": mdd_abs,
        "n_days": int(len(all_days)),
    }


def fold_stats(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    out = {}
    if len(y_true) > 2 and np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12:
        out["ic_spearman"] = float(spearmanr(y_pred, y_true).statistic)
    else:
        out["ic_spearman"] = float("nan")
    if target == "pnl":
        out["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        out["mae"] = float(np.mean(np.abs(y_true - y_pred)))
        out["directional_acc"] = float(np.mean((y_true > 0) == (y_pred > 0)))
    else:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) == 2 else float("nan")
    return out


def block_bootstrap_sharpe(dp: pd.Series, capital: float, ppy: int = 252, block: int = 5,
                           n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for the annualised Sharpe via a moving-block bootstrap of daily returns."""
    r = (dp / capital).to_numpy()
    n = len(r)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        s = np.concatenate([r[i:i + block] for i in starts])[:n]
        sd = s.std(ddof=1)
        out.append(s.mean() / sd * np.sqrt(ppy) if sd > 1e-12 else np.nan)
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def taken_vs_skipped(preds: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean P&L of trades the model takes minus trades it skips, with a day-clustered bootstrap 95% interval.
    A model with real skill should have taken > skipped (difference > 0 and an interval above 0)."""
    day = preds["ts_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    taken = (preds["pred"] > preds["threshold"]).to_numpy()
    df = pd.DataFrame({"day": day.to_numpy(), "pnl": preds["pnl"].to_numpy(), "t": taken})
    g = df.groupby("day")
    sT = g.apply(lambda x: x.loc[x.t, "pnl"].sum()).to_numpy()
    nT = g.apply(lambda x: float(x.t.sum())).to_numpy()
    sS = g.apply(lambda x: x.loc[~x.t, "pnl"].sum()).to_numpy()
    nS = g.apply(lambda x: float((~x.t).sum())).to_numpy()

    def diff(idx):
        a, b = nT[idx].sum(), nS[idx].sum()
        return sT[idx].sum() / a - sS[idx].sum() / b if a > 0 and b > 0 else np.nan

    all_idx = np.arange(len(sT))
    rng = np.random.default_rng(seed)
    boots = [diff(rng.integers(0, len(sT), len(sT))) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"diff": float(diff(all_idx)), "lo": float(lo), "hi": float(hi),
            "mean_taken": float(sT.sum() / nT.sum()) if nT.sum() else float("nan"),
            "mean_skipped": float(sS.sum() / nS.sum()) if nS.sum() else float("nan")}
