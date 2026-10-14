"""Charts. All figures are saved as PNG into the output directory."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cwt.metrics import daily_pnl


def _banner(ax, synthetic: bool):
    if synthetic:
        ax.text(0.99, 0.02, "SYNTHETIC DEMO DATA", transform=ax.transAxes, ha="right", va="bottom",
                color="crimson", fontsize=9, alpha=0.8)


def sentiment_timeline(daily: pd.DataFrame, events: pd.DataFrame, out: Path, synthetic=False) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    d = daily.set_index("date")
    axes[0].plot(d.index, d["macro_index"], color="tab:blue")
    axes[0].axhline(0, color="grey", lw=0.6)
    axes[0].set_ylabel("Macro surprise index\n(+ hotter / - softer)")
    ev = events[events["actual"].notna()]
    colors = {"CPI": "tab:red", "EMPLOYMENT": "tab:green", "FOMC": "tab:purple"}
    for t, g in ev.groupby("event_type"):
        axes[0].scatter(g["ts_utc"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize(),
                        np.zeros(len(g)), s=14, color=colors.get(t, "k"), label=t, zorder=3)
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    axes[1].plot(d.index, d["news_sent_7d"], color="tab:orange")
    axes[1].axhline(0, color="grey", lw=0.6)
    axes[1].set_ylabel("News sentiment\n(7d VADER mean)")
    axes[2].plot(d.index, d["composite"], color="black")
    axes[2].axhline(0, color="grey", lw=0.6)
    axes[2].set_ylabel("Composite\n(expanding z)")
    fig.suptitle("Macro sentiment over time")
    _banner(axes[2], synthetic)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    p = out / "01_macro_sentiment_timeline.png"
    try:
        fig.savefig(str(p), dpi=140, format="png")
    finally:
        plt.close(fig)
    return p


def surprise_matrix(events: pd.DataFrame, out: Path, synthetic=False) -> Path:
    ev = events[events["actual"].notna()].copy()
    ev["week"] = ev["ts_utc"].dt.tz_localize(None).dt.to_period("W").dt.start_time
    piv = ev.pivot_table(index="event_name", columns="week", values="signed_surprise_z", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, max(3, 0.45 * len(piv) + 1.5)))
    vmax = max(1.0, np.nanmax(np.abs(piv.to_numpy()))) if piv.notna().any().any() else 1.0
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(piv)), [n[:38] for n in piv.index], fontsize=8)
    step = max(1, len(piv.columns) // 12)
    ax.set_xticks(range(0, len(piv.columns), step), [c.strftime("%b %d") for c in piv.columns[::step]],
                  rotation=45, fontsize=8)
    fig.colorbar(im, ax=ax, label="signed surprise z (blank = no print / z unavailable)")
    ax.set_title("Macro surprise matrix (event x week)")
    _banner(ax, synthetic)
    fig.tight_layout()
    p = out / "02_surprise_matrix.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def equity_curves(policies: dict, all_days, capital: float, out: Path, synthetic=False) -> Path:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for name, taken in policies.items():
        dp = daily_pnl(taken, all_days)
        eq = capital + dp.cumsum()
        a1.plot(eq.index, eq.to_numpy(), label=name)
        a2.plot(eq.index, (eq / eq.cummax().clip(lower=capital) - 1).to_numpy() * 100, label=name)
    a1.set_ylabel("Equity")
    a1.legend(fontsize=8)
    a2.set_ylabel("Drawdown %")
    a1.set_title("Out-of-sample equity (walk-forward)")
    _banner(a2, synthetic)
    fig.autofmt_xdate()
    fig.tight_layout()
    p = out / "03_equity_drawdown.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def regime_pnl_matrix(df: pd.DataFrame, out: Path, synthetic=False) -> Path:
    d = df.dropna(subset=["macro_index", "minutes_to_event"]).copy()
    if d.empty:
        return out / "04_regime_pnl_matrix.png"
    d["regime"] = pd.cut(d["macro_index"], [-np.inf, -0.25, 0.25, np.inf], labels=["soft", "neutral", "hot"])
    d["timing"] = pd.cut(d["minutes_to_event"], [-1, 60, 360, 1440, 1e9],
                         labels=["<1h to event", "1-6h", "6-24h", ">24h"])
    piv = d.pivot_table(index="regime", columns="timing", values="pnl", aggfunc="mean", observed=True)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    vmax = max(1e-9, np.nanmax(np.abs(piv.to_numpy())))
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, fontsize=8)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if not np.isnan(piv.iat[i, j]):
                ax.text(j, i, f"{piv.iat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean P&L per trade")
    ax.set_title("Mean trade P&L: macro regime x time to next event")
    _banner(ax, synthetic)
    fig.tight_layout()
    p = out / "04_regime_pnl_matrix.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def feature_importance(gain: pd.Series, out: Path, synthetic=False, top: int = 15) -> Path:
    g = gain.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(g.index, g.to_numpy())
    ax.set_xlabel("share of total gain (mean over folds)")
    ax.set_title("Feature importance")
    _banner(ax, synthetic)
    fig.tight_layout()
    p = out / "05_feature_importance.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p
