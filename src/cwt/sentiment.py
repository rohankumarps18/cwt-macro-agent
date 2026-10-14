"""Macro sentiment series.

macro_index: time-decayed (14-day half-life) average of signed surprise z-scores, evaluated at each
event time. Positive = data hotter/stronger than forecast (hawkish pressure); negative = softer.
news_sentiment: rolling VADER mean of Tavily articles.
composite: average of the two, each z-scored with *expanding* statistics (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cwt.db import to_utc_ns


def macro_index_by_event(events: pd.DataFrame, halflife: str = "14D") -> pd.DataFrame:
    ev = events[events["actual"].notna()].sort_values("ts_utc")
    if ev.empty:
        return pd.DataFrame(columns=["ts_utc", "macro_index"])
    z = ev["signed_surprise_z"].fillna(0.0).to_numpy()
    s = pd.Series(z, index=pd.DatetimeIndex(ev["ts_utc"]))
    out = s.ewm(halflife=pd.Timedelta(halflife), times=s.index).mean()
    res = pd.DataFrame({"ts_utc": to_utc_ns(pd.Series(s.index)), "macro_index": out.to_numpy()})
    return res


def news_rolling(news: pd.DataFrame) -> pd.DataFrame:
    """Rolling means evaluated at each article's own timestamp (uses only articles up to then)."""
    if news.empty:
        return pd.DataFrame(columns=["ts_utc", "news_sent_3d", "news_sent_7d", "news_n_7d"])
    n = news.dropna(subset=["published_at"]).copy()
    ts = pd.to_datetime(to_utc_ns(n["published_at"]), utc=True)
    # date-only stamps (exactly 00:00:00 UTC) could be seen up to a day early: move them to end of that day
    ts = ts.where(ts != ts.dt.normalize(), ts + pd.Timedelta(hours=23, minutes=59, seconds=59))
    n["published_at"] = ts
    n = n.sort_values("published_at").set_index("published_at")
    out = pd.DataFrame({
        "news_sent_3d": n["sentiment"].rolling("3D").mean(),
        "news_sent_7d": n["sentiment"].rolling("7D").mean(),
        "news_n_7d": n["sentiment"].rolling("7D").count(),
    }).reset_index().rename(columns={"published_at": "ts_utc"})
    # many articles can share a timestamp; keep the last (which includes all of them)
    return out.drop_duplicates("ts_utc", keep="last")


def _expanding_z(s: pd.Series) -> pd.Series:
    mu = s.expanding(min_periods=5).mean()
    sd = s.expanding(min_periods=5).std()
    return ((s - mu) / sd.where(sd > 1e-12)).clip(-4, 4)


def daily_sentiment(events: pd.DataFrame, news: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Daily grid for charts: last known macro_index and 7d news sentiment, plus composite."""
    mi, nr = macro_index_by_event(events), news_rolling(news)
    stamps = pd.concat([mi["ts_utc"], nr["ts_utc"]]).dropna()
    if stamps.empty:
        return pd.DataFrame()
    start = pd.Timestamp(start or stamps.min()).tz_convert("UTC").normalize()
    end = pd.Timestamp(end or stamps.max()).tz_convert("UTC").normalize()
    grid = pd.DataFrame({"ts_utc": to_utc_ns(pd.Series(pd.date_range(start, end, freq="D", tz="UTC")
                                                    + pd.Timedelta(hours=23, minutes=59)))})
    out = grid
    if not mi.empty:
        out = pd.merge_asof(out, mi.sort_values("ts_utc"), on="ts_utc", direction="backward")
    else:
        out["macro_index"] = np.nan
    if not nr.empty:
        out = pd.merge_asof(out, nr.sort_values("ts_utc"), on="ts_utc", direction="backward")
    else:
        out[["news_sent_3d", "news_sent_7d", "news_n_7d"]] = np.nan
    out["date"] = out["ts_utc"].dt.normalize()
    out["composite"] = pd.concat([_expanding_z(out["macro_index"]), _expanding_z(out["news_sent_7d"])],
                                 axis=1).mean(axis=1, skipna=True)
    return out
