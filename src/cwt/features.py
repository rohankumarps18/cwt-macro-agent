"""Feature engineering. Every feature for a trade uses only information available at that trade's timestamp
(or the published event schedule). Event *outcomes* are only used after the event time."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from cwt.config import Settings
from cwt.db import read_table, to_utc_ns
from cwt.sentiment import macro_index_by_event, news_rolling

CAP_MIN = 7 * 24 * 60
IMPACT_ORD = {"low": 1, "medium": 2, "high": 3}
EVENT_TYPES = ["CPI", "EMPLOYMENT", "FOMC"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    est = df["ts_utc"].dt.tz_convert("America/New_York")
    mins = est.dt.hour * 60 + est.dt.minute
    df["hour_est"] = mins / 60.0
    df["dow"] = est.dt.dayofweek
    df["day_of_month"] = est.dt.day
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_est"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_est"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["is_cash_session"] = ((mins >= 570) & (mins < 960)).astype(int)  # 09:30-16:00 ET
    df["minutes_into_session"] = (mins - 570).clip(lower=-600, upper=600)
    return df


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    df["minutes_since_event"] = ((df["ts_utc"] - df["prev_event_ts"]).dt.total_seconds() / 60).clip(upper=CAP_MIN)
    df["minutes_to_event"] = ((df["next_event_ts"] - df["ts_utc"]).dt.total_seconds() / 60).clip(upper=CAP_MIN)
    df["pre_event_60m"] = (df["minutes_to_event"] <= 60).astype(int)
    df["post_event_30m"] = (df["minutes_since_event"] <= 30).astype(int)
    df["next_impact_ord"] = df["next_impact"].map(IMPACT_ORD)
    for c in ("prev_event_type", "next_event_type"):
        df[c] = pd.Categorical(df[c], categories=EVENT_TYPES)
    return df


def add_latest_by_type(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts_utc")
    for etype in EVENT_TYPES:
        sub = events[(events["event_type"] == etype) & events["actual"].notna()].sort_values("ts_utc")
        tag = etype.lower()
        right = pd.DataFrame({"ts_utc": to_utc_ns(sub["ts_utc"]),
                              f"last_{tag}_z": sub["signed_surprise_z"].fillna(0.0).to_numpy(),
                              f"_{tag}_ts": to_utc_ns(sub["ts_utc"])})
        if right.empty:
            df[f"last_{tag}_z"], df[f"days_since_{tag}"] = np.nan, np.nan
            continue
        df = pd.merge_asof(df, right, on="ts_utc", direction="backward")
        df[f"days_since_{tag}"] = (df["ts_utc"] - df[f"_{tag}_ts"]).dt.total_seconds() / 86400
        df = df.drop(columns=[f"_{tag}_ts"])
    return df


def add_sentiment_features(df: pd.DataFrame, events: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts_utc")
    mi = macro_index_by_event(events)
    if not mi.empty:
        df = pd.merge_asof(df, mi.sort_values("ts_utc"), on="ts_utc", direction="backward")
    else:
        df["macro_index"] = np.nan
    nr = news_rolling(news)
    if not nr.empty:
        df = pd.merge_asof(df, nr.sort_values("ts_utc"), on="ts_utc", direction="backward")
    else:
        for c in ("news_sent_3d", "news_sent_7d", "news_n_7d"):
            df[c] = np.nan
    return df


def expand_params(df: pd.DataFrame, param_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    p = pd.json_normalize(df["params"].tolist())
    p.index = df.index
    names = []
    for c in param_cols:
        c = c.lower()
        name = f"param_{c}"
        col = p[c]
        num = pd.to_numeric(col, errors="coerce")
        df[name] = num if num.notna().all() else pd.Categorical(col.astype(str))
        names.append(name)
    return df, names


def build_feature_frame(engine: Engine, settings: Settings) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Returns (frame, feature_columns, parameter_columns)."""
    ctx = read_table(engine, "SELECT * FROM trade_macro_context",
                     ts_cols=("ts_utc", "prev_event_ts", "next_event_ts"))
    if ctx.empty:
        raise RuntimeError("No trades found. Load a trading log first (cwt load-trades).")
    events = read_table(engine, "SELECT * FROM macro_events", ts_cols=("ts_utc",))
    news = read_table(engine, "SELECT url, published_at, sentiment FROM news_articles", ts_cols=("published_at",))

    df = ctx.sort_values("ts_utc").reset_index(drop=True)
    df["win"] = (df["pnl"] > 0).astype(int)
    df = add_time_features(df)
    df = add_event_features(df)
    df = add_latest_by_type(df, events)
    df = add_sentiment_features(df, events, news)
    df, param_names = expand_params(df, settings.param_cols)
    df = df.sort_values("ts_utc").reset_index(drop=True)

    time_f = ["hour_est", "dow", "day_of_month", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
              "is_cash_session", "minutes_into_session"]
    event_f = ["minutes_since_event", "minutes_to_event", "pre_event_60m", "post_event_30m",
               "next_impact_ord", "prev_event_type", "next_event_type", "prev_signed_z"]
    # FOMC surprises are always 0 (all decisions matched consensus), so last_fomc_z carries no information
    type_f = [c for t in EVENT_TYPES for c in (f"last_{t.lower()}_z", f"days_since_{t.lower()}")
              if c != "last_fomc_z"]
    sent_f = ["macro_index", "news_sent_3d", "news_sent_7d", "news_n_7d"]
    feats = time_f + event_f + type_f + sent_f + param_names
    # drop all-NaN columns (e.g. no news loaded) so they cannot silently pass as signal
    feats = [c for c in feats if not df[c].isna().all()]
    return df, feats, param_names
