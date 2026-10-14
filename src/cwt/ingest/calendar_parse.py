"""Normalise raw economic-calendar records (from any Apify actor or CSV) into one schema.

Actors differ in field names, so we resolve through alias lists. Anything we cannot map is kept
in `raw` so nothing is silently lost.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from cwt.db import stable_id

# Order matters: first match wins.
EVENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("FOMC", re.compile(r"fomc|federal funds|fed interest rate|interest rate decision", re.I)),
    ("CPI", re.compile(r"\bcpi\b|consumer price", re.I)),
    ("EMPLOYMENT", re.compile(
        r"non[- ]?farm|\bnfp\b|unemployment rate|average hourly earnings|adp (non[- ]?farm|employment)", re.I)),
]
# Speeches are not scheduled data releases or decisions; they add noise to the FOMC bucket.
EXCLUDE = re.compile(r"\bspeaks?\b|\bspeech(es)?\b", re.I)
# Higher-than-forecast reading is "hawkish / hotter economy" (+1) except unemployment (-1).
NEGATIVE_SIGN = re.compile(r"unemployment rate", re.I)

ALIASES = {
    "datetime": ["datetime", "datetimeiso", "timestamp", "date_time", "datetimeutc", "utc_datetime"],
    "date": ["date", "day"],
    "time": ["time", "release_time"],
    "name": ["event", "event_name", "title", "name", "indicator"],
    "country": ["country", "region", "currency"],
    "impact": ["impact", "importance", "volatility", "priority"],
    "actual": ["actual", "actual_value"],
    "forecast": ["forecast", "consensus", "forecast_value", "expected"],
    "previous": ["previous", "prior", "previous_value"],
}
_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def classify(name: str) -> str | None:
    if EXCLUDE.search(name or ""):
        return None
    for etype, pat in EVENT_PATTERNS:
        if pat.search(name or ""):
            return etype
    return None


def sign_for(name: str) -> int:
    return -1 if NEGATIVE_SIGN.search(name or "") else 1


def parse_value(x: Any) -> float:
    """'3.2%' -> 3.2, '225K' -> 225000, '-' / '' / None -> NaN."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("$", "").replace("%", "")
    if s in {"", "-", "--", "n/a", "N/A", "nan", "None"}:
        return np.nan
    mult = 1.0
    if s[-1].lower() in _SUFFIX:
        mult = _SUFFIX[s[-1].lower()]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def normalize_impact(x: Any) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"3", "high", "red"} or s.startswith("high"):
        return "high"
    if s in {"2", "medium", "med", "orange", "moderate"} or s.startswith("med"):
        return "medium"
    if s in {"1", "low", "yellow"} or s.startswith("low"):
        return "low"
    return None


def _pick(rec: dict, key: str):
    for alias in ALIASES[key]:
        if alias in rec and rec[alias] not in (None, ""):
            return rec[alias]
    return None


def _to_utc(rec: dict, source_tz: str) -> pd.Timestamp:
    """Timestamp -> UTC. tz-aware inputs are converted; naive inputs are localised to source_tz."""
    dt = _pick(rec, "datetime")
    if dt is None:
        d, t = _pick(rec, "date"), _pick(rec, "time")
        if d is None or t is None:
            return pd.NaT
        dt = f"{d} {t}"
    ts = pd.to_datetime(dt, errors="coerce")
    if pd.isna(ts):  # e.g. "All Day", "Tentative"
        return pd.NaT
    if ts.tzinfo is None:
        ts = ts.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
        if pd.isna(ts):
            return pd.NaT
    return ts.tz_convert("UTC")


def add_surprise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """surprise = actual - forecast; z uses only *earlier* surprises of the same event (no look-ahead).

    NOTE: with ~6 observations per monthly event in 180 days, the first 3 prints have no z-score (NaN).
    """
    df = df.sort_values("ts_utc").reset_index(drop=True).copy()
    df["surprise"] = df["actual"] - df["forecast"]
    z = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby("event_name").groups.items():
        s = df.loc[idx, "surprise"]
        prior_std = s.expanding(min_periods=3).std().shift(1)
        zz = (s / prior_std).where(prior_std > 1e-12)
        z.loc[idx] = zz.clip(-5, 5)
    df["surprise_z"] = z
    df["signed_surprise_z"] = df["surprise_z"] * df["event_name"].map(sign_for)
    return df


def normalize_records(records: Iterable[dict], source_tz: str, countries: list[str],
                      source: str) -> pd.DataFrame:
    allowed = {c.lower() for c in countries}
    rows = []
    for raw in records:
        rec = {str(k).lower(): v for k, v in raw.items()}
        name = _pick(rec, "name")
        etype = classify(str(name)) if name else None
        if etype is None:
            continue
        country = _pick(rec, "country")
        if country is not None and allowed and str(country).strip().lower() not in allowed:
            continue
        ts = _to_utc(rec, source_tz)
        if pd.isna(ts):
            continue
        rows.append({
            "ts_utc": ts, "event_name": str(name).strip(), "event_type": etype,
            "country": None if country is None else str(country),
            "impact": normalize_impact(_pick(rec, "impact")),
            "actual": parse_value(_pick(rec, "actual")),
            "forecast": parse_value(_pick(rec, "forecast")),
            "previous": parse_value(_pick(rec, "previous")),
            "source": source, "raw": raw,
        })
    if not rows:
        return pd.DataFrame(columns=["event_id", "ts_utc", "event_name", "event_type", "country", "impact",
                                     "actual", "forecast", "previous", "surprise", "surprise_z",
                                     "signed_surprise_z", "source", "raw"])
    df = pd.DataFrame(rows)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["event_id"] = [stable_id(t.isoformat(), n.lower(), c) for t, n, c in
                      zip(df["ts_utc"], df["event_name"], df["country"])]
    df = df.drop_duplicates("event_id")
    return add_surprise_columns(df)
