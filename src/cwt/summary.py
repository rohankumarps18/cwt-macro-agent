"""One-page macro outlook, generated deterministically from the stored data (no LLM, no invention)."""
from __future__ import annotations

import pandas as pd

from cwt.sentiment import daily_sentiment


def _fmt(x, nd=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def build_summary(events: pd.DataFrame, news: pd.DataFrame, synthetic: bool = False, n_headlines: int = 5) -> str:
    daily = daily_sentiment(events, news)
    lines = ["# Macro Outlook (1-page summary)", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DEMO DATA. Not a real market view.**", ""]
    done = events[events["actual"].notna()].sort_values("ts_utc")
    lines.append(f"*Data window:* {done['ts_utc'].min():%Y-%m-%d} to {done['ts_utc'].max():%Y-%m-%d} "
                 f"({len(done)} released events, {len(news)} news articles). Auto-generated from the database.")
    lines += ["", "## Latest prints"]
    lines += ["| Event | Date (UTC) | Actual | Forecast | Surprise | Signed z |", "|---|---|---|---|---|---|"]
    latest = done.groupby("event_name").tail(1).sort_values("ts_utc", ascending=False).head(8)
    for _, r in latest.iterrows():
        lines.append(f"| {r['event_name'][:40]} | {r['ts_utc']:%Y-%m-%d %H:%M} | {_fmt(r['actual'])} | "
                     f"{_fmt(r['forecast'])} | {_fmt(r['surprise'])} | {_fmt(r['signed_surprise_z'])} |")
    lines += ["", "## Sentiment regime"]
    if not daily.empty and daily["macro_index"].notna().any():
        last, ago = daily.iloc[-1], daily.iloc[max(0, len(daily) - 31)]
        mi, mi0 = last["macro_index"], ago["macro_index"]
        tone = "hotter / stronger than forecast" if mi > 0.25 else "softer than forecast" if mi < -0.25 else "broadly in line"
        trend = "rising" if mi - mi0 > 0.15 else "falling" if mi - mi0 < -0.15 else "flat"
        lines.append(f"- Macro surprise index: **{_fmt(mi)}** ({tone}); 30-day change {_fmt(mi - mi0)} ({trend}).")
        lines.append(f"- News sentiment (7d VADER): **{_fmt(last['news_sent_7d'])}** vs 30 days ago {_fmt(ago['news_sent_7d'])}.")
        lines.append(f"- Composite z-score: {_fmt(last['composite'])}.")
    else:
        lines.append("- Not enough released events to compute a regime.")
    upcoming = events[events["ts_utc"] > pd.Timestamp.now(tz="UTC")].sort_values("ts_utc").head(5)
    lines += ["", "## Upcoming events in the data"]
    lines += ([f"- {r['ts_utc']:%Y-%m-%d %H:%M} UTC: {r['event_name']} (forecast {_fmt(r['forecast'])})"
               for _, r in upcoming.iterrows()] or ["- None in the scraped window."])
    lines += ["", "## Recent headlines (Tavily)"]
    if news.empty:
        lines.append("- No news loaded.")
    else:
        top = news.sort_values("published_at", ascending=False).head(n_headlines)
        lines += [f"- {r['published_at']:%Y-%m-%d}: {r['title']} ({_fmt(r['sentiment'])})" for _, r in top.iterrows()]
    lines += ["", "## Caveats",
              "- Sentiment is a simple signed-surprise z-score plus a lexicon score on news text; it is a descriptive "
              "indicator, not a forecast.",
              "- Early prints have no z-score (min. 3 prior observations per event), so they count as 0."]
    return "\n".join(lines) + "\n"
