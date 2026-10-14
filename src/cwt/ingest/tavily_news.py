"""Fetch macro news with Tavily in weekly windows and score sentiment with VADER."""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd
from tavily import TavilyClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from cwt.config import Settings, require

log = logging.getLogger(__name__)
DEFAULT_QUERIES = [
    "US CPI inflation report",
    "FOMC Federal Reserve interest rate decision",
    "US jobs report nonfarm payrolls unemployment",
]
_analyzer = SentimentIntensityAnalyzer()


def score_sentiment(title: str | None, content: str | None) -> float:
    text = f"{title or ''}. {(content or '')[:1000]}"
    return float(_analyzer.polarity_scores(text)["compound"])


def _articles_from_response(resp: dict, query: str, window_end: date, source: str) -> list[dict]:
    out = []
    for r in resp.get("results", []):
        pub = pd.to_datetime(r.get("published_date"), utc=True, errors="coerce")
        inferred = "no"
        if pd.isna(pub):
            # Undated result: place it at the END of the search window so it can never appear
            # earlier than it was really published (avoids look-ahead).
            pub = pd.Timestamp(window_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
            inferred = "yes"
        out.append({
            "url": r["url"], "title": r.get("title"), "content": r.get("content"),
            "published_at": pub, "date_inferred": inferred, "relevance": r.get("score"),
            "query": query, "sentiment": score_sentiment(r.get("title"), r.get("content")),
            "source": source,
        })
    return out


def fetch_news(settings: Settings, queries: list[str] | None = None, window_days: int = 7,
               max_results: int = 10, source: str = "tavily") -> pd.DataFrame:
    client = TavilyClient(api_key=require(settings.tavily_api_key, "TAVILY_API_KEY"))
    start, end = settings.window()
    queries = queries or DEFAULT_QUERIES
    rows, calls, failures = [], 0, 0
    cur = start
    while cur < end:
        win_end = min(cur + timedelta(days=window_days), end)
        for q in queries:
            calls += 1
            try:
                resp = client.search(query=q, topic="news", search_depth="basic", max_results=max_results,
                                     start_date=cur.isoformat(), end_date=win_end.isoformat())
                rows += _articles_from_response(resp, q, win_end, source)
            except Exception as exc:  # network/quota errors: keep going, report at the end
                failures += 1
                log.warning("Tavily call failed (%s, %s..%s): %s", q, cur, win_end, exc)
            time.sleep(0.2)
        cur = win_end
    if calls and failures == calls:
        raise RuntimeError("Every Tavily call failed - check TAVILY_API_KEY, quota and network.")
    log.info("Tavily: %d calls, %d failed, %d articles", calls, failures, len(rows))
    df = pd.DataFrame(rows)
    return df.drop_duplicates("url") if not df.empty else df


def load_news_csv(path: str, source: str = "csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["published_at"] = pd.to_datetime(df["published_date"], utc=True)
    df["date_inferred"] = "no"
    df["relevance"] = df.get("score")
    df["query"] = df.get("query", "csv")
    df["sentiment"] = [score_sentiment(t, c) for t, c in zip(df["title"], df["content"])]
    df["source"] = source
    return df[["url", "title", "content", "published_at", "date_inferred", "relevance", "query",
               "sentiment", "source"]].drop_duplicates("url")
