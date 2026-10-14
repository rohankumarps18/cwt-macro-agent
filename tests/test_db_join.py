"""Integration test against a real Postgres. Set TEST_DATABASE_URL (a throw-away database); skipped otherwise."""
import os

import pandas as pd
import pytest
from sqlalchemy import text

from cwt import db

URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


def test_context_view_uses_only_past_outcomes_and_future_schedule():
    eng = db.get_engine(URL)
    db.init_db(eng)
    with eng.begin() as c:
        c.execute(text("TRUNCATE macro_events, news_articles, trades"))
    ev = pd.DataFrame({
        "event_id": ["e1", "e2"], "event_name": ["CPI m/m", "CPI m/m"], "event_type": ["CPI", "CPI"],
        "ts_utc": pd.to_datetime(["2026-03-10 12:30", "2026-04-10 12:30"], utc=True),
        "actual": [0.4, None], "forecast": [0.3, 0.3], "signed_surprise_z": [1.5, None],
        "impact": ["high", "high"], "source": "test"})
    tr = pd.DataFrame({
        "trade_id": ["t_before", "t_after", "t_between_ev2"],
        "ts_utc": pd.to_datetime(["2026-03-10 12:00", "2026-03-10 13:00", "2026-04-11 00:00"], utc=True),
        "strategy_id": "s", "pnl": 1.0, "params": [{"a": 1}] * 3, "source": "test"})
    db.upsert(eng, db.macro_events, ev, "event_id")
    db.upsert(eng, db.trades, tr, "trade_id")
    ctx = db.read_table(eng, "SELECT * FROM trade_macro_context", ts_cols=("ts_utc", "prev_event_ts", "next_event_ts"))
    ctx = ctx.set_index("trade_id")
    assert pd.isna(ctx.loc["t_before", "prev_event_ts"])                   # outcome not yet released
    assert ctx.loc["t_before", "next_event_ts"] == pd.Timestamp("2026-03-10 12:30", tz="UTC")
    assert ctx.loc["t_after", "prev_signed_z"] == pytest.approx(1.5)       # known once released
    assert ctx.loc["t_after", "next_event_ts"] == pd.Timestamp("2026-04-10 12:30", tz="UTC")
    # event 2 has no actual yet, so it is never exposed as an outcome
    assert ctx.loc["t_between_ev2", "prev_event_ts"] == pd.Timestamp("2026-03-10 12:30", tz="UTC")
    # re-upserting is idempotent
    db.upsert(eng, db.trades, tr, "trade_id")
    assert len(db.read_table(eng, "SELECT trade_id FROM trades")) == 3
