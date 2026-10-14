"""Postgres layer: schema, idempotent upserts, and the SQL join view."""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd
from sqlalchemy import (Column, DateTime, Float, MetaData, String, Table, Text,
                        create_engine, text)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)
metadata = MetaData()

macro_events = Table(
    "macro_events", metadata,
    Column("event_id", String(40), primary_key=True),
    Column("ts_utc", DateTime(timezone=True), nullable=False, index=True),
    Column("event_name", Text, nullable=False),
    Column("event_type", String(20), nullable=False),
    Column("country", String(40)),
    Column("impact", String(20)),
    Column("actual", Float),
    Column("forecast", Float),
    Column("previous", Float),
    Column("surprise", Float),
    Column("surprise_z", Float),
    Column("signed_surprise_z", Float),
    Column("source", String(20), nullable=False),
    Column("raw", JSONB),
)

news_articles = Table(
    "news_articles", metadata,
    Column("url", Text, primary_key=True),
    Column("title", Text),
    Column("content", Text),
    Column("published_at", DateTime(timezone=True), index=True),
    Column("date_inferred", String(5)),
    Column("relevance", Float),
    Column("query", Text),
    Column("sentiment", Float),
    Column("source", String(20), nullable=False),
)

trades = Table(
    "trades", metadata,
    Column("trade_id", String(40), primary_key=True),
    Column("ts_utc", DateTime(timezone=True), nullable=False, index=True),
    Column("strategy_id", String(200), nullable=False, index=True),
    Column("pnl", Float, nullable=False),
    Column("params", JSONB, nullable=False),
    Column("source", String(20), nullable=False),
)

# Each trade joined to the most recent event at/before it and the next event after it.
# Only information available at trade time is used for "prev"; "next" uses the *schedule*
# (event time/type/impact), never its outcome.
CONTEXT_VIEW = """
CREATE OR REPLACE VIEW trade_macro_context AS
SELECT t.trade_id, t.ts_utc, t.strategy_id, t.pnl, t.params, t.source,
       prev.ts_utc AS prev_event_ts, prev.event_type AS prev_event_type,
       prev.event_name AS prev_event_name, prev.signed_surprise_z AS prev_signed_z,
       nxt.ts_utc AS next_event_ts, nxt.event_type AS next_event_type,
       nxt.event_name AS next_event_name, nxt.impact AS next_impact
FROM trades t
LEFT JOIN LATERAL (
    SELECT e.ts_utc, e.event_type, e.event_name, e.signed_surprise_z
    FROM macro_events e
    WHERE e.ts_utc <= t.ts_utc AND e.actual IS NOT NULL
    ORDER BY e.ts_utc DESC LIMIT 1
) prev ON TRUE
LEFT JOIN LATERAL (
    SELECT e.ts_utc, e.event_type, e.event_name, e.impact
    FROM macro_events e
    WHERE e.ts_utc > t.ts_utc
    ORDER BY e.ts_utc ASC LIMIT 1
) nxt ON TRUE
"""


def get_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(CONTEXT_VIEW))


def stable_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()


def _records(df: pd.DataFrame) -> list[dict]:
    clean = df.astype(object).where(df.notna(), None)
    return clean.to_dict("records")


def upsert(engine: Engine, table: Table, df: pd.DataFrame, pk: str, chunk: int = 1000) -> int:
    if df.empty:
        return 0
    cols = [c.name for c in table.columns if c.name in df.columns]
    recs = _records(df[cols])
    with engine.begin() as conn:
        for i in range(0, len(recs), chunk):
            stmt = pg_insert(table).values(recs[i:i + chunk])
            stmt = stmt.on_conflict_do_update(
                index_elements=[pk],
                set_={c: stmt.excluded[c] for c in cols if c != pk},
            )
            conn.execute(stmt)
    return len(recs)


def to_utc_ns(s: pd.Series) -> pd.Series:
    """Normalise to datetime64[ns, UTC] so merge_asof keys always match."""
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def read_table(engine: Engine, sql: str, ts_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    df = pd.read_sql(text(sql), engine)
    for c in ts_cols:
        df[c] = to_utc_ns(df[c])
    return df


def sources_in(engine: Engine, table: str) -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT DISTINCT source FROM {table}")).fetchall()  # noqa: S608 (internal names)
    return sorted(r[0] for r in rows)
