import pandas as pd
from cwt import db
from cwt.pipeline import _engine
from cwt.db import read_table

s, eng = _engine()
n = read_table(eng, "SELECT * FROM news_articles", ts_cols=("published_at",))
print("columns:", list(n.columns))
print("articles:", len(n), "| with published_at:", int(n["published_at"].notna().sum()))
p = pd.to_datetime(n["published_at"].dropna(), utc=True)
print("range:", p.min(), "->", p.max())
mid = (p.dt.hour == 0) & (p.dt.minute == 0) & (p.dt.second == 0)
print("share stamped exactly 00:00:00 UTC: %.2f" % mid.mean())
if "date_inferred" in n.columns:
    print("share with date_inferred: %.2f" % n["date_inferred"].fillna(False).astype(bool).mean())
print("articles per month:\n", p.dt.to_period("M").value_counts().sort_index().to_string())
print("news sources:", db.sources_in(eng, "news_articles"))
print("macro_event sources:", db.sources_in(eng, "macro_events"))
ev = read_table(eng, "SELECT * FROM macro_events", ts_cols=("ts_utc",))
print("events with actual:", int(ev["actual"].notna().sum()), "| by type:",
      ev.loc[ev["actual"].notna(), "event_type"].value_counts().to_dict())
