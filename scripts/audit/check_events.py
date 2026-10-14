import pandas as pd
from cwt.pipeline import _engine
from cwt.db import read_table

s, eng = _engine()
ev = read_table(eng, "SELECT * FROM macro_events", ts_cols=("ts_utc",))
ev = ev[ev["actual"].notna()].copy()
ev["et_time"] = ev["ts_utc"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
print("distinct release timestamps by type:\n", ev.groupby("event_type")["ts_utc"].nunique().to_string())
print("\nrows per event_name (and how many have a z-score):")
print(ev.groupby("event_name").agg(n=("ts_utc", "size"),
      z=("signed_surprise_z", lambda x: int(x.notna().sum()))).sort_values("n", ascending=False).to_string())
print("\nmost common ET release times:\n", ev["et_time"].value_counts().head(8).to_string())

n = read_table(eng, "SELECT * FROM news_articles", ts_cols=("published_at",))
p = pd.to_datetime(n["published_at"], utc=True)
inferred = n["date_inferred"].astype(str).str.lower().eq("yes")
mid = (p.dt.hour == 0) & (p.dt.minute == 0) & (p.dt.second == 0)
print("\nnews: inferred (undated) = %.2f | midnight-stamped = %.2f | both = %.2f" % (
    inferred.mean(), mid.mean(), (inferred & mid).mean()))
