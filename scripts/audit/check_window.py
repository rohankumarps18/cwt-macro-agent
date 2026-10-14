import pandas as pd
from cwt.pipeline import _engine
from cwt.db import read_table

s, eng = _engine()
ev = read_table(eng, "SELECT * FROM macro_events", ts_cols=("ts_utc",))
ev = ev[ev["actual"].notna()]
w = ev[(ev["ts_utc"] >= "2026-04-17") & (ev["ts_utc"] <= "2026-10-13 23:59:59+00:00")]
print("in-window distinct release timestamps by type:\n", w.groupby("event_type")["ts_utc"].nunique().to_string())
print("total in-window:", w["ts_utc"].nunique(), "| all history:", ev["ts_utc"].nunique())
