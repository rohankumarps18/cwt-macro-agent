from cwt.pipeline import _engine
from cwt.db import read_table

s, eng = _engine()
d = read_table(
    eng,
    "SELECT ts_utc, event_name, actual, forecast, previous, surprise, signed_surprise_z "
    "FROM macro_events WHERE event_name = 'Federal Funds Rate' ORDER BY ts_utc",
    ts_cols=("ts_utc",),
)
print(d.to_string())
print()
print("rows:", len(d), "| actual notna:", int(d["actual"].notna().sum()),
      "| forecast notna:", int(d["forecast"].notna().sum()),
      "| surprise notna:", int(d["surprise"].notna().sum()),
      "| surprise nonzero:", int((d["surprise"].fillna(0) != 0).sum()))
