import json

from cwt.ingest.apify_calendar import load_calendar_csv


def test_loads_saved_apify_json(tmp_path):
    rec = [{"title": "CPI m/m", "datetimeISO": "2026-06-10T12:30:00Z", "actual": "0.4%"}]
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(rec))
    assert load_calendar_csv(str(p)) == rec
