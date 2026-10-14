import numpy as np
import pandas as pd
import pytest

from cwt.ingest.calendar_parse import (add_surprise_columns, classify, normalize_records, parse_value, sign_for)


@pytest.mark.parametrize("raw,expected", [("3.2%", 3.2), ("225K", 225000.0), ("-0.1%", -0.1), ("1.2M", 1.2e6),
                                          ("$1,234", 1234.0), (4, 4.0)])
def test_parse_value(raw, expected):
    assert parse_value(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "-", "n/a", "abc"])
def test_parse_value_missing(raw):
    assert np.isnan(parse_value(raw))


def test_classify():
    assert classify("Core CPI m/m") == "CPI"
    assert classify("Non-Farm Employment Change") == "EMPLOYMENT"
    assert classify("Fed Interest Rate Decision") == "FOMC"
    assert classify("Retail Sales m/m") is None
    assert classify("FOMC Member Powell Speaks") is None            # speeches are excluded
    for t in ("FOMC Statement", "FOMC Press Conference", "FOMC Meeting Minutes", "Federal Funds Rate"):
        assert classify(t) == "FOMC"
    assert sign_for("Unemployment Rate") == -1 and sign_for("CPI m/m") == 1


def test_timezone_alignment_dst():
    # SAMPLE records (format assumed, not from a real actor). 08:30 New York is 13:30 UTC in winter, 12:30 UTC in summer.
    recs = [
        {"date": "2026-01-15", "time": "08:30", "event": "CPI m/m", "country": "US", "actual": "0.3%", "forecast": "0.2%"},
        {"date": "2026-07-15", "time": "08:30", "event": "CPI m/m", "country": "US", "actual": "0.3%", "forecast": "0.2%"},
    ]
    df = normalize_records(recs, "America/New_York", ["US"], "test").sort_values("ts_utc")
    assert [t.hour for t in df["ts_utc"]] == [13, 12]


def test_tz_aware_input_and_country_filter_and_untimed_rows():
    recs = [
        {"datetime": "2026-03-10T12:30:00+00:00", "event": "CPI m/m", "country": "US", "actual": "0.3", "forecast": "0.2"},
        {"datetime": "2026-03-10T12:30:00+00:00", "event": "CPI m/m", "country": "JP", "actual": "0.3", "forecast": "0.2"},
        {"date": "2026-03-11", "time": "All Day", "event": "CPI m/m", "country": "US"},
    ]
    df = normalize_records(recs, "America/New_York", ["US"], "test")
    assert len(df) == 1 and df.iloc[0]["ts_utc"].hour == 12


def test_surprise_z_is_causal():
    ts = pd.date_range("2026-01-01", periods=6, freq="30D", tz="UTC")
    actual = [1.0, 1.2, 0.7, 1.5, 0.9, 5.0]
    df = pd.DataFrame({"ts_utc": ts, "event_name": "CPI m/m", "actual": actual, "forecast": 1.0})
    out = add_surprise_columns(df)
    s = out["surprise"].to_numpy()
    assert out["surprise_z"].iloc[:3].isna().all()          # needs 3 priors
    assert out["surprise_z"].iloc[3] == pytest.approx(s[3] / np.std(s[:3], ddof=1))
    # changing the LAST outcome must not change any earlier z
    df2 = df.copy(); df2.loc[5, "actual"] = -50.0
    out2 = add_surprise_columns(df2)
    pd.testing.assert_series_equal(out["surprise_z"].iloc[:5], out2["surprise_z"].iloc[:5])


def test_xtracto_forexfactory_row_shape():
    # Field NAMES copied from the actor's documented output (xtracto/forexfactory-calendar); VALUES are illustrative.
    rec = {"id": 1, "title": "Core CPI m/m", "prefixedName": "US Core CPI m/m", "currency": "USD", "country": "US",
           "datetimeISO": "2026-06-10T12:30:00Z", "timeLabel": "8:30am", "impact": "high",
           "actual": "0.4%", "forecast": "0.3%", "previous": "0.2%", "revision": "", "actualBetterWorse": 1}
    unreleased = dict(rec, id=2, datetimeISO="2026-07-14T12:30:00Z", actual=None)
    df = normalize_records([rec, unreleased], "America/New_York", ["US", "USD"], "test").sort_values("ts_utc")
    assert df["event_type"].tolist() == ["CPI", "CPI"] and df["impact"].tolist() == ["high", "high"]
    assert df["ts_utc"].iloc[0] == pd.Timestamp("2026-06-10 12:30", tz="UTC")
    assert df["surprise"].iloc[0] == pytest.approx(0.1) and np.isnan(df["actual"].iloc[1])
