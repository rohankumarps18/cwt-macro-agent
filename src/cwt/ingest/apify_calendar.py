"""Fetch the macro calendar through an Apify actor and store it in Postgres."""
from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd
from apify_client import ApifyClient

from cwt.config import Settings, require
from cwt.ingest.calendar_parse import normalize_records

log = logging.getLogger(__name__)


def run_actor(settings: Settings, start: date, end: date) -> list[dict]:
    token = require(settings.apify_token, "APIFY_TOKEN")
    actor_id = require(settings.apify_actor_id, "APIFY_ACTOR_ID")
    client = ApifyClient(token)
    run_input = settings.apify_actor_input(start, end)
    log.info("Running Apify actor %s with input %s", actor_id, run_input)
    run = client.actor(actor_id).call(run_input=run_input)
    if run is None:
        raise RuntimeError("Apify actor run returned no result (it may have timed out).")
    # Client versions return either a dict or a model object.
    status = run["status"] if isinstance(run, dict) else run.status
    dataset_id = run["defaultDatasetId"] if isinstance(run, dict) else run.default_dataset_id
    status = getattr(status, "value", status)
    if str(status).upper() != "SUCCEEDED":
        raise RuntimeError(f"Apify actor finished with status {status}")
    items = list(client.dataset(dataset_id).iterate_items())
    log.info("Apify returned %d raw items", len(items))
    return items


def load_calendar_csv(path: str) -> list[dict]:
    """Load saved calendar records: a CSV, or the raw JSON written by `ingest-calendar --dump`."""
    if path.lower().endswith(".json"):
        with open(path) as fh:
            return json.load(fh)
    return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")


def fetch_calendar(settings: Settings, source: str = "apify", csv_path: str | None = None,
                   raw_dump_path: str | None = None) -> pd.DataFrame:
    if csv_path:
        records = load_calendar_csv(csv_path)
    else:
        start, end = settings.calendar_window()
        records = run_actor(settings, start, end)
        if raw_dump_path:  # keep the untouched actor output for reproducibility / debugging
            with open(raw_dump_path, "w") as fh:
                json.dump(records, fh, default=str)
    df = normalize_records(records, settings.calendar_source_tz, settings.calendar_countries, source)
    if df.empty:
        raise RuntimeError(
            "No CPI/FOMC/employment events could be parsed. Inspect the raw records: the actor's field "
            "names may need to be added to ALIASES in calendar_parse.py.")
    # Guard: some actors silently fall back to a forecast-only feed (no actuals) without a residential proxy.
    past = df[df["ts_utc"] < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)]
    if len(past) and past["actual"].notna().mean() < 0.5:
        raise RuntimeError(
            f"Only {past['actual'].notna().mean():.0%} of past events have an 'actual' value, so surprise features "
            "cannot be built. For xtracto/forexfactory-calendar this usually means no Residential proxy was used "
            "(see proxyConfiguration in APIFY_ACTOR_INPUT).")
    return df
