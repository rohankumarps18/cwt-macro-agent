"""Settings read from environment variables (never hard-coded)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()


def require(value, name: str):
    if not value:
        raise RuntimeError(f"Missing {name}. Set it in your environment or .env (see .env.example).")
    return value


def _csv(name: str, default: str) -> list[str]:
    return [p.strip() for p in os.getenv(name, default).split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    database_url: str
    apify_token: str | None
    apify_actor_id: str | None
    apify_actor_input_raw: str
    tavily_api_key: str | None
    calendar_source_tz: str
    calendar_countries: list[str]
    lookback_days: int
    warmup_days: int
    upcoming_days: int
    trade_log_tz: str
    param_cols: list[str]
    initial_capital: float
    periods_per_year: int

    def apify_actor_input(self, start: date, end: date) -> dict:
        raw = self.apify_actor_input_raw
        raw = raw.replace("{from_date}", start.isoformat()).replace("{to_date}", end.isoformat())
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"APIFY_ACTOR_INPUT is not valid JSON: {exc}") from exc

    def calendar_window(self, today: date | None = None) -> tuple[date, date]:
        """Calendar fetch = lookback + warm-up, so surprise z-scores exist from the start of the lookback."""
        end = today or date.today()
        return (end - timedelta(days=self.lookback_days + self.warmup_days),
                end + timedelta(days=self.upcoming_days))

    def window(self, today: date | None = None) -> tuple[date, date]:
        end = today or date.today()
        return end - timedelta(days=self.lookback_days), end


def get_settings() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql+psycopg2://cwt:cwt@localhost:5432/cwt"),
        apify_token=os.getenv("APIFY_TOKEN"),
        apify_actor_id=os.getenv("APIFY_ACTOR_ID"),
        apify_actor_input_raw=os.getenv("APIFY_ACTOR_INPUT", '{"fromDate": "{from_date}", "toDate": "{to_date}"}'),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        calendar_source_tz=os.getenv("CALENDAR_SOURCE_TZ", "America/New_York"),
        calendar_countries=_csv("CALENDAR_COUNTRIES", "US,United States,USD"),
        lookback_days=int(os.getenv("LOOKBACK_DAYS", "180")),
        warmup_days=int(os.getenv("WARMUP_DAYS", "365")),
        upcoming_days=int(os.getenv("UPCOMING_DAYS", "14")),
        trade_log_tz=os.getenv("TRADE_LOG_TZ", "UTC"),
        param_cols=_csv("PARAM_COLS", "lookback,stop_atr,mode"),
        initial_capital=float(os.getenv("INITIAL_CAPITAL", "100000")),
        periods_per_year=int(os.getenv("PERIODS_PER_YEAR", "252")),
    )
