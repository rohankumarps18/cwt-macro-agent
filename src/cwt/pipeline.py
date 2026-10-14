"""Command-line entry point:  python -m cwt <command>"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from cwt import db
from cwt.config import get_settings
from cwt.features import build_feature_frame
from cwt.ingest.apify_calendar import fetch_calendar
from cwt.ingest import backtest_log
from cwt.ingest.tavily_news import fetch_news, load_news_csv
from cwt.ingest.trading_log import load_trading_log
from cwt.model import WFConfig, build_policies, evaluate_policies, latest_ranking, walk_forward
from cwt.metrics import block_bootstrap_sharpe, daily_pnl, taken_vs_skipped
from cwt.report import build_report
from cwt.sentiment import daily_sentiment
from cwt.summary import build_summary
from cwt import viz

log = logging.getLogger("cwt")


def _engine():
    s = get_settings()
    eng = db.get_engine(s.database_url)
    db.init_db(eng)
    return s, eng


def cmd_init_db(a):
    _engine()
    print("Schema and join view ready.")


def cmd_reset_db(a):
    if not a.yes:
        raise SystemExit("Refusing to truncate without --yes")
    _, eng = _engine()
    with eng.begin() as c:
        c.execute(text("TRUNCATE macro_events, news_articles, trades"))
    print("Tables truncated.")


def cmd_ingest_calendar(a):
    s, eng = _engine()
    df = fetch_calendar(s, source=a.source, csv_path=a.csv, raw_dump_path=a.dump)
    n = db.upsert(eng, db.macro_events, df, "event_id")
    print(f"Upserted {n} macro events ({df['event_type'].value_counts().to_dict()})")


def cmd_ingest_news(a):
    s, eng = _engine()
    df = load_news_csv(a.csv, a.source) if a.csv else fetch_news(s, source=a.source)
    print(f"Upserted {db.upsert(eng, db.news_articles, df, 'url')} articles")


def cmd_load_trades(a):
    s, eng = _engine()
    df = load_trading_log(a.csv, s, source=a.source)
    print(f"Upserted {db.upsert(eng, db.trades, df, 'trade_id')} trades, "
          f"{df['strategy_id'].nunique()} permutations")


def cmd_build_trade_log(a):
    """Create a CSV trade log by backtesting permutations on free public price data (see backtest_log.py)."""
    s = get_settings()
    start, end = s.window()
    if a.prices_csv:
        bars = backtest_log.load_prices_csv(a.prices_csv)
    else:
        warm = pd.Timestamp(start) - pd.Timedelta(days=10)
        bars = backtest_log.fetch_prices_yfinance(a.symbol, warm.date().isoformat(), end.isoformat(), a.interval)
    log_df = backtest_log.build_trade_log(
        bars, symbol=a.symbol, since=pd.Timestamp(start, tz="UTC"), hold_bars=a.hold_bars,
        notional=a.notional, cost_bps=a.cost_bps, slippage_bps=a.slippage_bps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(a.out, index=False)
    print(f"Wrote {len(log_df)} simulated trades, {log_df.groupby(['lookback', 'stop_atr', 'mode']).ngroups} "
          f"permutations -> {a.out}. Load with: python -m cwt load-trades --csv {a.out} --source backtest")


def cmd_demo_data(a):
    from cwt.demo import make_demo
    paths = make_demo(a.out, signal=not a.no_signal, seed=a.seed, strength=a.strength)
    print("SYNTHETIC files written:", paths)


def cmd_run(a):
    s, eng = _engine()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = WFConfig(train_min_days=a.train_min_days, test_days=a.test_days, step_days=a.step_days,
                   embargo_days=a.embargo_days, target=a.target, top_k=a.top_k, min_train_rows=a.min_train_rows)
    sources = {t: db.sources_in(eng, t) for t in ("trades", "macro_events", "news_articles")}
    synthetic = any("synthetic" in v for v in sources.values())
    backtest = "backtest" in sources["trades"]

    df, feats, param_cols = build_feature_frame(eng, s)
    log.info("feature frame: %d trades, %d features", len(df), len(feats))
    preds, folds, gain = walk_forward(df, feats, cfg)
    policies = build_policies(df, preds, cfg)
    metrics = evaluate_policies(policies, preds, s.initial_capital, s.periods_per_year)
    ranking = latest_ranking(df, feats, cfg, [f"param_{c.lower()}" for c in s.param_cols])

    events = db.read_table(eng, "SELECT * FROM macro_events", ts_cols=("ts_utc",))
    news = db.read_table(eng, "SELECT * FROM news_articles", ts_cols=("published_at",))
    win_start = df["ts_utc"].min().normalize()  # warm-up history seeds the state but is not displayed
    daily = daily_sentiment(events, news, start=win_start)
    events_win = events[events["ts_utc"] >= win_start]

    all_days = pd.DatetimeIndex(sorted(preds["ts_utc"].dt.tz_convert("America/New_York")
                                       .dt.tz_localize(None).dt.normalize().unique()))
    uncertainty = {
        "sharpe_ci": {k: block_bootstrap_sharpe(daily_pnl(t, all_days), s.initial_capital, s.periods_per_year)
                      for k, t in policies.items()},
        "taken_vs_skipped": taken_vs_skipped(preds),
    }
    charts = []
    if not daily.empty:
        charts.append(viz.sentiment_timeline(daily, events_win, out, synthetic))
    charts.append(viz.surprise_matrix(events_win, out, synthetic))
    charts.append(viz.equity_curves(policies, all_days, s.initial_capital, out, synthetic))
    charts.append(viz.regime_pnl_matrix(df, out, synthetic))
    charts.append(viz.feature_importance(gain, out, synthetic))

    preds.to_csv(out / "oos_predictions.csv", index=False)
    metrics.to_csv(out / "strategy_metrics.csv")
    folds.to_csv(out / "fold_metrics.csv", index=False)
    ranking.to_csv(out / "latest_permutation_ranking.csv", index=False)
    (out / "macro_outlook.md").write_text(build_summary(events_win, news, synthetic))
    (out / "evaluation_report.md").write_text(build_report(
        metrics, folds, gain, cfg, len(df), (df["ts_utc"].min(), df["ts_utc"].max()),
        s.initial_capital, s.periods_per_year, sources, ranking, synthetic, backtest, uncertainty))
    print(metrics[["n_trades", "total_pnl", "sharpe", "sortino", "max_drawdown_pct"]].round(3).to_string())
    print(f"Outputs written to {out}/ ({'SYNTHETIC' if synthetic else 'real data'})")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="cwt")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db").set_defaults(fn=cmd_init_db)
    r = sub.add_parser("reset-db"); r.add_argument("--yes", action="store_true"); r.set_defaults(fn=cmd_reset_db)
    c = sub.add_parser("ingest-calendar")
    c.add_argument("--csv"); c.add_argument("--source", default="apify"); c.add_argument("--dump")
    c.set_defaults(fn=cmd_ingest_calendar)
    n = sub.add_parser("ingest-news"); n.add_argument("--csv"); n.add_argument("--source", default="tavily")
    n.set_defaults(fn=cmd_ingest_news)
    t = sub.add_parser("load-trades"); t.add_argument("--csv", required=True); t.add_argument("--source", default="csv")
    t.set_defaults(fn=cmd_load_trades)
    b = sub.add_parser("build-trade-log")
    b.add_argument("--symbol", default="SPY"); b.add_argument("--prices-csv"); b.add_argument("--out", default="data/backtest_trades.csv")
    b.add_argument("--interval", default="1h"); b.add_argument("--hold-bars", type=int, default=4)
    b.add_argument("--notional", type=float, default=10_000.0)
    b.add_argument("--cost-bps", type=float, default=1.0); b.add_argument("--slippage-bps", type=float, default=1.0)
    b.set_defaults(fn=cmd_build_trade_log)
    d = sub.add_parser("demo-data"); d.add_argument("--out", default="data/demo")
    d.add_argument("--no-signal", action="store_true"); d.add_argument("--strength", type=float, default=1.0); d.add_argument("--seed", type=int, default=7)
    d.set_defaults(fn=cmd_demo_data)
    g = sub.add_parser("run"); g.add_argument("--out", default="outputs")
    g.add_argument("--target", choices=["pnl", "win"], default="pnl")
    g.add_argument("--train-min-days", type=int, default=60); g.add_argument("--test-days", type=int, default=7)
    g.add_argument("--step-days", type=int, default=7); g.add_argument("--embargo-days", type=int, default=1)
    g.add_argument("--top-k", type=int, default=3); g.add_argument("--min-train-rows", type=int, default=300)
    g.set_defaults(fn=cmd_run)
    args = p.parse_args(argv)
    args.fn(args)
