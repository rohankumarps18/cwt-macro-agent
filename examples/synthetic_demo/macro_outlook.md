# Macro Outlook (1-page summary)

> **SYNTHETIC DEMO DATA. Not a real market view.**

*Data window:* 2026-05-01 to 2026-10-02 (32 released events, 359 news articles). Auto-generated from the database.

## Latest prints
| Event | Date (UTC) | Actual | Forecast | Surprise | Signed z |
|---|---|---|---|---|---|
| Non-Farm Employment Change | 2026-10-02 12:30 | 78707.75 | 180000.00 | -101292.25 | -2.16 |
| Unemployment Rate | 2026-10-02 12:30 | 4.00 | 4.00 | 0.00 | 0.00 |
| Average Hourly Earnings m/m | 2026-10-02 12:30 | 0.27 | 0.30 | -0.03 | -0.34 |
| Fed Interest Rate Decision | 2026-09-21 18:00 | 5.25 | 5.25 | 0.00 | 0.00 |
| Core CPI m/m | 2026-09-14 12:30 | 0.37 | 0.30 | 0.07 | 0.87 |
| CPI m/m | 2026-09-14 12:30 | 0.30 | 0.30 | -0.00 | -0.02 |

## Sentiment regime
- Macro surprise index: **-0.40** (softer than forecast); 30-day change -0.31 (falling).
- News sentiment (7d VADER): **-0.32** vs 30 days ago 0.24.
- Composite z-score: -1.02.

## Upcoming events in the data
- 2026-10-12 12:30 UTC: CPI m/m (forecast 0.30)
- 2026-10-12 12:30 UTC: Core CPI m/m (forecast 0.30)
- 2026-11-02 19:00 UTC: Fed Interest Rate Decision (forecast 5.25)
- 2026-11-06 13:30 UTC: Non-Farm Employment Change (forecast 180000.00)
- 2026-11-06 13:30 UTC: Unemployment Rate (forecast 4.00)

## Recent headlines (Tavily)
- 2026-10-10: Stocks slide on hawkish Fed fears (-0.68)
- 2026-10-09: Markets rally as data beats expectations (0.00)
- 2026-10-09: Weak outlook weighs on sentiment (-0.70)
- 2026-10-08: Solid growth reassures investors (0.89)
- 2026-10-08: Stocks slide on hawkish Fed fears (-0.68)

## Caveats
- Sentiment is a simple signed-surprise z-score plus a lexicon score on news text; it is a descriptive indicator, not a forecast.
- Early prints have no z-score (min. 3 prior observations per event), so they count as 0.
