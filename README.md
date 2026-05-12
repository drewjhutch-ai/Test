# MLB Sports Betting Intelligence System

An evolving, data-driven MLB betting analysis platform that combines statistical modeling,
live odds tracking, weather data, sharp money detection, and machine learning to identify
positive expected-value opportunities across DraftKings and other sportsbooks.

---

## Features

| Category | What It Does |
|---|---|
| **Live Odds** | Pulls DraftKings, FanDuel, BetMGM, Caesars, Pinnacle, and more via The Odds API |
| **Value Bets** | Compares model win probability to implied odds — flags +EV plays with Kelly-sized recommendations |
| **Arbitrage** | Scans every book pair for guaranteed-profit two-way arbs and totals middles |
| **Sharp Money** | Tracks Pinnacle discrepancies, steam moves, reverse line movement, and multi-book consensus |
| **Weather** | Fetches game-time weather for every outdoor stadium — wind direction, temperature, precipitation |
| **Streaks** | Hot/cold detection for both teams and starting pitchers (last 5 starts ERA, last 10 games) |
| **Park Factors** | Built-in run-environment adjustments for all 30 stadiums |
| **ML Model** | XGBoost win-probability and run-total model — auto-retrains as games complete |
| **Dashboard** | Rich terminal UI with color-coded tables for every signal type |
| **ROI Tracking** | Grades every bet after completion, tracks season ROI and closing-line value |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env and add your keys

# 3. Run once (shows today's plays)
python main.py

# 4. Run continuously (auto-refreshes every 5 minutes)
python main.py --loop

# 5. Run with mock data (no API keys needed for testing)
python main.py --demo
```

---

## API Keys

| API | Free Tier | Used For |
|---|---|---|
| [The Odds API](https://the-odds-api.com) | 500 req/month | Live odds from DraftKings + all books |
| [OpenWeatherMap](https://openweathermap.org/api) | 60 calls/min | Game-time weather at every stadium |

The MLB Stats API is completely free — no key required.

---

## CLI Options

```
python main.py                  # Run once and display full dashboard
python main.py --loop           # Continuous mode (default: 5-min refresh)
python main.py --interval 10    # Continuous mode, 10-minute refresh
python main.py --report         # Generate and print daily report
python main.py --retrain        # Force ML model retrain
python main.py --export-csv     # Export value bets to CSV
python main.py --roi            # Show season ROI summary
```

---

## How the Model Evolves

1. Every game played adds a data point to the SQLite database.
2. After `RETRAIN_THRESHOLD` (default: 10) new games complete, the model retrains automatically.
3. A daily hard retrain runs at 6:00 AM.
4. The model tracks Brier score and log-loss to measure calibration improvement over time.
5. All graded bets are stored with actual P&L for closing-line value analysis.

---

## Architecture

```
sports_betting/
├── config.py               — API keys, thresholds, stadium coordinates
├── database.py             — SQLite schema and query helpers
├── engine.py               — Main pipeline orchestrator
├── collectors/
│   ├── mlb_collector.py    — MLB StatsAPI: schedule, standings, pitcher logs
│   ├── odds_collector.py   — The Odds API: live lines from all books
│   ├── weather_collector.py— OpenWeatherMap: per-stadium forecasts
│   └── sharp_money.py      — Steam moves, Pinnacle discrepancy, RLM
├── analysis/
│   ├── streak_analyzer.py  — Team and pitcher hot/cold streak scoring
│   ├── arbitrage_detector.py— Cross-book arb and totals middles
│   ├── value_analyzer.py   — EV calculation and Kelly Criterion sizing
│   └── line_movement.py    — Opening-to-current line delta tracking
├── models/
│   ├── prediction_model.py — XGBoost win-prob + totals model
│   └── trainer.py          — Auto-retrain scheduler + bet grader
└── output/
    ├── dashboard.py        — Rich terminal dashboard
    └── reporter.py         — JSON/CSV report generator
```

---

## Risk Disclaimer

This tool is for **research and educational purposes**. Sports betting involves substantial
financial risk. Past model performance does not guarantee future results. Always bet
responsibly and within legal jurisdictions.
