#!/usr/bin/env python3
"""
MLB Sports Betting Intelligence System — Main Entry Point

Usage:
    python main.py                  # Run model v4.0 (default — 12-layer framework)
    python main.py --date 2026-05-12 # Run v4 for a specific date
    python main.py --loop           # Run v4 continuously (refreshes every 5 min)
    python main.py --report         # Generate daily report only
    python main.py --retrain        # Force ML model retraining
    python main.py --export-csv     # Export value bets to CSV
    python main.py --roi            # Show season ROI summary

Setup:
    1. Copy .env.example to .env
    2. Add your API keys (The Odds API, OpenWeatherMap)
    3. pip install -r requirements.txt
    4. python main.py
"""
import sys
import time
import logging
import argparse
from datetime import datetime

import schedule
from rich.console import Console

# Ensure the project root is on the path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from sports_betting.engine import BettingEngine
from sports_betting.output.dashboard import render_full_dashboard, console as dash_console
from sports_betting.output.reporter import generate_daily_report, print_best_plays, export_csv
from sports_betting.models.trainer import maybe_retrain, compute_roi_summary
from sports_betting.database import get_recent_value_bets, get_recent_arb_opportunities, get_recent_sharp_plays
from sports_betting.config import LOG_LEVEL
from sports_betting.model_v4.daily_runner import run_daily_model

console = Console()


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("sports_betting/data/app.log", mode="a"),
        ],
    )
    # Quiet noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def run_pipeline(engine: BettingEngine) -> dict:
    """Run the full analysis pipeline and refresh the dashboard."""
    console.print(f"[dim]{datetime.now().strftime('%H:%M:%S')} — Refreshing data...[/dim]")

    try:
        results = engine.run()
    except Exception as e:
        console.print(f"[red]Pipeline error: {e}[/red]")
        logging.exception("Pipeline run failed")
        # Still show cached data from DB on error
        results = {
            "games": [],
            "predictions": [],
            "value_bets": [],
            "arb": [],
            "sharp": [],
        }

    # Build dashboard data — fall back to DB cache if pipeline had gaps
    predictions = results.get("predictions") or []
    value_bets = results.get("value_bets") or get_recent_value_bets(hours=24)
    arb_opps = results.get("arb") or get_recent_arb_opportunities(hours=6)
    sharp_plays = results.get("sharp") or get_recent_sharp_plays(hours=24)

    render_full_dashboard(
        value_bets=value_bets,
        arb_opportunities=arb_opps,
        sharp_plays=sharp_plays,
        predictions=predictions,
    )

    # Print best plays panel
    report = generate_daily_report(predictions, value_bets, arb_opps, sharp_plays)
    print_best_plays(report)

    return results


def run_loop(engine: BettingEngine, interval_minutes: int = 5):
    """Run pipeline on a repeating schedule."""
    console.print(f"[bold cyan]Starting continuous mode — refreshing every {interval_minutes} minutes.[/bold cyan]")
    console.print("[dim]Press Ctrl+C to exit.[/dim]\n")

    # Run immediately on start
    run_pipeline(engine)

    # Schedule repeating runs
    schedule.every(interval_minutes).minutes.do(run_pipeline, engine=engine)

    # Schedule daily model retrain at 6am
    schedule.every().day.at("06:00").do(
        lambda: maybe_retrain(engine.predictor, force=True)
    )

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting — goodbye.[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="MLB Sports Betting Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously, refreshing every 5 minutes")
    parser.add_argument("--interval", type=int, default=5,
                        help="Refresh interval in minutes when using --loop (default: 5)")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to run model for (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--report", action="store_true",
                        help="Print today's report and exit")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retraining and exit")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export value bets to CSV and exit")
    parser.add_argument("--roi", action="store_true",
                        help="Show ROI summary and exit")
    parser.add_argument("--log-level", default=LOG_LEVEL,
                        help="Logging level (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Sports Betting Intelligence System starting up")

    # ---- Quick-exit modes ------------------------------------------ #

    if args.roi:
        roi = compute_roi_summary()
        if roi.get("status") == "no_graded_bets":
            console.print("[yellow]No graded bets yet. Bets are graded after games complete.[/yellow]")
        else:
            console.print(f"""
[bold cyan]ROI Summary[/bold cyan]
  Total Bets  : {roi['total_bets']}
  Record      : {roi['wins']}-{roi['losses']}
  Hit Rate    : {roi['hit_rate']:.1%}
  Total Staked: ${roi['total_staked']:,.2f}
  Total Profit: ${roi['total_profit']:+,.2f}
  ROI         : [bold {'green' if roi['roi'] >= 0 else 'red'}]{roi['roi']:+.2f}%[/bold {'green' if roi['roi'] >= 0 else 'red'}]
  Avg Edge    : {roi['avg_edge']:.1%}
""")
        return

    if args.export_csv:
        path = export_csv()
        console.print(f"[green]Exported to {path}[/green]")
        return

    # ---- Engine-dependent modes ------------------------------------ #

    engine = BettingEngine()

    if args.retrain:
        console.print("[cyan]Forcing model retrain...[/cyan]")
        metrics = maybe_retrain(engine.predictor, force=True)
        if metrics:
            console.print(f"[green]Retrain complete: {metrics}[/green]")
        else:
            console.print("[yellow]Retrain skipped (insufficient data).[/yellow]")
        return

    if args.report:
        results = run_pipeline(engine)
        return

    if args.loop:
        # v4 loop mode
        console.print(f"[bold cyan]Model v4.0 — continuous mode ({args.interval}-min refresh)[/bold cyan]")
        run_daily_model(args.date)
        schedule.every(args.interval).minutes.do(run_daily_model, date_str=args.date)
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting.[/yellow]")
        return

    # Default: model v4.0 single run
    run_daily_model(args.date)


if __name__ == "__main__":
    main()
