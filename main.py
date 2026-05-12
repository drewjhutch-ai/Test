#!/usr/bin/env python3
"""
MLB Sports Betting Intelligence System — Main Entry Point

Usage:
    python main.py                       # Run model v4.0 for today
    python main.py --date 2026-05-13     # Run for a specific date
    python main.py --schedule            # Run at 10am daily + text you picks (leave running)
    python main.py --loop                # Refresh every 5 min continuously
    python main.py --text-now            # Send picks to your phone right now
    python main.py --record-bet          # Record a bet you placed (for feedback loop)
    python main.py --signals             # Show which signals are performing best
    python main.py --roi                 # Show season ROI summary
    python main.py --retrain             # Force ML model retraining
    python main.py --export-csv          # Export value bets to CSV

Setup:
    1. Copy .env.example to .env and fill in API keys
    2. pip install -r requirements.txt
    3. python main.py
"""
import os
import sys
import time
import logging
import argparse
from datetime import datetime

import schedule
from rich.console import Console
from rich.table import Table
from rich import box
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from sports_betting.models.trainer import maybe_retrain, compute_roi_summary
from sports_betting.output.reporter import export_csv
from sports_betting.output.sms_sender import send_daily_picks
from sports_betting.model_v4.daily_runner import run_daily_model
from sports_betting.analysis.signal_tracker import (
    get_signal_performance_report, record_placed_bet
)
from sports_betting.config import LOG_LEVEL
from sports_betting.database import init_db

console = Console()

DAILY_TEXT_TIME = os.getenv("DAILY_TEXT_TIME", "10:00")


def setup_logging(level: str = "INFO"):
    log_dir = __import__("pathlib").Path("sports_betting/data")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("sports_betting/data/app.log", mode="a"),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pybaseball").setLevel(logging.WARNING)


def run_and_text(date_str: str | None = None):
    """Run the full model then text the picks."""
    console.print(f"[dim]{datetime.now().strftime('%H:%M:%S')} — Running model + sending text...[/dim]")
    results = run_daily_model(date_str=date_str, verbose=True)
    sent = send_daily_picks(results)
    if sent:
        console.print("[bold green]✓ Picks texted successfully.[/bold green]")
    else:
        console.print(
            "[yellow]Text not sent — check TWILIO credentials in .env[/yellow]\n"
            "[dim]See setup instructions: python main.py --help[/dim]"
        )
    return results


def show_roi():
    roi = compute_roi_summary()
    if roi.get("status") == "no_graded_bets":
        console.print("[yellow]No graded bets yet. Bets are graded after games complete.[/yellow]")
        return
    color = "green" if roi.get("roi", 0) >= 0 else "red"
    console.print(f"""
[bold cyan]Season ROI Summary[/bold cyan]
  Total Bets  : {roi['total_bets']}
  Record      : {roi['wins']}-{roi['losses']}
  Hit Rate    : {roi.get('hit_rate', 0):.1%}
  Total Staked: ${roi.get('total_staked', 0):,.2f}
  Total Profit: [bold {color}]${roi.get('total_profit', 0):+,.2f}[/bold {color}]
  ROI         : [bold {color}]{roi.get('roi', 0):+.2f}%[/bold {color}]
  Avg Edge    : {roi.get('avg_edge', 0):.1%}
""")


def show_signals():
    report = get_signal_performance_report()
    if not report:
        console.print("[yellow]No signal data yet — needs graded bets to calculate.[/yellow]")
        return

    t = Table(title="Signal Performance (Feedback Loop)", box=box.ROUNDED,
              header_style="bold cyan")
    t.add_column("Signal",      min_width=28)
    t.add_column("Bets",        justify="right", min_width=6)
    t.add_column("Wins",        justify="right", min_width=6)
    t.add_column("Hit Rate",    justify="right", min_width=10)
    t.add_column("Weight",      justify="right", min_width=8)
    t.add_column("Trend",       min_width=10)

    for s in report:
        hr = s["hit_rate"]
        color = "green" if hr >= 0.55 else "red" if hr <= 0.45 else "white"
        trend = "↑ Boosted" if s["current_weight"] > 1.05 else \
                "↓ Reduced" if s["current_weight"] < 0.95 else "→ Neutral"
        t.add_row(
            s["signal"],
            str(s["bets"]),
            str(s["wins"]),
            f"[{color}]{hr:.1%}[/{color}]",
            f"{s['current_weight']:.3f}",
            trend,
        )
    console.print(t)


def prompt_record_bet():
    """Interactive prompt to record a bet you placed."""
    console.print("\n[bold cyan]Record a Placed Bet[/bold cyan]")
    console.print("[dim]This tells the model what you actually bet so it can learn from results.[/dim]\n")

    game_id   = console.input("Game ID (e.g. mlb_12345, or type the matchup like NYY-BOS): ").strip()
    book      = console.input("Book (e.g. draftkings): ").strip() or "draftkings"
    market    = console.input("Market (ml / f5_ml / k_over / er_under / nrfi): ").strip() or "ml"
    side      = console.input("Side (home / away / over / under): ").strip()
    price_str = console.input("Price (American odds, e.g. -135 or +115): ").strip()
    units_str = console.input("Units placed (1 / 2 / 3): ").strip()
    signals   = console.input("Top signals (comma-separated, e.g. 'era fraud, hot streak'): ").strip()

    try:
        price = int(price_str)
        units = float(units_str)
    except ValueError:
        console.print("[red]Invalid price or units — must be numbers.[/red]")
        return

    signal_list = [s.strip() for s in signals.split(",")] if signals else []

    record_placed_bet(
        game_id=game_id,
        book=book,
        market=market,
        side=side,
        price=price,
        units=units,
        signals_present=signal_list,
    )
    console.print(f"\n[green]✓ Bet recorded: {side.upper()} {market} @ {price:+d} ({units}u)[/green]")
    console.print("[dim]Result will be graded automatically after the game completes.[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="MLB Sports Betting Intelligence System v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date",       type=str,  default=None,
                        help="Date to run model for (YYYY-MM-DD)")
    parser.add_argument("--schedule",   action="store_true",
                        help=f"Run daily at {DAILY_TEXT_TIME} and text picks (leave running)")
    parser.add_argument("--loop",       action="store_true",
                        help="Refresh every N minutes continuously")
    parser.add_argument("--interval",   type=int,  default=5,
                        help="Refresh interval in minutes for --loop (default: 5)")
    parser.add_argument("--text-now",   action="store_true",
                        help="Run model now and text picks immediately")
    parser.add_argument("--record-bet", action="store_true",
                        help="Record a bet you placed (feeds the learning loop)")
    parser.add_argument("--signals",    action="store_true",
                        help="Show signal performance report (feedback loop)")
    parser.add_argument("--roi",        action="store_true",
                        help="Show season ROI summary")
    parser.add_argument("--retrain",    action="store_true",
                        help="Force ML model retraining")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export value bets to CSV")
    parser.add_argument("--log-level",  default=LOG_LEVEL,
                        help="Logging level (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    init_db()
    logger = logging.getLogger(__name__)
    logger.info("MLB Betting Model v4.0 starting")

    # ------------------------------------------------------------------ #
    #  Quick-exit modes                                                    #
    # ------------------------------------------------------------------ #

    if args.roi:
        show_roi()
        return

    if args.signals:
        show_signals()
        return

    if args.export_csv:
        path = export_csv()
        console.print(f"[green]Exported to {path}[/green]")
        return

    if args.record_bet:
        prompt_record_bet()
        return

    if args.retrain:
        console.print("[cyan]Forcing model retrain...[/cyan]")
        from sports_betting.models.prediction_model import MLBPredictor
        predictor = MLBPredictor()
        metrics = maybe_retrain(predictor, force=True)
        if metrics:
            console.print(f"[green]Retrain complete: {metrics}[/green]")
        else:
            console.print("[yellow]Retrain skipped (need 50+ completed games).[/yellow]")
        return

    # ------------------------------------------------------------------ #
    #  Text now                                                            #
    # ------------------------------------------------------------------ #

    if args.text_now:
        run_and_text(args.date)
        return

    # ------------------------------------------------------------------ #
    #  Scheduled mode — runs daily at DAILY_TEXT_TIME and texts picks     #
    # ------------------------------------------------------------------ #

    if args.schedule:
        console.print(
            f"[bold cyan]Scheduled mode ON[/bold cyan] — "
            f"model runs every day at [bold]{DAILY_TEXT_TIME}[/bold] and texts your picks.\n"
            f"[dim]Leave this window open. Press Ctrl+C to stop.[/dim]\n"
        )

        # Run once immediately so you see it working
        run_and_text(args.date)

        # Daily pick text
        schedule.every().day.at(DAILY_TEXT_TIME).do(run_and_text, date_str=None)

        # Pre-game refresh 30 min before typical first pitch (7pm ET)
        schedule.every().day.at("18:30").do(run_and_text, date_str=None)

        # Daily retrain at 6am
        schedule.every().day.at("06:00").do(_silent_retrain)

        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            console.print("\n[yellow]Scheduler stopped.[/yellow]")
        return

    # ------------------------------------------------------------------ #
    #  Continuous loop mode                                                #
    # ------------------------------------------------------------------ #

    if args.loop:
        console.print(
            f"[bold cyan]Continuous mode — refreshing every {args.interval} minutes[/bold cyan]\n"
            f"[dim]Press Ctrl+C to stop.[/dim]\n"
        )
        run_daily_model(args.date)
        schedule.every(args.interval).minutes.do(run_daily_model, date_str=args.date)
        schedule.every().day.at("06:00").do(_silent_retrain)
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting.[/yellow]")
        return

    # ------------------------------------------------------------------ #
    #  Default — single run                                                #
    # ------------------------------------------------------------------ #

    run_daily_model(args.date)


def _silent_retrain():
    try:
        from sports_betting.models.prediction_model import MLBPredictor
        maybe_retrain(MLBPredictor(), force=True)
    except Exception as e:
        logging.getLogger(__name__).error("Silent retrain failed: %s", e)


if __name__ == "__main__":
    main()
