"""
Text and JSON report generator.
Produces daily summary reports and alert-style outputs
for the best plays of the day.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich import box
from ..config import DATA_DIR, BANKROLL
from ..database import get_recent_value_bets, get_recent_arb_opportunities, get_recent_sharp_plays
from ..models.trainer import compute_roi_summary

logger = logging.getLogger(__name__)
console = Console()

REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def generate_daily_report(
    predictions: list[dict],
    value_bets: list[dict],
    arb_opportunities: list[dict],
    sharp_plays: list[dict],
) -> dict:
    """Build the full daily report as a structured dict."""
    roi = compute_roi_summary()
    today = datetime.now().strftime("%Y-%m-%d")

    report = {
        "generated_at": datetime.now().isoformat(),
        "date": today,
        "bankroll": BANKROLL,
        "roi_summary": roi,
        "games_today": len(predictions),
        "top_value_bets": _rank_value_bets(value_bets),
        "arbitrage_opportunities": arb_opportunities[:5],
        "sharp_money_plays": _rank_sharp_plays(sharp_plays),
        "game_predictions": predictions,
        "alerts": _generate_alerts(value_bets, arb_opportunities, sharp_plays),
    }

    # Save JSON report
    report_path = REPORTS_DIR / f"report_{today}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Daily report saved to %s", report_path)

    return report


def _rank_value_bets(value_bets: list[dict]) -> list[dict]:
    """Return top value bets ranked by edge * confidence."""
    def score(bet):
        conf_mult = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}.get(bet.get("confidence"), 0.5)
        return bet.get("edge", 0) * conf_mult

    sorted_bets = sorted(value_bets, key=score, reverse=True)
    return sorted_bets[:10]


def _rank_sharp_plays(sharp_plays: list[dict]) -> list[dict]:
    """Return top sharp plays by signal strength."""
    return sorted(sharp_plays, key=lambda x: x.get("signal_strength", 0), reverse=True)[:5]


def _generate_alerts(
    value_bets: list[dict],
    arb_opportunities: list[dict],
    sharp_plays: list[dict],
) -> list[str]:
    """Generate human-readable alert strings for the most important plays."""
    alerts = []

    # High-edge value bets
    for bet in value_bets:
        edge = bet.get("edge", 0)
        if edge >= 0.07 and bet.get("confidence") == "HIGH":
            game = f"{bet.get('away_team')} @ {bet.get('home_team')}"
            alerts.append(
                f"STRONG VALUE: {game} — {bet['side'].upper()} {bet['market']} "
                f"@ {bet['book'].upper()} ({bet['book_price']:+.0f}) | "
                f"Edge: +{edge:.1%} | Bet: ${bet['recommended_bet']:.0f}"
            )

    # Any live arbitrage
    for arb in arb_opportunities:
        if arb.get("profit_pct", 0) >= 0.5:
            alerts.append(
                f"ARB ALERT: {arb.get('away_team')} @ {arb.get('home_team')} — "
                f"{arb['book_a'].upper()} + {arb['book_b'].upper()} | "
                f"Profit: +{arb['profit_pct']:.3f}%"
            )

    # High-confidence sharp signals
    for play in sharp_plays:
        if play.get("signal_strength", 0) >= 0.75:
            alerts.append(
                f"SHARP ACTION: {play.get('away_team')} @ {play.get('home_team')} — "
                f"{play['signal_type']} on {(play.get('side') or '?').upper()} "
                f"({play.get('market')}) | Strength: {play['signal_strength']:.0%}"
            )

    return alerts


def print_best_plays(report: dict):
    """Print a formatted best-plays summary to the terminal."""
    alerts = report.get("alerts", [])
    top_bets = report.get("top_value_bets", [])

    if not alerts and not top_bets:
        console.print(Panel("[dim]No high-confidence plays identified today.[/dim]",
                            title="Best Plays", box=box.ROUNDED))
        return

    lines = []
    for i, alert in enumerate(alerts[:8], 1):
        lines.append(f"  [bold cyan]{i}.[/bold cyan] {alert}")

    roi = report.get("roi_summary", {})
    if roi.get("total_bets"):
        lines.append("")
        lines.append(
            f"  [dim]Season ROI: [bold green]{roi.get('roi', 0):+.2f}%[/bold green]  |  "
            f"Record: {roi.get('wins', 0)}-{roi.get('losses', 0)}  |  "
            f"Hit Rate: {roi.get('hit_rate', 0):.1%}[/dim]"
        )

    console.print(Panel(
        "\n".join(lines) if lines else "[dim]No alerts[/dim]",
        title=f"[bold yellow]BEST PLAYS — {report.get('date')}[/bold yellow]",
        box=box.DOUBLE_EDGE,
        expand=True,
    ))


def export_csv(output_path: str = None):
    """Export all value bets to CSV for external analysis."""
    import csv
    bets = get_recent_value_bets(hours=168)  # Last week
    if not bets:
        return

    path = output_path or str(REPORTS_DIR / f"value_bets_{datetime.now().strftime('%Y%m%d')}.csv")
    fields = ["game_id", "home_team", "away_team", "game_date", "book", "market",
              "side", "book_price", "model_probability", "implied_probability",
              "edge", "kelly_fraction", "recommended_bet", "confidence",
              "result", "profit_loss", "detected_at"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(bets)

    logger.info("Exported %d bets to %s", len(bets), path)
    return path
