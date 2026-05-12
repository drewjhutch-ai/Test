"""
Rich terminal dashboard for the sports betting system.
Displays value bets, arbitrage, sharp money, and model predictions.
"""
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box
from rich.layout import Layout
from rich.live import Live
from ..database import (
    get_recent_value_bets, get_recent_arb_opportunities,
    get_recent_sharp_plays
)
from ..collectors.odds_collector import american_to_implied_prob
from ..config import BANKROLL

console = Console()


def render_header():
    now = datetime.now().strftime("%A %B %d, %Y  %I:%M:%S %p")
    console.print(Panel(
        f"[bold cyan]MLB SPORTS BETTING INTELLIGENCE SYSTEM[/bold cyan]\n"
        f"[dim]{now}[/dim]  |  [yellow]Bankroll: ${BANKROLL:,.2f}[/yellow]",
        box=box.DOUBLE_EDGE,
        style="bold",
    ))


def render_value_bets(value_bets: list[dict], predictions: list[dict] = None):
    table = Table(
        title="VALUE BETS  [dim](sorted by edge)[/dim]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Game", style="cyan", min_width=24)
    table.add_column("Book", style="white", min_width=10)
    table.add_column("Side", style="white", min_width=8)
    table.add_column("Market", min_width=8)
    table.add_column("Price", justify="right", min_width=7)
    table.add_column("Our Prob", justify="right", min_width=9)
    table.add_column("Implied", justify="right", min_width=8)
    table.add_column("Edge", justify="right", min_width=7)
    table.add_column("Bet $", justify="right", min_width=7)
    table.add_column("Confidence", min_width=10)

    if not value_bets:
        table.add_row("[dim]No value bets detected in last 24 hours[/dim]", *[""] * 9)
    else:
        for bet in value_bets[:15]:
            edge = bet.get("edge", 0)
            edge_color = "green" if edge >= 0.05 else "yellow" if edge >= 0.03 else "white"
            conf = bet.get("confidence", "LOW")
            conf_color = "green" if conf == "HIGH" else "yellow" if conf == "MEDIUM" else "red"

            game = f"{bet.get('away_team', '?')} @ {bet.get('home_team', '?')}"
            if len(game) > 26:
                parts = game.split(" @ ")
                game = f"{parts[0][:10]}.. @ {parts[1][:10]}.."

            table.add_row(
                game,
                bet.get("book", "?").upper(),
                bet.get("side", "?").upper(),
                bet.get("market", "?"),
                f"[bold]{bet.get('book_price', 0):+.0f}[/bold]",
                f"[{edge_color}]{bet.get('model_probability', 0):.1%}[/{edge_color}]",
                f"{bet.get('implied_probability', 0):.1%}",
                f"[{edge_color}]+{edge:.1%}[/{edge_color}]",
                f"[bold green]${bet.get('recommended_bet', 0):.0f}[/bold green]",
                f"[{conf_color}]{conf}[/{conf_color}]",
            )
    console.print(table)


def render_arbitrage(arb_opportunities: list[dict]):
    table = Table(
        title="ARBITRAGE OPPORTUNITIES  [dim](guaranteed profit)[/dim]",
        box=box.ROUNDED,
        header_style="bold green",
        expand=True,
    )
    table.add_column("Game", style="cyan", min_width=24)
    table.add_column("Market", min_width=10)
    table.add_column("Book A", min_width=12)
    table.add_column("Side A / Price", min_width=16)
    table.add_column("Book B", min_width=12)
    table.add_column("Side B / Price", min_width=16)
    table.add_column("Profit %", justify="right", min_width=9)

    if not arb_opportunities:
        table.add_row("[dim]No arbitrage opportunities currently[/dim]", *[""] * 6)
    else:
        for arb in arb_opportunities[:10]:
            game = f"{arb.get('away_team', '?')} @ {arb.get('home_team', '?')}"
            profit = arb.get("profit_pct", 0)
            profit_color = "bold green" if profit >= 0.5 else "green"
            table.add_row(
                game,
                arb.get("market", "?"),
                arb.get("book_a", "?").upper(),
                f"{arb.get('side_a', '?')} ({arb.get('price_a', 0):+.0f})",
                arb.get("book_b", "?").upper(),
                f"{arb.get('side_b', '?')} ({arb.get('price_b', 0):+.0f})",
                f"[{profit_color}]+{profit:.3f}%[/{profit_color}]",
            )
    console.print(table)


def render_sharp_money(sharp_plays: list[dict]):
    table = Table(
        title="SHARP MONEY SIGNALS  [dim](follow the wiseguys)[/dim]",
        box=box.ROUNDED,
        header_style="bold yellow",
        expand=True,
    )
    table.add_column("Game", style="cyan", min_width=24)
    table.add_column("Signal Type", min_width=22)
    table.add_column("Side", min_width=10)
    table.add_column("Market", min_width=8)
    table.add_column("Strength", justify="right", min_width=10)
    table.add_column("Notes", min_width=40)

    if not sharp_plays:
        table.add_row("[dim]No sharp money signals in last 24 hours[/dim]", *[""] * 5)
    else:
        for play in sharp_plays[:10]:
            strength = play.get("signal_strength", 0)
            strength_bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))
            strength_color = "red" if strength >= 0.7 else "yellow" if strength >= 0.4 else "white"

            game = f"{play.get('away_team', '?')} @ {play.get('home_team', '?')}"
            signal_type = play.get("signal_type", "?")
            type_styles = {
                "STEAM_MOVE": "[bold red]STEAM MOVE[/bold red]",
                "PINNACLE_DISCREPANCY": "[bold yellow]PINNACLE EDGE[/bold yellow]",
                "LINE_MOVEMENT": "[yellow]LINE MOVE[/yellow]",
                "REVERSE_LINE_MOVEMENT": "[bold orange1]RLM[/bold orange1]",
            }
            styled_type = type_styles.get(signal_type, signal_type)

            table.add_row(
                game,
                styled_type,
                (play.get("side") or "?").upper(),
                (play.get("market") or "?"),
                f"[{strength_color}]{strength_bar}[/{strength_color}]",
                play.get("notes", "")[:50],
            )
    console.print(table)


def render_game_predictions(predictions: list[dict]):
    table = Table(
        title="TODAY'S GAME PREDICTIONS",
        box=box.ROUNDED,
        header_style="bold blue",
        expand=True,
    )
    table.add_column("Game", style="cyan", min_width=26)
    table.add_column("Home Win%", justify="right", min_width=11)
    table.add_column("Away Win%", justify="right", min_width=11)
    table.add_column("Pred Total", justify="right", min_width=11)
    table.add_column("Weather", min_width=14)
    table.add_column("Streak Edge", min_width=14)
    table.add_column("Confidence", min_width=10)
    table.add_column("Method", min_width=10)

    if not predictions:
        table.add_row("[dim]No predictions available[/dim]", *[""] * 7)
    else:
        for pred in predictions:
            home_prob = pred.get("home_win_prob", 0.5)
            away_prob = pred.get("away_win_prob", 0.5)
            conf = pred.get("confidence", 0.5)
            conf_color = "green" if conf >= 0.65 else "yellow" if conf >= 0.50 else "red"

            factors = pred.get("factors", {})
            weather_score = factors.get("weather", 0)
            weather_str = (
                f"[green]OVER +{abs(weather_score):.2f}[/green]" if weather_score > 0.1
                else f"[blue]UNDER -{abs(weather_score):.2f}[/blue]" if weather_score < -0.1
                else "[dim]NEUTRAL[/dim]"
            )

            streak_edge = factors.get("streak_edge", 0)
            streak_str = (
                f"[green]HOME +{streak_edge:.3f}[/green]" if streak_edge > 0.05
                else f"[red]AWAY +{abs(streak_edge):.3f}[/red]" if streak_edge < -0.05
                else "[dim]EVEN[/dim]"
            )

            table.add_row(
                pred.get("game", "Unknown"),
                f"[bold]{home_prob:.1%}[/bold]",
                f"{away_prob:.1%}",
                f"{pred.get('predicted_total', 0):.1f}",
                weather_str,
                streak_str,
                f"[{conf_color}]{conf:.0%}[/{conf_color}]",
                pred.get("method", "?"),
            )
    console.print(table)


def render_model_stats():
    from ..database import get_db
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM model_performance
            ORDER BY updated_at DESC LIMIT 1
        """).fetchone()

    if row:
        row = dict(row)
        console.print(Panel(
            f"[bold]Model: {row.get('model_version', '?')}[/bold]  |  "
            f"Games: {row.get('total_predictions', 0)}  |  "
            f"Accuracy: {(row.get('accuracy') or 0):.1%}  |  "
            f"Brier Score: {(row.get('brier_score') or 0):.4f}  |  "
            f"Log Loss: {(row.get('log_loss') or 0):.4f}",
            title="[bold cyan]Model Performance[/bold cyan]",
            box=box.SIMPLE,
        ))


def render_full_dashboard(
    value_bets: list[dict] = None,
    arb_opportunities: list[dict] = None,
    sharp_plays: list[dict] = None,
    predictions: list[dict] = None,
):
    console.clear()
    render_header()
    render_game_predictions(predictions or [])
    console.print()
    render_value_bets(value_bets or get_recent_value_bets())
    console.print()
    render_arbitrage(arb_opportunities or get_recent_arb_opportunities())
    console.print()
    render_sharp_money(sharp_plays or get_recent_sharp_plays())
    console.print()
    render_model_stats()
    console.print(
        "[dim]Data refreshes every 5 minutes. Press Ctrl+C to exit.[/dim]"
    )
