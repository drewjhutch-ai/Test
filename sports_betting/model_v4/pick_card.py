"""
Pick card formatter — Rich terminal output for the full daily card.
Renders the POTD, numbered picks, skipped games, parlays, NRFI section,
and lessons/cheat sheet panel.
"""
from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from .layer_engine import PickCandidate
from .parlay_builder import Parlay, format_parlay_output
from .nrfi_yrfi import NrfiProfile, rank_games_for_nrfi_parlay

console = Console()

UNIT_SIZE_DOLLARS = 5  # $5/unit


def render_pick_card(
    picks: list[PickCandidate],
    parlays: list[Parlay],
    nrfi_ranked: list[dict],
    skipped_games: list[dict],
    date_str: str,
    roi_summary: dict | None = None,
):
    console.clear()
    _render_header(date_str, roi_summary)
    _render_potd(picks)
    _render_picks_table(picks)
    _render_parlays(parlays)
    _render_nrfi_section(nrfi_ranked)
    _render_skipped(skipped_games)
    _render_cheat_sheet()


def _render_header(date_str: str, roi: dict | None):
    roi_line = ""
    if roi and roi.get("total_bets"):
        roi_color = "green" if roi.get("roi", 0) >= 0 else "red"
        roi_line = (
            f"\n[dim]Season: {roi.get('wins',0)}-{roi.get('losses',0)} | "
            f"ROI: [{roi_color}]{roi.get('roi',0):+.2f}%[/{roi_color}] | "
            f"Hit Rate: {roi.get('hit_rate',0):.1%}[/dim]"
        )

    console.print(Panel(
        f"[bold cyan]MLB BETTING MODEL v4.0[/bold cyan]   [dim]|   {date_str}[/dim]\n"
        f"[dim]Platform: DraftKings   Unit: ${UNIT_SIZE_DOLLARS}   Built from 24 days live data[/dim]"
        + roi_line,
        box=box.DOUBLE_EDGE,
    ))


def _render_potd(picks: list[PickCandidate]):
    strong = [p for p in picks if p.tier == "STRONG"]
    if not strong:
        return
    potd = strong[0]
    game_str = f"{potd.away_team} @ {potd.home_team}"
    units = 3
    dollar = units * UNIT_SIZE_DOLLARS
    console.print(Panel(
        f"[bold yellow]⭐  PICK OF THE DAY[/bold yellow]\n\n"
        f"  [bold white]{game_str}[/bold white]\n"
        f"  [cyan]{potd.recommended_market or potd.proposed_market}[/cyan]   "
        f"[bold green]STRONG   {units}u / ${dollar}[/bold green]\n\n"
        f"  [dim]Factors ({potd.factor_count}): {', '.join(potd.factors[:4])}[/dim]\n"
        f"  [dim]Losing scenario ({potd.losing_pct:.0%}): {potd.losing_scenario}[/dim]",
        box=box.HEAVY,
        border_style="yellow",
    ))


def _render_picks_table(picks: list[PickCandidate]):
    active = [p for p in picks if p.tier != "SKIP"]
    if not active:
        console.print("[dim]No picks survived the 12-layer filter today.[/dim]")
        return

    t = Table(
        title=f"TODAY'S CARD ({len(active)} PICKS)",
        box=box.ROUNDED,
        header_style="bold magenta",
        expand=True,
    )
    t.add_column("#", style="dim", width=3)
    t.add_column("Game", min_width=26)
    t.add_column("Market", min_width=18)
    t.add_column("Tier", min_width=8)
    t.add_column("Bet", min_width=10)
    t.add_column("Factors", justify="right", min_width=8)
    t.add_column("Lose%", justify="right", min_width=7)
    t.add_column("Evidence (top 3)", min_width=40)

    tier_colors = {"STRONG": "bold green", "MEDIUM": "yellow", "LEAN": "white"}
    unit_map = {"STRONG": 3, "MEDIUM": 2, "LEAN": 1}

    for i, p in enumerate(active, 1):
        color = tier_colors.get(p.tier, "white")
        units = unit_map.get(p.tier, 1)
        bet_str = f"{units}u / ${units * UNIT_SIZE_DOLLARS}"
        top_evidence = " | ".join(p.factors[:3])
        game_str = f"{p.away_team} @ {p.home_team}"

        t.add_row(
            str(i),
            game_str,
            p.recommended_market or p.proposed_market,
            f"[{color}]{p.tier}[/{color}]",
            f"[{color}]{bet_str}[/{color}]",
            f"{p.factor_count}/{'7' if p.tier=='STRONG' else '5' if p.tier=='MEDIUM' else '3'}",
            f"{p.losing_pct:.0%}",
            top_evidence[:48],
        )

    console.print(t)


def _render_parlays(parlays: list[Parlay]):
    if not parlays:
        console.print(Panel("[dim]No parlays met EV thresholds today.[/dim]",
                            title="PARLAYS", box=box.ROUNDED))
        return

    console.print(Panel(
        format_parlay_output(parlays),
        title="[bold cyan]PARLAYS   (P1 Anchor → P5 Moonshot)[/bold cyan]",
        box=box.ROUNDED,
    ))


def _render_nrfi_section(nrfi_ranked: list[dict]):
    t = Table(
        title="NRFI/YRFI RANKINGS",
        box=box.ROUNDED,
        header_style="bold blue",
        expand=True,
    )
    t.add_column("Game", min_width=26)
    t.add_column("Home SP", min_width=14)
    t.add_column("Away SP", min_width=14)
    t.add_column("NRFI%", justify="right", min_width=8)
    t.add_column("YRFI%", justify="right", min_width=8)
    t.add_column("Lean", min_width=12)
    t.add_column("Parlay?", min_width=10)
    t.add_column("Tier", min_width=24)

    for g in nrfi_ranked:
        nrfi = g.get("nrfi_probability", 0)
        lean = g.get("lean", "?")
        lean_color = "green" if lean == "NRFI" else "red"
        eligible = g.get("parlay_eligible", False)
        eligible_str = "[green]YES[/green]" if eligible else "[dim]no[/dim]"

        t.add_row(
            g.get("game", "?"),
            g.get("home_pitcher", "?"),
            g.get("away_pitcher", "?"),
            f"{nrfi:.1%}",
            f"{g.get('yrfi_probability', 0):.1%}",
            f"[{lean_color}]{lean}[/{lean_color}]",
            eligible_str,
            g.get("tier", "?"),
        )

    console.print(t)


def _render_skipped(skipped: list[dict]):
    if not skipped:
        return
    t = Table(
        title=f"SKIPPED GAMES ({len(skipped)})",
        box=box.SIMPLE,
        header_style="dim",
        expand=True,
    )
    t.add_column("Game", min_width=28)
    t.add_column("Reason", min_width=55)

    for s in skipped:
        t.add_row(s.get("game", "?"), s.get("reason", "?"))

    console.print(t)


def _render_cheat_sheet():
    console.print(Panel(
        "[bold]QUICK REFERENCE[/bold]\n"
        "[green]STRONG (3u/$15)[/green]: 7+ factors | ≤25% lose | +8% EV\n"
        "[yellow]MEDIUM (2u/$10)[/yellow]: 5-6 factors | ≤32% lose | +5% EV\n"
        "[white]LEAN   (1u/$5)[/white] : 3-4 factors | ≤40% lose | +3% EV\n\n"
        "[bold red]PERMANENT PARLAY BANS:[/bold red] "
        "Game Totals | LAA | Sweep G3 | Debut K Props | Same-Series 2-Loss | Same Game (>1 bet)\n\n"
        "[bold cyan]MARKET GUIDE:[/bold cyan] "
        "Elite SP + .520 team → Full-game ML | "
        "Elite SP + sub-.500 → F5 ML | "
        "ERA fraud → Back opponent | "
        "High K/9 → K Over | "
        "Two elite SPs + dome → NRFI",
        title="[dim]MODEL CHEAT SHEET[/dim]",
        box=box.SIMPLE,
        border_style="dim",
    ))
