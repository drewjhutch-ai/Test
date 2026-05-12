"""
Hard Rules Engine — fires BEFORE any analysis runs.
These are absolute blockers with no overrides.
Derived from 24 days of live data losses and pattern analysis.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from ..database import get_db


@dataclass
class RuleViolation:
    rule: str
    reason: str
    action: str
    evidence: str = ""


@dataclass
class HardRulesResult:
    game_id: str
    home_team: str
    away_team: str
    parlay_eligible: bool = True
    standalone_eligible: bool = True
    violations: list[RuleViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    redirects: list[str] = field(default_factory=list)

    def block_parlay(self, rule: str, reason: str, evidence: str = ""):
        self.parlay_eligible = False
        self.violations.append(RuleViolation(
            rule=rule, reason=reason, action="BLOCK from parlay", evidence=evidence
        ))

    def block_all(self, rule: str, reason: str, evidence: str = ""):
        self.parlay_eligible = False
        self.standalone_eligible = False
        self.violations.append(RuleViolation(
            rule=rule, reason=reason, action="BLOCK entirely", evidence=evidence
        ))

    def redirect(self, message: str):
        self.redirects.append(message)

    def warn(self, message: str):
        self.warnings.append(message)


def run_hard_rules(
    game_id: str,
    home_team: str,
    away_team: str,
    backing_team: str,
    market: str,
    series_game_number: int = 1,
    team_series_wins: int = 0,
    rain_pct: float = 0.0,
    pitcher_is_debut: bool = False,
    pitcher_is_laa: bool = False,
    backing_team_win_pct: float = 0.500,
) -> HardRulesResult:
    """
    Run all hard rules against a single pick candidate.
    Returns a HardRulesResult with all violations and redirects.
    """
    result = HardRulesResult(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
    )

    # ------------------------------------------------------------------ #
    # RULE 1: No game totals in parlays
    # ------------------------------------------------------------------ #
    if market in ("totals", "over", "under", "game_total"):
        result.block_parlay(
            rule="no_game_totals_in_parlays",
            reason="Game total picks are NEVER parlay legs (24-day data confirmed)",
            evidence="Too much variance. Allow as standalone only.",
        )
        result.redirect("Move this total to standalone card.")

    # ------------------------------------------------------------------ #
    # RULE 2: LAA permanent parlay ban
    # ------------------------------------------------------------------ #
    if "angels" in home_team.lower() or "angels" in away_team.lower() or \
       home_team == "Los Angeles Angels" or away_team == "Los Angeles Angels" or \
       "LAA" in home_team or "LAA" in away_team:
        result.block_parlay(
            rule="laa_permanent_parlay_ban",
            reason="LAA offense (27th MLB) destroys any pitcher edge",
            evidence="Soriano 0.84 ERA, team still loses 6-0. Permanent ban.",
        )
        result.redirect("LAA: standalone lean maximum.")

    # ------------------------------------------------------------------ #
    # RULE 3: Sweep attempt G3 parlay ban
    # ------------------------------------------------------------------ #
    if series_game_number == 3 and team_series_wins == 2:
        result.block_parlay(
            rule="sweep_attempt_parlay_ban",
            reason="Sweep attempt G3 fails 33-38% of time",
            evidence="ATH sweep ban busted parlays May 10, PIT May 9",
        )
        result.redirect("G3 sweep attempt: standalone 1u lean only.")

    # ------------------------------------------------------------------ #
    # RULE 4: Same-series recycling ban (lost 2+ consecutive in series)
    # ------------------------------------------------------------------ #
    # Caller must set team_series_wins == -2 to signal 2 consecutive losses
    if team_series_wins <= -2:
        result.block_parlay(
            rule="same_series_recycling_ban",
            reason="Never back a team that lost 2+ consecutive games in same series",
            evidence="Opponent has momentum. Series psychology is real.",
        )
        result.block_all(
            rule="same_series_recycling_ban",
            reason="Backing the losing team in a series is blocked.",
        )

    # ------------------------------------------------------------------ #
    # RULE 5: Debut pitcher parlay ban
    # ------------------------------------------------------------------ #
    if pitcher_is_debut:
        result.block_parlay(
            rule="no_debut_start_in_parlay",
            reason="Debut/first-MLB-start pitcher — K props unreliable (Lodolo lesson)",
            evidence="Never parlay. Standalone lean only.",
        )

    # ------------------------------------------------------------------ #
    # RULE 6: Losing-record team full-game ML redirect
    # ------------------------------------------------------------------ #
    if backing_team_win_pct < 0.500 and market == "full_game_ml":
        result.warn(
            f"Backing team win% {backing_team_win_pct:.3f} < .500 — "
            "full-game ML not recommended."
        )
        result.redirect(
            "Team below .500: redirect to F5 ML (isolates pitcher) or K prop."
        )

    # ------------------------------------------------------------------ #
    # RULE 7: Rain exclusion thresholds
    # ------------------------------------------------------------------ #
    if rain_pct >= 50:
        result.block_all(
            rule="rain_exclusion",
            reason=f"Rain probability {rain_pct:.0f}% — do not bet this game",
            evidence="Above 50% rain = postponement risk too high.",
        )
    elif rain_pct >= 30:
        result.block_parlay(
            rule="rain_parlay_exclusion",
            reason=f"Rain probability {rain_pct:.0f}% — exclude from parlays",
            evidence="30-50% rain: standalone only, monitor.",
        )

    return result


def check_seven_day_cap(team_name: str) -> dict:
    """
    Check how many direct ML picks have been placed on this team
    in the rolling 7-day window. Max = 3.
    Counts: full-game ML, F5 ML, run line.
    Does NOT count: K props, ER props, totals, NRFIs.
    """
    cutoff = (datetime.now() - timedelta(days=7)).isoformat()
    with get_db() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as cnt FROM value_bets vb
            JOIN games g ON vb.game_id = g.game_id
            WHERE (g.home_team = ? OR g.away_team = ?)
            AND vb.market IN ('h2h', 'f5_ml', 'run_line')
            AND vb.side = CASE
                WHEN g.home_team = ? THEN 'home'
                WHEN g.away_team = ? THEN 'away'
            END
            AND vb.detected_at >= ?
        """, (team_name, team_name, team_name, team_name, cutoff)).fetchone()

    count = row["cnt"] if row else 0
    cap_remaining = max(0, 3 - count)
    is_capped = count >= 3

    return {
        "team": team_name,
        "picks_in_7_days": count,
        "cap_remaining": cap_remaining,
        "is_capped": is_capped,
        "action": "BLOCK — 7-day cap reached" if is_capped else f"OK — {cap_remaining} picks remaining",
    }


def run_all_hard_rules_for_card(picks: list[dict]) -> list[dict]:
    """
    Run hard rules across a full proposed pick card.
    Also enforces max-1-bet-per-game-in-parlay.
    Returns picks with hard_rules_result attached.
    """
    games_in_parlay: set[str] = set()

    for pick in picks:
        result = run_hard_rules(
            game_id=pick.get("game_id", ""),
            home_team=pick.get("home_team", ""),
            away_team=pick.get("away_team", ""),
            backing_team=pick.get("backing_team", ""),
            market=pick.get("market", ""),
            series_game_number=pick.get("series_game_number", 1),
            team_series_wins=pick.get("team_series_wins", 0),
            rain_pct=pick.get("rain_pct", 0.0),
            pitcher_is_debut=pick.get("pitcher_is_debut", False),
            backing_team_win_pct=pick.get("backing_team_win_pct", 0.500),
        )

        # Max 1 bet per game in parlay (unless positive correlation)
        game_key = pick.get("game_id", "")
        positive_corr = pick.get("positive_correlation", False)
        if game_key in games_in_parlay and not positive_corr:
            result.block_parlay(
                rule="max_one_bet_per_game_in_parlay",
                reason="Already have a pick from this game in the parlay",
                evidence="Max 1 bet per game. Positive correlation pairs are the only exception.",
            )
        elif result.parlay_eligible:
            games_in_parlay.add(game_key)

        # 7-day cap
        backing_team = pick.get("backing_team", "")
        if backing_team and pick.get("market") in ("h2h", "f5_ml", "run_line", "full_game_ml"):
            cap = check_seven_day_cap(backing_team)
            if cap["is_capped"]:
                result.block_parlay(
                    rule="seven_day_team_cap",
                    reason=f"{backing_team} at 7-day ML pick cap (3/3 used)",
                )
                result.block_all(
                    rule="seven_day_team_cap",
                    reason=f"{backing_team} at 7-day cap — cannot add more ML picks",
                )

        pick["hard_rules_result"] = result

    return picks
