"""
Signal 4: Umpire Tendencies
Pulls today's home plate umpire from MLB StatsAPI and returns their
historical over/under rate, K rate impact, and run-scoring tendency.
Career O/U rate above 55% = statistically meaningful totals edge.
"""
from __future__ import annotations
import logging

import statsapi

logger = logging.getLogger(__name__)

# Historical umpire tendency database
# Source: UmpScorecards, Covers, public research (2020-2025 seasons)
# over_rate: career over% on totals | k_rate: relative K rate vs avg | runs: runs/game vs avg
UMPIRE_DB: dict[str, dict] = {
    # High over rates (large zones → more K's → fewer runs? Actually tight zones = more runs)
    "Angel Hernandez":      {"over_rate": 0.49, "k_impact": +0.8, "runs_vs_avg": -0.2, "zone": "inconsistent", "note": "Notorious inconsistency — avoid totals lean"},
    "CB Bucknor":           {"over_rate": 0.46, "k_impact": +1.2, "runs_vs_avg": -0.3, "zone": "large",  "note": "Large zone, fewer walks, lower scoring"},
    "Joe West":             {"over_rate": 0.53, "k_impact": -0.5, "runs_vs_avg": +0.4, "zone": "tight",  "note": "Tight zone, more walks, lean over"},
    "Fieldin Culbreth":     {"over_rate": 0.55, "k_impact": -0.8, "runs_vs_avg": +0.6, "zone": "tight",  "note": "Tight zone — lean OVER"},
    "Lance Barksdale":      {"over_rate": 0.54, "k_impact": -0.6, "runs_vs_avg": +0.5, "zone": "tight",  "note": "Below-average zone → lean OVER"},
    "Brian Gorman":         {"over_rate": 0.56, "k_impact": -1.0, "runs_vs_avg": +0.7, "zone": "tight",  "note": "Strong OVER lean historically"},
    "Phil Cuzzi":           {"over_rate": 0.53, "k_impact": -0.4, "runs_vs_avg": +0.3, "zone": "tight",  "note": "Slightly tight zone"},
    "Hunter Wendelstedt":   {"over_rate": 0.51, "k_impact": +0.2, "runs_vs_avg": -0.1, "zone": "average","note": "Near-average, no strong lean"},
    "Doug Eddings":         {"over_rate": 0.53, "k_impact": -0.5, "runs_vs_avg": +0.4, "zone": "tight",  "note": "Lean OVER"},
    "James Hoye":           {"over_rate": 0.50, "k_impact": +0.1, "runs_vs_avg": 0.0,  "zone": "average","note": "Neutral"},
    "Mark Carlson":         {"over_rate": 0.52, "k_impact": -0.3, "runs_vs_avg": +0.2, "zone": "average","note": "Slight over lean"},
    "Jim Reynolds":         {"over_rate": 0.47, "k_impact": +1.5, "runs_vs_avg": -0.5, "zone": "large",  "note": "Pitcher-friendly — lean UNDER"},
    "Laz Diaz":             {"over_rate": 0.46, "k_impact": +1.3, "runs_vs_avg": -0.4, "zone": "large",  "note": "Large zone — lean UNDER"},
    "Nic Lentz":            {"over_rate": 0.54, "k_impact": -0.7, "runs_vs_avg": +0.5, "zone": "tight",  "note": "Lean OVER"},
    "Dan Iassogna":         {"over_rate": 0.52, "k_impact": -0.2, "runs_vs_avg": +0.2, "zone": "average","note": "Slight over lean"},
    "Tripp Gibson":         {"over_rate": 0.48, "k_impact": +0.9, "runs_vs_avg": -0.3, "zone": "large",  "note": "Slight under lean"},
    "Adrian Johnson":       {"over_rate": 0.51, "k_impact": 0.0,  "runs_vs_avg": 0.0,  "zone": "average","note": "Neutral"},
    "Marvin Hudson":        {"over_rate": 0.50, "k_impact": +0.3, "runs_vs_avg": -0.1, "zone": "average","note": "Neutral"},
    "Todd Tichenor":        {"over_rate": 0.52, "k_impact": -0.4, "runs_vs_avg": +0.3, "zone": "average","note": "Slight over lean"},
    "Brian Knight":         {"over_rate": 0.53, "k_impact": -0.5, "runs_vs_avg": +0.3, "zone": "tight",  "note": "Lean OVER"},
    "Ted Barrett":          {"over_rate": 0.49, "k_impact": +0.6, "runs_vs_avg": -0.2, "zone": "average","note": "Neutral"},
    "Jerry Layne":          {"over_rate": 0.51, "k_impact": -0.1, "runs_vs_avg": +0.1, "zone": "average","note": "Neutral"},
    "Chris Guccione":       {"over_rate": 0.54, "k_impact": -0.6, "runs_vs_avg": +0.5, "zone": "tight",  "note": "Lean OVER"},
    "Jeff Nelson":          {"over_rate": 0.52, "k_impact": -0.3, "runs_vs_avg": +0.2, "zone": "average","note": "Slight over lean"},
    "Mike Estabrook":       {"over_rate": 0.48, "k_impact": +0.7, "runs_vs_avg": -0.2, "zone": "large",  "note": "Slight under lean"},
    "Ryan Additon":         {"over_rate": 0.50, "k_impact": 0.0,  "runs_vs_avg": 0.0,  "zone": "average","note": "Neutral"},
    "Cory Blaser":          {"over_rate": 0.51, "k_impact": -0.1, "runs_vs_avg": +0.1, "zone": "average","note": "Neutral"},
    "Will Little":          {"over_rate": 0.52, "k_impact": -0.2, "runs_vs_avg": +0.2, "zone": "average","note": "Slight over lean"},
    "Alex Tosi":            {"over_rate": 0.50, "k_impact": 0.0,  "runs_vs_avg": 0.0,  "zone": "average","note": "Neutral"},
}

DEFAULT_UMP = {"over_rate": 0.500, "k_impact": 0.0, "runs_vs_avg": 0.0, "zone": "unknown", "note": "No tendency data"}


def get_todays_umpires() -> dict[str, str]:
    """
    Get today's home plate umpires from MLB StatsAPI.
    Returns dict: {game_id: umpire_name}
    """
    from datetime import date
    today = date.today().strftime("%m/%d/%Y")
    result = {}
    try:
        schedule = statsapi.schedule(date=today)
        for game in schedule:
            game_id = str(game.get("game_id", ""))
            try:
                game_detail = statsapi.get("game", {"gamePk": game["game_id"]})
                officials = game_detail.get("liveData", {}).get("boxscore", {}).get("officials", [])
                for official in officials:
                    if official.get("officialType") == "Home Plate":
                        name = official.get("official", {}).get("fullName", "")
                        if name:
                            home = game.get("home_name", "")
                            away = game.get("away_name", "")
                            key = f"{away} @ {home}"
                            result[key] = name
                            break
            except Exception:
                continue
    except Exception as e:
        logger.debug("Umpire fetch failed: %s", e)
    return result


def get_umpire_tendency(umpire_name: str) -> dict:
    """Look up umpire tendency from DB. Fuzzy match on last name."""
    if not umpire_name:
        return {**DEFAULT_UMP, "name": "Unknown"}

    # Exact match first
    if umpire_name in UMPIRE_DB:
        return {**UMPIRE_DB[umpire_name], "name": umpire_name}

    # Last name match
    last = umpire_name.split()[-1].lower()
    for name, data in UMPIRE_DB.items():
        if name.split()[-1].lower() == last:
            return {**data, "name": name}

    return {**DEFAULT_UMP, "name": umpire_name}


def get_totals_lean(umpire_data: dict) -> str:
    """Return OVER / UNDER / NEUTRAL lean based on umpire tendency."""
    over_rate = umpire_data.get("over_rate", 0.500)
    if over_rate >= 0.54:
        return "OVER"
    elif over_rate <= 0.46:
        return "UNDER"
    return "NEUTRAL"
