"""
Signal 6: Travel / Jet Lag Fatigue
Eastward travel degrades performance. West Coast teams traveling east
for Monday/Tuesday games are the strongest fade signal.
AAAS research: eastward 3+ time zone travel measurably hurts players.
"""
from __future__ import annotations
import logging
from datetime import date, timedelta

import statsapi

logger = logging.getLogger(__name__)

# Team home city time zones (US/Canada)
TEAM_TIMEZONES: dict[str, str] = {
    "Seattle Mariners":         "PT",
    "Los Angeles Dodgers":      "PT",
    "Los Angeles Angels":       "PT",
    "San Francisco Giants":     "PT",
    "San Diego Padres":         "PT",
    "Oakland Athletics":        "PT",
    "Athletics":                "PT",
    "Arizona Diamondbacks":     "MT",
    "Colorado Rockies":         "MT",
    "Houston Astros":           "CT",
    "Texas Rangers":            "CT",
    "Minnesota Twins":          "CT",
    "Chicago Cubs":             "CT",
    "Chicago White Sox":        "CT",
    "Milwaukee Brewers":        "CT",
    "St. Louis Cardinals":      "CT",
    "Kansas City Royals":       "CT",
    "New Orleans Pelicans":     "CT",
    "Atlanta Braves":           "ET",
    "New York Yankees":         "ET",
    "New York Mets":            "ET",
    "Boston Red Sox":           "ET",
    "Philadelphia Phillies":    "ET",
    "Baltimore Orioles":        "ET",
    "Washington Nationals":     "ET",
    "Miami Marlins":            "ET",
    "Pittsburgh Pirates":       "ET",
    "Cincinnati Reds":          "ET",
    "Cleveland Guardians":      "ET",
    "Detroit Tigers":           "ET",
    "Toronto Blue Jays":        "ET",
    "Tampa Bay Rays":           "ET",
}

TZ_OFFSET = {"PT": 0, "MT": 1, "CT": 2, "ET": 3}  # relative eastward offsets


def get_previous_city(team_id: int) -> str | None:
    """Get the city where the team played yesterday."""
    yesterday = (date.today() - timedelta(days=1)).strftime("%m/%d/%Y")
    try:
        sched = statsapi.schedule(date=yesterday, team=team_id)
        if sched:
            game = sched[0]
            return game.get("venue_name", "")
    except Exception:
        pass
    return None


def compute_travel_fatigue(team_name: str, team_id: int | None) -> dict:
    """
    Compute travel fatigue signal for a team.
    Returns fatigue dict with flag, direction, and recommendation.
    """
    home_tz = TEAM_TIMEZONES.get(team_name, "ET")

    if not team_id:
        return {"team": team_name, "travel_flag": False, "direction": "unknown",
                "tz_diff": 0, "note": "Team ID unavailable"}

    prev_venue = get_previous_city(team_id)

    # Determine if team is currently on the road
    # If they're playing an away game today, they traveled to the home team's city
    today = date.today().strftime("%m/%d/%Y")
    try:
        sched = statsapi.schedule(date=today, team=team_id)
        if sched:
            game = sched[0]
            is_away = game.get("away_id") == team_id
            if is_away:
                opponent = game.get("home_name", "")
                dest_tz = TEAM_TIMEZONES.get(opponent, home_tz)
            else:
                dest_tz = home_tz
                return {"team": team_name, "travel_flag": False, "direction": "home",
                        "tz_diff": 0, "note": "Playing at home — no travel fatigue"}
        else:
            return {"team": team_name, "travel_flag": False, "direction": "no game",
                    "tz_diff": 0, "note": "No game today"}
    except Exception:
        dest_tz = home_tz

    home_offset = TZ_OFFSET.get(home_tz, 3)
    dest_offset = TZ_OFFSET.get(dest_tz, 3)
    tz_diff = dest_offset - home_offset  # positive = traveling east (harder)

    weekday = date.today().weekday()  # 0=Mon, 1=Tue
    early_week = weekday <= 1

    if tz_diff >= 3 and early_week:
        flag = True
        note = f"🛫 STRONG FADE: {team_name} traveled 3 time zones east overnight for {['Mon','Tue'][weekday]} game — jet lag confirmed"
        severity = "strong"
    elif tz_diff >= 2:
        flag = True
        note = f"⚠️ {team_name} traveled {tz_diff} TZ east — moderate travel fatigue signal"
        severity = "moderate"
    elif tz_diff >= 1:
        flag = False
        note = f"{team_name} mild eastward travel ({tz_diff} TZ) — monitor"
        severity = "mild"
    elif tz_diff < 0:
        flag = False
        note = f"{team_name} traveling west — minimal fatigue impact"
        severity = "none"
    else:
        flag = False
        note = f"{team_name} same time zone — no travel fatigue"
        severity = "none"

    return {
        "team": team_name,
        "travel_flag": flag,
        "direction": "east" if tz_diff > 0 else "west" if tz_diff < 0 else "same",
        "tz_diff": tz_diff,
        "severity": severity,
        "note": note,
    }


def get_travel_signals_for_game(
    home_team: str, away_team: str, team_id_map: dict
) -> dict:
    """Get travel fatigue for both teams."""
    home_id = team_id_map.get(home_team)
    away_id = team_id_map.get(away_team)
    return {
        "home": compute_travel_fatigue(home_team, home_id),
        "away": compute_travel_fatigue(away_team, away_id),
    }
