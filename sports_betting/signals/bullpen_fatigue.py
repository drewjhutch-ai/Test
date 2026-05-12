"""
Signal 1: Bullpen Fatigue Tracker
Tracks reliever usage over the last 48-72 hours via MLB StatsAPI.
Flags teams whose top relievers are overworked.
Proven signal: Better Bullpen Underdog = +60 units in 2025 (VSiN).
"""
from __future__ import annotations
import logging
from datetime import date, timedelta
from collections import defaultdict

import statsapi

logger = logging.getLogger(__name__)


def get_recent_game_pks(team_id: int, days_back: int = 3) -> list[int]:
    """Return gamePks for a team's last N days of games."""
    end = date.today()
    start = end - timedelta(days=days_back)
    try:
        sched = statsapi.schedule(
            start_date=start.strftime("%m/%d/%Y"),
            end_date=end.strftime("%m/%d/%Y"),
            team=team_id,
        )
        return [g["game_id"] for g in sched if g.get("status") in ("Final", "Game Over")]
    except Exception as e:
        logger.debug("schedule fetch failed for team %s: %s", team_id, e)
        return []


def get_reliever_usage(game_pk: int) -> list[dict]:
    """Return list of relievers and their IP/pitches from a completed game."""
    try:
        box = statsapi.boxscore_data(game_pk)
        pitchers = []
        for side in ("home", "away"):
            team_pitchers = box.get(side, {}).get("pitchers", [])
            for i, pid in enumerate(team_pitchers):
                if i == 0:
                    continue  # skip starter
                pdata = box.get("playerInfo", {}).get(f"ID{pid}", {})
                stats = box.get(side, {}).get("players", {}).get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
                ip = float(stats.get("inningsPitched", "0").replace(".1", ".33").replace(".2", ".67") or 0)
                pitches = int(stats.get("numberOfPitches", 0) or 0)
                if pitches > 0 or ip > 0:
                    pitchers.append({
                        "id": pid,
                        "name": pdata.get("fullName", f"P{pid}"),
                        "team": side,
                        "ip": ip,
                        "pitches": pitches,
                        "high_leverage": pitches >= 15,  # proxy for high-leverage appearance
                    })
        return pitchers
    except Exception as e:
        logger.debug("boxscore fetch failed for game %s: %s", game_pk, e)
        return []


def compute_bullpen_fatigue(team_id: int, team_name: str) -> dict:
    """
    Compute bullpen fatigue score for a team.
    Returns dict with fatigue level, tired pitchers, and recommendation.
    """
    game_pks = get_recent_game_pks(team_id, days_back=3)
    if not game_pks:
        return {"team": team_name, "fatigue_level": "unknown", "score": 0.0,
                "tired_pitchers": [], "note": "No recent game data available"}

    pitcher_days: dict[int, list[dict]] = defaultdict(list)
    for pk in game_pks[-3:]:  # last 3 games
        for p in get_reliever_usage(pk):
            if p["team"] in ("home", "away"):  # both sides — filter by team later
                pitcher_days[p["id"]].append(p)

    # Count consecutive days used and high-leverage appearances
    tired = []
    for pid, appearances in pitcher_days.items():
        if len(appearances) >= 2:
            total_pitches = sum(a["pitches"] for a in appearances)
            high_lev_count = sum(1 for a in appearances if a["high_leverage"])
            tired.append({
                "name": appearances[0]["name"],
                "appearances": len(appearances),
                "total_pitches": total_pitches,
                "high_leverage_apps": high_lev_count,
            })

    tired.sort(key=lambda x: x["total_pitches"], reverse=True)
    top_tired = tired[:3]

    # Fatigue score: 0 = fresh, 1 = heavily fatigued
    if not top_tired:
        score = 0.0
        level = "fresh"
        note = "Bullpen well-rested"
    elif top_tired[0]["appearances"] >= 3 or top_tired[0]["total_pitches"] >= 60:
        score = 0.85
        level = "heavily_fatigued"
        note = f"⚠️ {top_tired[0]['name']} used {top_tired[0]['appearances']}x in 3 days ({top_tired[0]['total_pitches']} pitches)"
    elif len(top_tired) >= 2 and top_tired[0]["appearances"] >= 2:
        score = 0.60
        level = "moderately_fatigued"
        note = f"Top relievers used 2+ days in a row — degraded late-game reliability"
    elif top_tired and top_tired[0]["high_leverage_apps"] >= 2:
        score = 0.40
        level = "mildly_fatigued"
        note = f"High-leverage relievers stretched — monitor availability"
    else:
        score = 0.15
        level = "slightly_used"
        note = "Normal recent usage"

    return {
        "team": team_name,
        "fatigue_level": level,
        "score": score,
        "tired_pitchers": top_tired,
        "note": note,
    }


def get_bullpen_fatigue_for_game(
    home_team: str, away_team: str, team_id_map: dict
) -> dict:
    """Return bullpen fatigue for both teams in a game."""
    home_id = team_id_map.get(home_team)
    away_id = team_id_map.get(away_team)

    home_fatigue = compute_bullpen_fatigue(home_id, home_team) if home_id else {"score": 0, "note": "ID unknown", "fatigue_level": "unknown"}
    away_fatigue = compute_bullpen_fatigue(away_id, away_team) if away_id else {"score": 0, "note": "ID unknown", "fatigue_level": "unknown"}

    return {"home": home_fatigue, "away": away_fatigue}
