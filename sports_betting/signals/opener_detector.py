"""
Signal 9: Opener / Bulk Pitcher Detector
Detects when a team is using a reliever as an "opener" (1-2 IP) followed
by a bulk pitcher. Adjusts total expectation up 0.3-0.5 runs and flags
full-game ML risk for the team facing the bulk pitcher.
"""
from __future__ import annotations
import logging

import statsapi

logger = logging.getLogger(__name__)

# Known relievers sometimes used as openers (update as needed)
KNOWN_OPENERS: set[str] = {
    "ryan yarbrough", "john curtiss", "jalen beeks", "josh fleming",
    "brendan mckay", "adam conley", "tommy romero", "jeffrey springs",
    "shane mcclanahan",  # sometimes used as opener in TB system
}


def is_likely_opener(pitcher_name: str, team: str = "") -> bool:
    """Check if a pitcher is likely being used as an opener."""
    if not pitcher_name or pitcher_name.lower() in ("tbd", ""):
        return False
    lower = pitcher_name.lower()
    if lower in KNOWN_OPENERS:
        return True

    # Try to check MLB stats for career SP%
    try:
        results = statsapi.lookup_player(pitcher_name)
        if results:
            pid = results[0]["id"]
            stats = statsapi.player_stat_data(pid, group="pitching", type="career")
            for split in stats.get("stats", []):
                for s in split.get("splits", []):
                    gs = int(s.get("stat", {}).get("gamesStarted", 0) or 0)
                    gp = int(s.get("stat", {}).get("gamesPitched", 1) or 1)
                    sp_pct = gs / max(1, gp)
                    if sp_pct < 0.30:  # Less than 30% career starts = reliever used as opener
                        return True
    except Exception:
        pass
    return False


def detect_opener_game(
    home_sp: str, away_sp: str, home_team: str, away_team: str
) -> dict:
    """
    Detect if either team is using an opener.
    Returns analysis dict with flags and adjustments.
    """
    home_is_opener = is_likely_opener(home_sp, home_team)
    away_is_opener = is_likely_opener(away_sp, away_team)

    result = {
        "home_opener": home_is_opener,
        "away_opener": away_is_opener,
        "total_adjustment": 0.0,
        "notes": [],
        "flags": [],
    }

    if home_is_opener:
        result["total_adjustment"] += 0.35
        result["notes"].append(f"{home_team} using opener ({home_sp}) — bulk pitcher takes over in 2nd; runs in innings 2-4 elevated")
        result["flags"].append(f"OPENER: {home_sp} ({home_team}) — lean OVER first 4 innings")

    if away_is_opener:
        result["total_adjustment"] += 0.35
        result["notes"].append(f"{away_team} using opener ({away_sp}) — bulk pitcher risk")
        result["flags"].append(f"OPENER: {away_sp} ({away_team}) — lean OVER first 4 innings")

    if not home_is_opener and not away_is_opener:
        result["notes"].append("Both starters appear to be traditional starting pitchers")

    return result
