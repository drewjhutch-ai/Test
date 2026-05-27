"""
Travel / time-zone fatigue collector.

Uses the MLB Stats API schedule to reconstruct where each team played yesterday
and 2 days ago, then calculates time-zone change direction and magnitude.

Westward travel crossing 3+ time zones is the strongest fatigue signal in MLB.
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

_CACHE: dict = {}
_CACHE_TS: float = 0.0
_TTL: float = 6 * 3600  # 6 hours

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

# Hard-coded city timezone offsets (UTC standard-time hours).
# We use named timezones via pytz when available, fall back to offsets.
TEAM_TIMEZONES: dict[str, str] = {
    "New York Yankees":       "America/New_York",
    "New York Mets":          "America/New_York",
    "Boston Red Sox":         "America/New_York",
    "Baltimore Orioles":      "America/New_York",
    "Toronto Blue Jays":      "America/Toronto",
    "Tampa Bay Rays":         "America/New_York",
    "Philadelphia Phillies":  "America/New_York",
    "Washington Nationals":   "America/New_York",
    "Atlanta Braves":         "America/New_York",
    "Miami Marlins":          "America/New_York",
    "Chicago Cubs":           "America/Chicago",
    "Chicago White Sox":      "America/Chicago",
    "St. Louis Cardinals":    "America/Chicago",
    "Milwaukee Brewers":      "America/Chicago",
    "Cincinnati Reds":        "America/New_York",
    "Pittsburgh Pirates":     "America/New_York",
    "Cleveland Guardians":    "America/New_York",
    "Detroit Tigers":         "America/New_York",
    "Minnesota Twins":        "America/Chicago",
    "Kansas City Royals":     "America/Chicago",
    "Houston Astros":         "America/Chicago",
    "Texas Rangers":          "America/Chicago",
    "Colorado Rockies":       "America/Denver",
    "Arizona Diamondbacks":   "America/Phoenix",
    "Los Angeles Dodgers":    "America/Los_Angeles",
    "Los Angeles Angels":     "America/Los_Angeles",
    "San Francisco Giants":   "America/Los_Angeles",
    "San Diego Padres":       "America/Los_Angeles",
    "Seattle Mariners":       "America/Los_Angeles",
    "Oakland Athletics":      "America/Los_Angeles",
    "Athletics":              "America/Los_Angeles",
}

# UTC offset (hours) for each IANA timezone (standard time, not DST-adjusted).
# We use a simple approximation: ET=-5, CT=-6, MT=-7, PT=-8.
# During DST these shift by +1h, but cross-tz comparison is still directionally correct.
_TZ_UTC_OFFSET: dict[str, int] = {
    "America/New_York":    -5,
    "America/Toronto":     -5,
    "America/Chicago":     -6,
    "America/Denver":      -7,
    "America/Phoenix":     -7,
    "America/Los_Angeles": -8,
}


def _tz_offset(team_name: str) -> int:
    """Return approximate UTC offset (hours) for team's home city."""
    tz = TEAM_TIMEZONES.get(team_name, "America/New_York")
    return _TZ_UTC_OFFSET.get(tz, -5)


def _fetch_schedule(date_str: str) -> dict[str, str]:
    """
    Fetch MLB schedule for date_str and return {team_name: home_venue_city}.
    Returns {} on failure.
    """
    try:
        resp = requests.get(
            MLB_SCHEDULE_URL,
            params={
                "sportId": 1,
                "date": date_str,
                "hydrate": "team,venue",
            },
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("travel_fatigue_collector: schedule fetch failed for %s: %s", date_str, exc)
        return {}

    # Build {team_name: city_where_playing} for each game on that date
    team_location: dict[str, str] = {}
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            venue_name = game.get("venue", {}).get("name", "")
            home_team = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            away_team = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
            if home_team:
                team_location[home_team] = home_team  # home team is in their home city
            if away_team:
                team_location[away_team] = home_team  # away team is in the home team's city
    return team_location


def get_travel_fatigue(today_str: str) -> dict[str, dict]:
    """
    Calculate travel fatigue for all teams playing today.

    Parameters
    ----------
    today_str : "YYYY-MM-DD"

    Returns
    -------
    dict keyed by team name →
        {
          tz_change_hours : float  (positive = traveled westward = harder)
          direction       : "west" | "east" | "none"
          consecutive_road: int    (games played away from home in last 3 days)
          fatigue_flag    : bool   (True = 3+ tz westward)
        }
    """
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE and today_str in _CACHE and (now - _CACHE_TS) < _TTL:
        return _CACHE[today_str]

    today_dt   = datetime.strptime(today_str, "%Y-%m-%d")
    yesterday  = (today_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    two_ago    = (today_dt - timedelta(days=2)).strftime("%Y-%m-%d")

    today_loc     = _fetch_schedule(today_str)
    yesterday_loc = _fetch_schedule(yesterday)
    two_ago_loc   = _fetch_schedule(two_ago)

    result: dict[str, dict] = {}

    for team, today_city in today_loc.items():
        prev_city = yesterday_loc.get(team) or two_ago_loc.get(team)

        if not prev_city:
            # No prior game data — assume no travel
            result[team] = {
                "tz_change_hours": 0.0,
                "direction": "none",
                "consecutive_road": 0,
                "fatigue_flag": False,
            }
            continue

        today_offset = _tz_offset(today_city)
        prev_offset  = _tz_offset(prev_city)

        # tz_change > 0 = traveled westward (lost hours, harder)
        tz_change = prev_offset - today_offset

        direction: str
        if tz_change >= 1:
            direction = "west"
        elif tz_change <= -1:
            direction = "east"
        else:
            direction = "none"

        # Count consecutive road games (away from home city) in last 3 days
        home_tz   = _tz_offset(team)
        consec_road = 0
        for loc in (two_ago_loc.get(team), yesterday_loc.get(team), today_city):
            if loc is None:
                continue
            loc_tz = _tz_offset(loc)
            if loc_tz != home_tz:
                consec_road += 1

        fatigue_flag = (abs(tz_change) >= 3 and direction == "west")

        result[team] = {
            "tz_change_hours": float(tz_change),
            "direction": direction,
            "consecutive_road": consec_road,
            "fatigue_flag": fatigue_flag,
        }

    if result:
        _CACHE[today_str] = result
        _CACHE_TS = now
        logger.info(
            "travel_fatigue_collector: processed %d teams for %s", len(result), today_str
        )

    return result
