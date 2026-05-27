"""
Catcher framing collector — fetches framing runs above average from Baseball Savant
and maps catchers to their teams via the MLB Stats API.
Cached module-level with ~6-hour TTL.
"""
from __future__ import annotations
import csv
import io
import logging
import time

import requests
from .savant_headers import SAVANT_HEADERS

logger = logging.getLogger(__name__)

_FRAMING_CACHE: dict[str, float] = {}   # {catcher_name: framing_runs}
_TEAM_CACHE: dict[str, str] = {}        # {catcher_name: team_abbrev}
_CACHE_TS: float = 0.0
_TTL: float = 6 * 3600

_FRAMING_URL = (
    "https://baseballsavant.mlb.com/catcher_framing"
    "?year=2026&team=&min=q&sort=4&sortDir=desc&csv=true"
)
_ROSTER_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
_TEAMS_URL  = "https://statsapi.mlb.com/api/v1/teams?sportId=1"

_HEADERS = SAVANT_HEADERS


def _safe_float(val: str | None, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def _fetch_framing_csv() -> dict[str, float]:
    """Download the catcher framing CSV from Savant. Returns {name: framing_runs}."""
    try:
        resp = requests.get(_FRAMING_URL, headers=_HEADERS, timeout=12)
        if resp.status_code == 403:
            logger.warning("catcher_framing_collector: Savant returned 403")
            return {}
        resp.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(resp.text)))
    except Exception as exc:
        logger.warning("catcher_framing_collector: CSV fetch failed: %s", exc)
        return {}

    result: dict[str, float] = {}
    for row in rows:
        last  = row.get("last_name", "").strip()
        first = row.get("first_name", "").strip()
        if not last:
            continue
        full = f"{first} {last}".strip() if first else last
        # Column may be "runs_extra_strikes", "framing_runs", or similar
        framing = _safe_float(
            row.get("runs_extra_strikes")
            or row.get("framing_runs")
            or row.get("strike_rate_value"),
            0.0,
        )
        result[full] = framing
        if last:
            result.setdefault(last, framing)
    return result


def _build_catcher_team_map() -> dict[str, str]:
    """
    Walk all active MLB rosters and return {catcher_full_name: team_abbrev}.
    Gracefully skips any team that fails.
    """
    try:
        teams_resp = requests.get(_TEAMS_URL, timeout=12)
        teams_resp.raise_for_status()
        teams_data = teams_resp.json().get("teams", [])
    except Exception as exc:
        logger.warning("catcher_framing_collector: teams fetch failed: %s", exc)
        return {}

    catcher_map: dict[str, str] = {}
    for team in teams_data:
        team_id  = team.get("id")
        team_abbr = team.get("abbreviation", "").upper()
        if not team_id:
            continue
        try:
            r = requests.get(_ROSTER_URL.format(team_id=team_id), timeout=12)
            r.raise_for_status()
            roster = r.json().get("roster", [])
        except Exception as exc:
            logger.debug("catcher_framing_collector: roster fetch failed for %s: %s", team_abbr, exc)
            continue
        for player in roster:
            pos = player.get("position", {}).get("abbreviation", "")
            if pos == "C":
                pname = player.get("person", {}).get("fullName", "").strip()
                if pname:
                    catcher_map[pname] = team_abbr
                    # Also last name key
                    last = pname.split()[-1]
                    catcher_map.setdefault(last, team_abbr)
    return catcher_map


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def get_catcher_framing() -> dict[str, float]:
    """
    Returns {catcher_name: framing_runs_above_avg}.
    Positive = above average (more strikes called), negative = below.
    Refreshes every 6 hours.
    """
    global _FRAMING_CACHE, _CACHE_TS
    now = time.time()
    if _FRAMING_CACHE and (now - _CACHE_TS) < _TTL:
        return _FRAMING_CACHE

    framing = _fetch_framing_csv()
    if framing:
        _FRAMING_CACHE = framing
        _CACHE_TS = now
        logger.info("catcher_framing_collector: loaded %d catchers", len(framing))
    else:
        logger.warning("catcher_framing_collector: empty result; using stale cache")

    return _FRAMING_CACHE


def get_catcher_team_map() -> dict[str, str]:
    """
    Returns {catcher_name: team_abbrev} for all active catchers.
    Refreshes every 6 hours alongside framing data.
    """
    global _TEAM_CACHE, _CACHE_TS
    now = time.time()
    if _TEAM_CACHE and (now - _CACHE_TS) < _TTL:
        return _TEAM_CACHE

    team_map = _build_catcher_team_map()
    if team_map:
        _TEAM_CACHE = team_map
        logger.info("catcher_framing_collector: mapped %d catchers to teams", len(team_map))
    else:
        logger.warning("catcher_framing_collector: empty team map; using stale")

    return _TEAM_CACHE


def get_framing_by_team() -> dict[str, float]:
    """
    Convenience function: returns {team_abbrev: framing_runs} for the starting catcher
    on each team (highest framing value when multiple catchers exist for a team).
    """
    framing  = get_catcher_framing()
    team_map = get_catcher_team_map()

    by_team: dict[str, float] = {}
    for catcher, runs in framing.items():
        team = team_map.get(catcher)
        if not team:
            continue
        # Keep the best (most positive) value per team — proxy for starter
        if team not in by_team or runs > by_team[team]:
            by_team[team] = runs
    return by_team
