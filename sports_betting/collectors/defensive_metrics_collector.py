"""
Defensive metrics collector — fetches team OAA (Outs Above Average) from
Baseball Savant and aggregates to team-level totals.

OAA > 0 = above-average defense → ERA will be lower than FIP suggests.
OAA < 0 = below-average defense → ERA will be higher than FIP suggests.

defensive_run_value = total_team_oaa * 0.82  (runs saved per out above average)
"""
from __future__ import annotations
import csv
import io
import logging
import time

import requests

logger = logging.getLogger(__name__)

_CACHE: dict = {}
_CACHE_TS: float = 0.0
_TTL: float = 6 * 3600  # 6 hours

_OAA_URL = (
    "https://baseballsavant.mlb.com/leaderboard/outs_above_average"
    "?type=Fielder&year=2026&team=1&min=1&csv=true"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Baseball Savant team abbreviation → full team name mapping
# Used to normalize team keys when merging with other data sources
_TEAM_ABBREV_MAP: dict[str, str] = {
    "NYY": "New York Yankees",   "NYM": "New York Mets",
    "BOS": "Boston Red Sox",     "BAL": "Baltimore Orioles",
    "TOR": "Toronto Blue Jays",  "TB":  "Tampa Bay Rays",
    "PHI": "Philadelphia Phillies", "WSH": "Washington Nationals",
    "ATL": "Atlanta Braves",     "MIA": "Miami Marlins",
    "CHC": "Chicago Cubs",       "CWS": "Chicago White Sox",
    "STL": "St. Louis Cardinals", "MIL": "Milwaukee Brewers",
    "CIN": "Cincinnati Reds",    "PIT": "Pittsburgh Pirates",
    "CLE": "Cleveland Guardians", "DET": "Detroit Tigers",
    "MIN": "Minnesota Twins",    "KC":  "Kansas City Royals",
    "HOU": "Houston Astros",     "TEX": "Texas Rangers",
    "COL": "Colorado Rockies",   "ARI": "Arizona Diamondbacks",
    "LAD": "Los Angeles Dodgers", "LAA": "Los Angeles Angels",
    "SF":  "San Francisco Giants", "SD": "San Diego Padres",
    "SEA": "Seattle Mariners",   "OAK": "Oakland Athletics",
    "ATH": "Oakland Athletics",
}


def _safe_float(val: str | None, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def get_team_oaa() -> dict[str, dict]:
    """
    Fetch and aggregate fielder OAA from Baseball Savant.

    Returns dict keyed by team abbreviation (3-letter) →
        {
          total_oaa          : float  (sum of all fielder OAA for that team)
          defensive_run_value: float  (total_oaa * 0.82)
        }

    Returns {} on failure. Cached 6-hour TTL.
    """
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE and (now - _CACHE_TS) < _TTL:
        return _CACHE

    try:
        resp = requests.get(_OAA_URL, headers=_HEADERS, timeout=12)
        if resp.status_code == 403:
            logger.warning("defensive_metrics_collector: Baseball Savant returned 403")
            return _CACHE
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        rows = list(reader)
    except Exception as exc:
        logger.warning("defensive_metrics_collector: fetch failed: %s", exc)
        return _CACHE

    # Accumulate OAA by team abbreviation
    team_oaa_accum: dict[str, float] = {}
    for row in rows:
        # Column may be "team_name" (abbrev), "team", or "team_abbrev"
        team_abbr = (
            row.get("team_name") or row.get("team") or row.get("team_abbrev") or ""
        ).strip().upper()
        if not team_abbr:
            continue

        # OAA column varies: "outs_above_average", "oaa", "OAA"
        oaa_raw = (
            row.get("outs_above_average")
            or row.get("oaa")
            or row.get("OAA")
            or "0"
        )
        oaa_val = _safe_float(oaa_raw, 0.0)
        team_oaa_accum[team_abbr] = team_oaa_accum.get(team_abbr, 0.0) + oaa_val

    result: dict[str, dict] = {}
    for abbr, total_oaa in team_oaa_accum.items():
        drv = round(total_oaa * 0.82, 2)
        result[abbr] = {
            "total_oaa":           round(total_oaa, 1),
            "defensive_run_value": drv,
        }

    if result:
        _CACHE = result
        _CACHE_TS = now
        logger.info(
            "defensive_metrics_collector: loaded OAA for %d teams (%d fielder rows)",
            len(result), len(rows),
        )
    else:
        logger.warning(
            "defensive_metrics_collector: no rows parsed; returning stale or empty cache"
        )

    return _CACHE
