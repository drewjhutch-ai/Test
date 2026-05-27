"""
xStats collector — fetches pitcher xERA/xFIP/SIERA and team xwOBA from Baseball Savant.
Cached module-level with ~6-hour TTL.
"""
from __future__ import annotations
import csv
import io
import logging
import time

import requests

logger = logging.getLogger(__name__)

_PITCHER_CACHE: dict = {}
_PITCHER_CACHE_TS: float = 0.0

_BATTER_CACHE: dict = {}
_BATTER_CACHE_TS: float = 0.0

_TTL: float = 6 * 3600  # 6 hours

_PITCHER_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=pitcher&year=2026&position=&team=&min=25&csv=true"
)
_BATTER_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=batter&year=2026&position=&team=&min=25&csv=true"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _fetch_csv(url: str) -> list[dict]:
    """Fetch a CSV URL and return list-of-dicts. Returns [] on any error."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        if resp.status_code == 403:
            logger.warning("Baseball Savant returned 403 for %s", url)
            return []
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        return list(reader)
    except Exception as exc:
        logger.warning("xstats_collector fetch failed (%s): %s", url, exc)
        return []


def _safe_float(val: str | None, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def get_pitcher_xstats() -> dict[str, dict]:
    """
    Returns dict keyed by player_name (last, first or full).
    Each value: {xera, xfip, siera, era}
    """
    global _PITCHER_CACHE, _PITCHER_CACHE_TS
    now = time.time()
    if _PITCHER_CACHE and (now - _PITCHER_CACHE_TS) < _TTL:
        return _PITCHER_CACHE

    rows = _fetch_csv(_PITCHER_URL)
    result: dict[str, dict] = {}
    for row in rows:
        # Savant CSV columns vary; common names: last_name, first_name, xera, xfip, siera, era
        last = row.get("last_name", "").strip()
        first = row.get("first_name", "").strip()
        if not last:
            continue
        full = f"{first} {last}".strip() if first else last
        entry = {
            "xera":  _safe_float(row.get("xera"),  4.50),
            "xfip":  _safe_float(row.get("xfip"),  4.50),
            "siera": _safe_float(row.get("siera"), 4.50),
            "era":   _safe_float(row.get("era"),   4.50),
        }
        result[full] = entry
        # Also index by "Last" alone for fuzzy matching in layer
        if last:
            result[last] = entry

    if result:
        _PITCHER_CACHE = result
        _PITCHER_CACHE_TS = now
        logger.info("xstats_collector: loaded %d pitcher xstat rows", len(rows))
    else:
        logger.warning("xstats_collector: no pitcher rows fetched; returning stale or empty cache")

    return _PITCHER_CACHE


def get_team_xwoba() -> dict[str, dict]:
    """
    Returns dict keyed by team abbreviation.
    Each value: {xwoba, woba}
    Aggregates batter rows by team.
    """
    global _BATTER_CACHE, _BATTER_CACHE_TS
    now = time.time()
    if _BATTER_CACHE and (now - _BATTER_CACHE_TS) < _TTL:
        return _BATTER_CACHE

    rows = _fetch_csv(_BATTER_URL)

    # Accumulate per team: sum xwoba and woba, count rows
    team_accum: dict[str, dict] = {}
    for row in rows:
        team = row.get("team_name_abbrev", row.get("team", "")).strip().upper()
        if not team:
            continue
        xwoba = _safe_float(row.get("xwoba"), 0.0)
        woba  = _safe_float(row.get("woba"),  0.0)
        if xwoba == 0.0 and woba == 0.0:
            continue
        if team not in team_accum:
            team_accum[team] = {"xwoba_sum": 0.0, "woba_sum": 0.0, "count": 0}
        team_accum[team]["xwoba_sum"] += xwoba
        team_accum[team]["woba_sum"]  += woba
        team_accum[team]["count"]     += 1

    result: dict[str, dict] = {}
    for team, acc in team_accum.items():
        n = max(1, acc["count"])
        result[team] = {
            "xwoba": round(acc["xwoba_sum"] / n, 3),
            "woba":  round(acc["woba_sum"]  / n, 3),
        }

    if result:
        _BATTER_CACHE = result
        _BATTER_CACHE_TS = now
        logger.info("xstats_collector: loaded xwOBA for %d teams", len(result))
    else:
        logger.warning("xstats_collector: no batter rows fetched; returning stale or empty cache")

    return _BATTER_CACHE
