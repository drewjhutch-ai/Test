"""
Defensive metrics collector.
No OAA in MLB Stats API. Uses team fielding% as proxy. Fetches all 30 teams
in parallel to avoid sequential-request slowdown.
Cached module-level with ~6-hour TTL.
"""
from __future__ import annotations
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logger = logging.getLogger(__name__)

_CACHE: dict = {}
_CACHE_TS: float = 0.0
_TTL: float = 6 * 3600

_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId=1&season=2026"
_TEAM_STATS_URL = (
    "https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
    "?stats=season&group=fielding&gameType=R&season=2026"
)
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_LG_FIELDING_PCT = 0.9845


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def _fielding_to_oaa(fielding_pct: float, errors: float) -> float:
    gap = fielding_pct - _LG_FIELDING_PCT
    oaa = max(-15.0, min(15.0, gap * 1000.0))
    if errors > 50:
        oaa -= (errors - 50) * 0.3
    return round(oaa, 1)


def _fetch_team(team_id: int, abbrev: str) -> tuple[str, dict] | None:
    try:
        url = _TEAM_STATS_URL.format(team_id=team_id)
        resp = requests.get(url, headers=_HEADERS, timeout=5)
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        stat = splits[0].get("stat", {})
        fielding_pct = _safe_float(stat.get("fielding"), _LG_FIELDING_PCT)
        errors = _safe_float(stat.get("errors"), 0.0)
        if fielding_pct == 0.0:
            return None
        oaa = _fielding_to_oaa(fielding_pct, errors)
        return abbrev, {"total_oaa": oaa, "defensive_run_value": round(oaa * 0.82, 2)}
    except Exception:
        return None


def get_team_oaa() -> dict[str, dict]:
    """Returns {team_abbrev: {total_oaa, defensive_run_value}}. All 30 teams fetched in parallel."""
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE and (now - _CACHE_TS) < _TTL:
        return _CACHE

    try:
        teams_resp = requests.get(_TEAMS_URL, headers=_HEADERS, timeout=10)
        teams_resp.raise_for_status()
        teams = [
            (t.get("id"), t.get("abbreviation", "").strip().upper())
            for t in teams_resp.json().get("teams", [])
            if t.get("id") and t.get("abbreviation")
        ]
    except Exception as exc:
        logger.warning("defensive_metrics_collector: teams fetch failed: %s", exc)
        return _CACHE

    result: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_team, tid, abbr): abbr for tid, abbr in teams}
        for fut in as_completed(futures):
            val = fut.result()
            if val:
                result[val[0]] = val[1]

    if result:
        _CACHE = result
        _CACHE_TS = now
        logger.info("defensive_metrics_collector: loaded %d teams (fielding%% proxy)", len(result))
    else:
        logger.warning("defensive_metrics_collector: no rows parsed; returning stale or empty cache")

    return _CACHE

