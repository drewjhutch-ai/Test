"""
Bat Speed / Sprint Speed Collector — Tier 3, Layer 25.
Uses team-level hitting stats from MLB Stats API (SLG as bat-speed proxy,
SB rate as sprint-speed proxy). Fetches all 30 teams in parallel.
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
    "?stats=season&group=hitting&gameType=R&season=2026"
)
_HEADERS = {"User-Agent": "Mozilla/5.0"}

_LG_SLG    = 0.400
_LG_SB_RATE = 0.12


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def _fetch_team(team_id: int, abbrev: str) -> tuple[str, dict] | None:
    try:
        url = _TEAM_STATS_URL.format(team_id=team_id)
        resp = requests.get(url, headers=_HEADERS, timeout=5)
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        stat = splits[0].get("stat", {})
        slg = _safe_float(stat.get("slg"), 0.0)
        sb  = _safe_float(stat.get("stolenBases"), 0.0)
        cs  = _safe_float(stat.get("caughtStealing"), 0.0)
        if slg == 0.0 and sb == 0.0:
            return None

        sb_total = sb + cs
        sb_rate  = sb / sb_total if sb_total > 0 else _LG_SB_RATE
        bat_speed    = max(60.0, min(80.0, round(70.0 + (slg - _LG_SLG) * 100.0, 1)))
        sprint_speed = max(24.0, min(30.0, round(27.0 + (sb_rate - _LG_SB_RATE) * 20.0, 2)))
        return abbrev, {
            "avg_bat_speed":    bat_speed,
            "avg_attack_angle": 10.0,
            "avg_sprint_speed": sprint_speed,
            "is_speed_team":    sprint_speed >= 27.5,
        }
    except Exception:
        return None


def get_bat_speed_metrics() -> dict[str, dict]:
    """Returns {team_abbrev: {avg_bat_speed, avg_attack_angle, avg_sprint_speed, is_speed_team}}."""
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
        logger.warning("bat_speed_collector: teams fetch failed: %s", exc)
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
        logger.info("bat_speed_collector: loaded %d teams (SLG/SB proxy)", len(result))
    else:
        logger.warning("bat_speed_collector: no data returned")

    return _CACHE
