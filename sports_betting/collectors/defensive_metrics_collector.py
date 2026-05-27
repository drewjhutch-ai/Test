"""
Defensive metrics collector.
No OAA (Outs Above Average) in MLB Stats API. Uses team fielding percentage
and error count as a proxy. High fielding% / low errors → positive OAA proxy.
Cached module-level with ~6-hour TTL.
"""
from __future__ import annotations
import logging
import time

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

# League-average fielding% baseline (approximate)
_LG_FIELDING_PCT = 0.9845


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def _fielding_to_oaa(fielding_pct: float, errors: float) -> float:
    """Convert fielding% to an OAA-like proxy on ~[-15, +15] scale."""
    gap = fielding_pct - _LG_FIELDING_PCT
    oaa = gap * 1000.0          # each 0.001 fielding% ≈ 1 OAA unit
    oaa = max(-15.0, min(15.0, oaa))
    # Penalise extra errors (each error beyond ~50 season errors costs ~0.3 OAA)
    if errors > 50:
        oaa -= (errors - 50) * 0.3
    return round(oaa, 1)


def get_team_oaa() -> dict[str, dict]:
    """
    Returns {team_abbrev: {total_oaa, defensive_run_value}}.
    total_oaa derived from fielding% vs league average.
    defensive_run_value = total_oaa * 0.82 (runs saved per OAA unit).
    """
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE and (now - _CACHE_TS) < _TTL:
        return _CACHE

    try:
        teams_resp = requests.get(_TEAMS_URL, headers=_HEADERS, timeout=12)
        teams_resp.raise_for_status()
        teams = teams_resp.json().get("teams", [])
    except Exception as exc:
        logger.warning("defensive_metrics_collector: teams fetch failed: %s", exc)
        return _CACHE

    result: dict[str, dict] = {}

    for team in teams:
        team_id = team.get("id")
        abbrev  = team.get("abbreviation", "").strip().upper()
        if not team_id or not abbrev:
            continue
        try:
            url = _TEAM_STATS_URL.format(team_id=team_id)
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            splits = data.get("stats", [{}])[0].get("splits", [])
            if not splits:
                continue
            stat = splits[0].get("stat", {})
            fielding_pct = _safe_float(stat.get("fielding"), _LG_FIELDING_PCT)
            errors       = _safe_float(stat.get("errors"), 0.0)
            if fielding_pct == 0.0:
                continue
            oaa = _fielding_to_oaa(fielding_pct, errors)
            result[abbrev] = {
                "total_oaa":            oaa,
                "defensive_run_value":  round(oaa * 0.82, 2),
            }
        except Exception:
            continue

    if result:
        _CACHE = result
        _CACHE_TS = now
        logger.info("defensive_metrics_collector: loaded %d teams (fielding%% proxy)", len(result))
    else:
        logger.warning("defensive_metrics_collector: no rows parsed; returning stale or empty cache")

    return _CACHE
