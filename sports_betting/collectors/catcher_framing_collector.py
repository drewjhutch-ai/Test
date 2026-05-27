"""
Catcher framing collector.
No framing data in MLB Stats API. Uses team caught-stealing percentage
(CS / (CS + SB)) as a proxy — good framers also tend to suppress the running
game. Converts CS% to a framing_runs-like scale.
Cached module-level with ~6-hour TTL.
"""
from __future__ import annotations
import logging
import time

import requests

logger = logging.getLogger(__name__)

_FRAMING_CACHE: dict[str, float] = {}
_CACHE_TS: float = 0.0
_TTL: float = 6 * 3600

_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId=1&season=2026"
_TEAM_STATS_URL = (
    "https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
    "?stats=season&group=pitching&gameType=R&season=2026"
)
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def _cs_pct_to_framing(cs_pct: float) -> float:
    """Convert caught-stealing % to a framing_runs-like proxy on ~[-6, +6] scale."""
    if cs_pct > 0.35:
        return 5.0
    if cs_pct > 0.30:
        return 3.0
    if cs_pct > 0.25:
        return 1.0
    if cs_pct > 0.20:
        return -1.0
    if cs_pct > 0.15:
        return -3.0
    return -5.0


def get_framing_by_team() -> dict[str, float]:
    """
    Returns {team_abbrev: framing_runs_proxy}.
    Positive = above-average framing; negative = below-average.
    """
    global _FRAMING_CACHE, _CACHE_TS
    now = time.time()
    if _FRAMING_CACHE and (now - _CACHE_TS) < _TTL:
        return _FRAMING_CACHE

    try:
        teams_resp = requests.get(_TEAMS_URL, headers=_HEADERS, timeout=12)
        teams_resp.raise_for_status()
        teams = teams_resp.json().get("teams", [])
    except Exception as exc:
        logger.warning("catcher_framing_collector: teams fetch failed: %s", exc)
        return _FRAMING_CACHE

    result: dict[str, float] = {}

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
            cs = _safe_float(stat.get("caughtStealing"), 0.0)
            sb = _safe_float(stat.get("stolenBases"), 0.0)
            total = cs + sb
            if total < 5:
                continue
            cs_pct = cs / total
            result[abbrev] = _cs_pct_to_framing(cs_pct)
        except Exception:
            continue

    if result:
        _FRAMING_CACHE = result
        _CACHE_TS = now
        logger.info("catcher_framing_collector: loaded %d teams (CS%% proxy)", len(result))
    else:
        logger.warning("catcher_framing_collector: empty result; using stale cache")

    return _FRAMING_CACHE
