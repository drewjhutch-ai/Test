"""
Luck Metrics Collector — Tier 3, Layer 22.
Computes real team BABIP = (H - HR) / (AB - K - HR + SF) from MLB Stats API
team-level hitting stats. Fetches all 30 teams in parallel.
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


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def _compute_babip(h: float, hr: float, ab: float, k: float, sf: float) -> float:
    denom = ab - k - hr + sf
    if denom < 10:
        return -1.0
    return (h - hr) / denom


def _fetch_team(team_id: int, abbrev: str) -> tuple[str, dict] | None:
    try:
        url = _TEAM_STATS_URL.format(team_id=team_id)
        resp = requests.get(url, headers=_HEADERS, timeout=5)
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        stat = splits[0].get("stat", {})
        h   = _safe_float(stat.get("hits"))
        hr  = _safe_float(stat.get("homeRuns"))
        ab  = _safe_float(stat.get("atBats"))
        k   = _safe_float(stat.get("strikeOuts"))
        sf  = _safe_float(stat.get("sacFlies"))
        babip = _compute_babip(h, hr, ab, k, sf)
        if babip < 0:
            return None

        luck_score = 0.0
        if babip > 0.320:
            luck_score += 2
        elif babip < 0.270:
            luck_score -= 2

        return abbrev, {
            "babip":      round(babip, 3),
            "lob_pct":    0.720,
            "luck_score": luck_score,
        }
    except Exception:
        return None


def get_luck_metrics() -> dict[str, dict]:
    """Returns {team_abbrev: {babip, lob_pct, luck_score}}. All 30 teams fetched in parallel."""
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
        logger.warning("luck_metrics_collector: teams fetch failed: %s", exc)
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
        logger.info("luck_metrics_collector: computed BABIP for %d teams", len(result))
    else:
        logger.warning("luck_metrics_collector: no data returned")

    return _CACHE
