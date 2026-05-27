"""
Bat Speed / Sprint Speed Collector — Tier 3, Layer 25.
MLB Stats API has no bat-tracking or sprint speed data. Uses team slugging%
as a bat-speed proxy and stolen-base rate as a sprint-speed proxy.
Aggregates to team level.
"""
from __future__ import annotations
import logging
import time

import requests

logger = logging.getLogger(__name__)

_CACHE: dict = {}
_CACHE_TS: float = 0.0
_TTL: float = 6 * 3600

_BATTER_URL = (
    "https://statsapi.mlb.com/api/v1/stats"
    "?stats=season&group=hitting&gameType=R&season=2026"
    "&playerPool=ALL&limit=1000&sportId=1"
)
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# League-average baselines (approximate 2024/2025)
_LG_SLG   = 0.400   # maps to avg bat speed ~70 mph
_LG_SB_RATE = 0.12  # SB / (SB+CS), maps to avg sprint ~27.0 ft/sec


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def get_bat_speed_metrics() -> dict[str, dict]:
    """
    Returns {team_abbrev: {avg_bat_speed, avg_attack_angle, avg_sprint_speed, is_speed_team}}

    Proxies (MLB Stats API):
      avg_bat_speed    ≈ 70 + (team_slg - 0.400) * 100   (mph)
      avg_attack_angle = 10.0 (constant — no proxy available)
      avg_sprint_speed ≈ 27.0 + (sb_rate - 0.12) * 20    (ft/sec)
      is_speed_team    = avg_sprint_speed >= 27.5
    """
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE and (now - _CACHE_TS) < _TTL:
        return _CACHE

    try:
        resp = requests.get(_BATTER_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        splits = resp.json()["stats"][0]["splits"]
    except Exception as exc:
        logger.warning("bat_speed_collector: fetch failed: %s", exc)
        return _CACHE

    # Accumulate per team
    team_acc: dict[str, dict] = {}
    for sp in splits:
        team = sp.get("team", {}).get("abbreviation", "").strip().upper()
        if not team:
            continue
        stat = sp.get("stat", {})
        slg = _safe_float(stat.get("slg"), 0.0)
        sb  = _safe_float(stat.get("stolenBases"), 0.0)
        cs  = _safe_float(stat.get("caughtStealing"), 0.0)
        ab  = _safe_float(stat.get("atBats"), 0.0)
        if slg == 0.0 and sb == 0.0:
            continue
        if team not in team_acc:
            team_acc[team] = {"slg_sum": 0.0, "sb": 0.0, "cs": 0.0, "count": 0}
        team_acc[team]["slg_sum"] += slg
        team_acc[team]["sb"]      += sb
        team_acc[team]["cs"]      += cs
        team_acc[team]["count"]   += 1

    result: dict[str, dict] = {}
    for team, acc in team_acc.items():
        n = max(1, acc["count"])
        avg_slg  = acc["slg_sum"] / n
        sb_total = acc["sb"] + acc["cs"]
        sb_rate  = acc["sb"] / sb_total if sb_total > 0 else _LG_SB_RATE

        # Convert proxies to bat-speed / sprint-speed scale
        bat_speed    = round(70.0 + (avg_slg - _LG_SLG) * 100.0, 1)
        sprint_speed = round(27.0 + (sb_rate - _LG_SB_RATE) * 20.0, 2)

        # Clamp to realistic ranges
        bat_speed    = max(60.0, min(80.0, bat_speed))
        sprint_speed = max(24.0, min(30.0, sprint_speed))

        result[team] = {
            "avg_bat_speed":    bat_speed,
            "avg_attack_angle": 10.0,       # no proxy available
            "avg_sprint_speed": sprint_speed,
            "is_speed_team":    sprint_speed >= 27.5,
        }

    if result:
        _CACHE = result
        _CACHE_TS = now
        logger.info("bat_speed_collector: loaded %d teams (SLG/SB proxy, MLB Stats API)", len(result))
    else:
        logger.warning("bat_speed_collector: no data returned")

    return _CACHE
