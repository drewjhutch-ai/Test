"""
Platoon Splits Collector — Tier 3, Layer 23.
Fetches batter vs LHP / vs RHP splits from MLB Stats API.
All 30 teams fetched in parallel (60 requests total) to avoid slowdown.
"""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logger = logging.getLogger(__name__)

_BASE = "https://statsapi.mlb.com/api/v1"
_TEAMS_URL  = f"{_BASE}/teams?sportId=1"
_SPLITS_URL = (
    f"{_BASE}/teams/{{team_id}}/stats"
    "?stats=statSplits&group=hitting&season=2026"
    "&sitCodes={sit_code}&gameType=R"
)

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_LG_OPS_VS_LHP = 0.718
_LG_OPS_VS_RHP = 0.710


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def _get_all_teams() -> dict[int, str]:
    try:
        resp = requests.get(_TEAMS_URL, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        return {t["id"]: t.get("abbreviation", str(t["id"])) for t in resp.json().get("teams", [])}
    except Exception as exc:
        logger.warning("platoon_splits_collector: teams fetch failed: %s", exc)
        return {}


def _fetch_ops(team_id: int, sit_code: str) -> float:
    try:
        url = _SPLITS_URL.format(team_id=team_id, sit_code=sit_code)
        resp = requests.get(url, headers=_HEADERS, timeout=5)
        resp.raise_for_status()
        for grp in resp.json().get("stats", []):
            for split in grp.get("splits", []):
                ops = _safe_float(split.get("stat", {}).get("ops"), -1.0)
                if ops >= 0:
                    return ops
    except Exception:
        pass
    return 0.0


def _ops_to_wrc(ops: float, lg: float) -> float:
    return round(100.0 * ops / lg, 1) if lg > 0 and ops > 0 else 100.0


def get_platoon_splits() -> dict[str, dict]:
    """
    Returns {team_abbrev: {wrc_vs_lhp, wrc_vs_rhp, platoon_advantage_lhp, platoon_advantage_rhp}}.
    All 60 API calls (30 teams × 2 handedness) run in parallel.
    """
    teams = _get_all_teams()
    if not teams:
        logger.warning("platoon_splits_collector: could not retrieve team list")
        return {}

    # Submit all 60 fetches concurrently
    tasks: dict = {}
    with ThreadPoolExecutor(max_workers=15) as pool:
        for team_id, abbrev in teams.items():
            tasks[(team_id, "vl")] = pool.submit(_fetch_ops, team_id, "vl")
            tasks[(team_id, "vr")] = pool.submit(_fetch_ops, team_id, "vr")
        ops_results: dict[tuple, float] = {key: fut.result() for key, fut in tasks.items()}

    result: dict[str, dict] = {}
    for team_id, abbrev in teams.items():
        lhp = ops_results.get((team_id, "vl"), 0.0)
        rhp = ops_results.get((team_id, "vr"), 0.0)
        if lhp == 0.0 and rhp == 0.0:
            continue
        wl = _ops_to_wrc(lhp, _LG_OPS_VS_LHP) if lhp > 0 else 100.0
        wr = _ops_to_wrc(rhp, _LG_OPS_VS_RHP) if rhp > 0 else 100.0
        result[abbrev] = {
            "wrc_vs_lhp":            wl,
            "wrc_vs_rhp":            wr,
            "platoon_advantage_lhp": round(wl - 100.0, 1),
            "platoon_advantage_rhp": round(wr - 100.0, 1),
        }

    if not result:
        logger.warning("platoon_splits_collector: no splits data returned")
    else:
        logger.info("platoon_splits_collector: fetched %d teams", len(result))

    return result

