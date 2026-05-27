"""
Platoon Splits Collector — Tier 3, Layer 23.
Fetches batter vs LHP / vs RHP splits from MLB Stats API.
Aggregates to team level: avg wRC+ proxy (OPS) vs LHP and vs RHP.
"""
from __future__ import annotations
import logging

import requests

logger = logging.getLogger(__name__)

_BASE = "https://statsapi.mlb.com/api/v1"
_TEAMS_URL   = f"{_BASE}/teams?sportId=1"
_SPLITS_URL  = (
    f"{_BASE}/teams/{{team_id}}/stats"
    "?stats=statSplits&group=hitting&season=2026"
    "&sitCodes={sit_code}&gameType=R"
)

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def _fetch_json(url: str):
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("platoon_splits_collector: fetch failed (%s): %s", url[:80], exc)
        return None


def _get_all_teams() -> dict[int, str]:
    """Returns {team_id: abbrev}."""
    data = _fetch_json(_TEAMS_URL)
    if not data:
        return {}
    result = {}
    for t in data.get("teams", []):
        result[t["id"]] = t.get("abbreviation", str(t["id"]))
    return result


def _get_team_ops_vs(team_id: int, sit_code: str) -> float:
    """Return team OPS for the given situation code ('vl' or 'vr'). 0.0 on failure."""
    url = _SPLITS_URL.format(team_id=team_id, sit_code=sit_code)
    data = _fetch_json(url)
    if not data:
        return 0.0
    for split_group in data.get("stats", []):
        for split in split_group.get("splits", []):
            stat = split.get("stat", {})
            ops = _safe_float(stat.get("ops"), -1.0)
            if ops >= 0:
                return ops
    return 0.0


# League-average OPS baselines (approximate 2024/2025 actuals)
_LG_OPS_VS_LHP = 0.718
_LG_OPS_VS_RHP = 0.710


def _ops_to_wrc_proxy(ops: float, lg_ops: float) -> float:
    """Convert OPS to a wRC+-like index (100 = league average)."""
    if lg_ops <= 0 or ops <= 0:
        return 100.0
    return round(100.0 * ops / lg_ops, 1)


def get_platoon_splits() -> dict[str, dict]:
    """
    Fetch team batting splits vs LHP and vs RHP from MLB Stats API.

    Returns:
        {team_abbrev: {
            wrc_vs_lhp: float,             # OPS-based wRC+ proxy vs LHP
            wrc_vs_rhp: float,             # OPS-based wRC+ proxy vs RHP
            platoon_advantage_lhp: float,  # points above 100 vs LHP
            platoon_advantage_rhp: float,  # points above 100 vs RHP
        }}
    """
    teams = _get_all_teams()
    if not teams:
        logger.warning("platoon_splits_collector: could not retrieve team list")
        return {}

    result: dict[str, dict] = {}

    for team_id, abbrev in teams.items():
        ops_vs_lhp = _get_team_ops_vs(team_id, "vl")
        ops_vs_rhp = _get_team_ops_vs(team_id, "vr")

        if ops_vs_lhp == 0.0 and ops_vs_rhp == 0.0:
            continue

        wrc_vs_lhp = _ops_to_wrc_proxy(ops_vs_lhp, _LG_OPS_VS_LHP) if ops_vs_lhp > 0 else 100.0
        wrc_vs_rhp = _ops_to_wrc_proxy(ops_vs_rhp, _LG_OPS_VS_RHP) if ops_vs_rhp > 0 else 100.0

        result[abbrev] = {
            "wrc_vs_lhp":            wrc_vs_lhp,
            "wrc_vs_rhp":            wrc_vs_rhp,
            "platoon_advantage_lhp": round(wrc_vs_lhp - 100.0, 1),
            "platoon_advantage_rhp": round(wrc_vs_rhp - 100.0, 1),
        }

    if not result:
        logger.warning("platoon_splits_collector: no splits data returned")
    else:
        logger.info("platoon_splits_collector: fetched %d teams", len(result))

    return result
