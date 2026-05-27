"""
Platoon Splits Collector — Tier 3, Layer 23.
Fetches batter vs LHP / vs RHP splits from FanGraphs splits leaderboard.
Aggregates to team level: avg wRC+ vs LHP and vs RHP.
"""
from __future__ import annotations
import io
import logging

import requests

logger = logging.getLogger(__name__)

_SPLITS_VS_LHP_URL = (
    "https://www.fangraphs.com/api/leaders/splits/data"
    "?strsplits=vl&splitTeams=false&season=2026&player_type=batter"
    "&autoPt=true&sort=wOBAAgainst,d&pageitems=1000&pagenum=1&csv=true"
)

_SPLITS_VS_RHP_URL = (
    "https://www.fangraphs.com/api/leaders/splits/data"
    "?strsplits=vr&splitTeams=false&season=2026&player_type=batter"
    "&autoPt=true&sort=wOBAAgainst,d&pageitems=1000&pagenum=1&csv=true"
)

_FG_TEAM_MAP: dict[str, str] = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "OAK", "Blue Jays": "TOR",
    "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
    "D-backs": "ARI", "Diamondbacks": "ARI", "Dodgers": "LAD", "Giants": "SF",
    "Guardians": "CLE", "Indians": "CLE", "Mariners": "SEA", "Marlins": "MIA",
    "Mets": "NYM", "Nationals": "WSH", "Orioles": "BAL", "Padres": "SD",
    "Phillies": "PHI", "Pirates": "PIT", "Rangers": "TEX", "Rays": "TB",
    "Red Sox": "BOS", "Reds": "CIN", "Rockies": "COL", "Royals": "KC",
    "Tigers": "DET", "Twins": "MIN", "White Sox": "CWS", "Yankees": "NYY",
}


def _abbrev(team_raw: str) -> str:
    t = team_raw.strip()
    if t in _FG_TEAM_MAP:
        return _FG_TEAM_MAP[t]
    if len(t) <= 3:
        return t.upper()
    for name, abbr in _FG_TEAM_MAP.items():
        if name.lower() in t.lower() or t.lower() in name.lower():
            return abbr
    return t[:3].upper()


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(str(val).strip().replace("%", "").replace(",", ""))
    except (ValueError, AttributeError):
        return default


def _fetch_splits_csv(url: str) -> list[dict]:
    """Fetch a FanGraphs splits CSV and return list-of-dicts."""
    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        import csv
        reader = csv.DictReader(io.StringIO(resp.text))
        return [row for row in reader]
    except Exception as exc:
        logger.warning("platoon_splits_collector: fetch failed (%s): %s", url[:60], exc)
        return []


def _aggregate_team_wrc(rows: list[dict]) -> dict[str, list[float]]:
    """
    Aggregate individual batter wRC+ values by team.
    Returns {team_abbrev: [wrc_val, ...]}
    """
    team_wrc: dict[str, list[float]] = {}
    for row in rows:
        team_raw = row.get("Team", row.get("team", ""))
        if not team_raw:
            continue
        abbr = _abbrev(team_raw)
        # wRC+ column header variants
        wrc_raw = ""
        for col in ("wRC+", "wRC", "wrc+", "wrc"):
            wrc_raw = row.get(col, "")
            if wrc_raw:
                break
        wrc = _safe_float(wrc_raw)
        if wrc > 0:
            team_wrc.setdefault(abbr, []).append(wrc)
    return team_wrc


def get_platoon_splits() -> dict[str, dict]:
    """
    Fetch batter splits vs LHP and vs RHP from FanGraphs, aggregated by team.

    Returns:
        {team_abbrev: {
            wrc_vs_lhp: float,      # team avg wRC+ vs left-handed pitchers
            wrc_vs_rhp: float,      # team avg wRC+ vs right-handed pitchers
            platoon_advantage_lhp: float,  # points above 100 vs LHP
            platoon_advantage_rhp: float,  # points above 100 vs RHP
        }}
    """
    lhp_rows = _fetch_splits_csv(_SPLITS_VS_LHP_URL)
    rhp_rows = _fetch_splits_csv(_SPLITS_VS_RHP_URL)

    lhp_by_team = _aggregate_team_wrc(lhp_rows)
    rhp_by_team = _aggregate_team_wrc(rhp_rows)

    all_teams = set(lhp_by_team) | set(rhp_by_team)
    result: dict[str, dict] = {}

    for abbr in all_teams:
        lhp_vals = lhp_by_team.get(abbr, [])
        rhp_vals = rhp_by_team.get(abbr, [])

        wrc_vs_lhp = round(sum(lhp_vals) / len(lhp_vals), 1) if lhp_vals else 100.0
        wrc_vs_rhp = round(sum(rhp_vals) / len(rhp_vals), 1) if rhp_vals else 100.0

        result[abbr] = {
            "wrc_vs_lhp":          wrc_vs_lhp,
            "wrc_vs_rhp":          wrc_vs_rhp,
            "platoon_advantage_lhp": round(wrc_vs_lhp - 100.0, 1),
            "platoon_advantage_rhp": round(wrc_vs_rhp - 100.0, 1),
        }

    if not result:
        logger.info("platoon_splits_collector: no data returned (FanGraphs unavailable)")
    else:
        logger.info("platoon_splits_collector: fetched %d teams", len(result))

    return result
