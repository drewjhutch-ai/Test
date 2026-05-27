"""
Bat Speed / Exit Velocity Trends Collector — Tier 3, Layer 25.
Fetches bat tracking leaderboard and sprint speed from Baseball Savant.
Aggregates to team level.
"""
from __future__ import annotations
import io
import logging

import requests

logger = logging.getLogger(__name__)

_BAT_TRACKING_URL = (
    "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
    "?attackZone=&batSide=&contactType=&count=&dating=2026&gameType="
    "&isHardHit=&minSwings=100&minGroupSwings=1&pitchType=&playerType=batter"
    "&seasonStart=&statcast=&team=&csv=true"
)

_SPRINT_SPEED_URL = (
    "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
    "?min_opp=0&position=&team=&year=2026&csv=true"
)

# Savant uses full team names or abbreviations; normalise to standard abbrevs
_SAVANT_TEAM_MAP: dict[str, str] = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "OAK", "Blue Jays": "TOR",
    "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
    "Diamondbacks": "ARI", "D-backs": "ARI", "Dodgers": "LAD", "Giants": "SF",
    "Guardians": "CLE", "Mariners": "SEA", "Marlins": "MIA",
    "Mets": "NYM", "Nationals": "WSH", "Orioles": "BAL", "Padres": "SD",
    "Phillies": "PHI", "Pirates": "PIT", "Rangers": "TEX", "Rays": "TB",
    "Red Sox": "BOS", "Reds": "CIN", "Rockies": "COL", "Royals": "KC",
    "Tigers": "DET", "Twins": "MIN", "White Sox": "CWS", "Yankees": "NYY",
}


def _abbrev(team_raw: str) -> str:
    t = team_raw.strip()
    if t in _SAVANT_TEAM_MAP:
        return _SAVANT_TEAM_MAP[t]
    if len(t) <= 3:
        return t.upper()
    for name, abbr in _SAVANT_TEAM_MAP.items():
        if name.lower() in t.lower() or t.lower() in name.lower():
            return abbr
    return t[:3].upper()


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(str(val).strip().replace(",", ""))
    except (ValueError, AttributeError):
        return default


def _fetch_savant_csv(url: str) -> list[dict]:
    try:
        resp = requests.get(
            url, timeout=12,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        import csv
        reader = csv.DictReader(io.StringIO(resp.text))
        return [row for row in reader]
    except Exception as exc:
        logger.warning("bat_speed_collector: CSV fetch failed (%s): %s", url[:60], exc)
        return []


def get_bat_speed_metrics() -> dict[str, dict]:
    """
    Fetch bat tracking (bat speed, attack angle) and sprint speed from Baseball Savant.
    Aggregates to team level.

    Returns:
        {team_abbrev: {
            avg_bat_speed: float,       # mph
            avg_attack_angle: float,    # degrees
            avg_sprint_speed: float,    # ft/sec
            is_speed_team: bool,        # avg sprint >= 27.5 ft/sec
        }}
    """
    # ── Bat tracking ─────────────────────────────────────────────────
    bat_rows = _fetch_savant_csv(_BAT_TRACKING_URL)

    team_bat_speeds: dict[str, list[float]] = {}
    team_attack_angles: dict[str, list[float]] = {}

    for row in bat_rows:
        # Column names vary; try common variants
        team_raw = row.get("team_name", row.get("team", row.get("Team", "")))
        if not team_raw:
            continue
        abbr = _abbrev(team_raw)

        # Bat speed column variants
        for col in ("bat_speed", "bat_speed_mph", "avg_bat_speed", "Bat Speed"):
            raw = row.get(col, "")
            if raw:
                val = _safe_float(raw)
                if val > 0:
                    team_bat_speeds.setdefault(abbr, []).append(val)
                break

        # Attack angle column variants
        for col in ("attack_angle", "swing_path_tilt", "Attack Angle"):
            raw = row.get(col, "")
            if raw:
                val = _safe_float(raw)
                if val != 0.0:
                    team_attack_angles.setdefault(abbr, []).append(val)
                break

    # ── Sprint speed ──────────────────────────────────────────────────
    sprint_rows = _fetch_savant_csv(_SPRINT_SPEED_URL)

    team_sprint_speeds: dict[str, list[float]] = {}

    for row in sprint_rows:
        team_raw = row.get("team_id", row.get("team", row.get("Team", "")))
        if not team_raw:
            continue
        abbr = _abbrev(team_raw)

        for col in ("sprint_speed", "hp_to_1b", "Sprint Speed", "r_sprint_speed"):
            raw = row.get(col, "")
            if raw:
                val = _safe_float(raw)
                if val > 0:
                    team_sprint_speeds.setdefault(abbr, []).append(val)
                break

    # ── Aggregate ─────────────────────────────────────────────────────
    all_teams = set(team_bat_speeds) | set(team_sprint_speeds)
    result: dict[str, dict] = {}

    for abbr in all_teams:
        bat_vals    = team_bat_speeds.get(abbr, [])
        angle_vals  = team_attack_angles.get(abbr, [])
        sprint_vals = team_sprint_speeds.get(abbr, [])

        avg_bat_speed    = round(sum(bat_vals) / len(bat_vals), 2)       if bat_vals    else 70.0
        avg_attack_angle = round(sum(angle_vals) / len(angle_vals), 2)   if angle_vals  else 10.0
        avg_sprint_speed = round(sum(sprint_vals) / len(sprint_vals), 2) if sprint_vals else 27.0

        result[abbr] = {
            "avg_bat_speed":    avg_bat_speed,
            "avg_attack_angle": avg_attack_angle,
            "avg_sprint_speed": avg_sprint_speed,
            "is_speed_team":    avg_sprint_speed >= 27.5,
        }

    if not result:
        logger.info("bat_speed_collector: no data returned (Savant unavailable)")
    else:
        logger.info("bat_speed_collector: fetched %d teams", len(result))

    return result
