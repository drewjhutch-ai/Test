"""
Luck Metrics Collector — Tier 3, Layer 22.
Fetches team BABIP (batting) and pitcher LOB% (strand rate) from FanGraphs.
Returns a luck_score for each team indicating whether recent results are
luck-inflated (+) or luck-deflated (-).
"""
from __future__ import annotations
import logging
import io

import requests

logger = logging.getLogger(__name__)

_BATTING_URL = (
    "https://www.fangraphs.com/api/leaders/major-league/data"
    "?age=&pos=all&stats=bat&lg=all&qual=0&season=2026&season1=2026"
    "&ind=0&team=0,ts&pageitems=50&pagenum=1&type=0&sortcol=11"
    "&sortdir=desc&rosters=false&players=0&month=0"
    "&startdate=2026-01-01&enddate=2026-12-31&csv=true"
)

_PITCHING_URL = (
    "https://www.fangraphs.com/api/leaders/major-league/data"
    "?age=&pos=all&stats=pit&lg=all&qual=0&season=2026&season1=2026"
    "&ind=0&team=0,ts&pageitems=50&pagenum=1&type=0&sortcol=11"
    "&sortdir=desc&rosters=false&players=0&month=0"
    "&startdate=2026-01-01&enddate=2026-12-31&csv=true"
)

# FanGraphs uses these team name abbreviations in CSV exports
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


def _parse_pct(val: str) -> float:
    """Convert '30.5 %' or '0.305' or '30.5%' to float 0.305."""
    try:
        s = val.strip().replace("%", "").replace(" ", "")
        f = float(s)
        return f / 100 if f > 1.0 else f
    except (ValueError, AttributeError):
        return 0.0


def _fetch_csv(url: str) -> list[dict]:
    """Fetch a FanGraphs CSV endpoint and return list-of-dicts."""
    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        import csv
        reader = csv.DictReader(io.StringIO(resp.text))
        return [row for row in reader]
    except Exception as exc:
        logger.warning("luck_metrics_collector: CSV fetch failed (%s): %s", url[:60], exc)
        return []


def _abbrev(team_raw: str) -> str:
    """Normalise FanGraphs team name → standard 2-3 letter abbrev."""
    t = team_raw.strip()
    if t in _FG_TEAM_MAP:
        return _FG_TEAM_MAP[t]
    # Already an abbreviation?
    if len(t) <= 3:
        return t.upper()
    # Partial match
    for name, abbr in _FG_TEAM_MAP.items():
        if name.lower() in t.lower() or t.lower() in name.lower():
            return abbr
    return t[:3].upper()


def get_luck_metrics() -> dict[str, dict]:
    """
    Fetch team BABIP (batting) and pitcher LOB% (strand rate) from FanGraphs.

    Returns:
        {team_abbrev: {babip, lob_pct, luck_score}}
        luck_score range -4 to +4:
            positive = luck-inflated (expect regression)
            negative = luck-deflated (expect improvement)
    """
    babip_by_team: dict[str, float] = {}
    lob_by_team: dict[str, float] = {}

    # ── Batting CSV — BABIP ───────────────────────────────────────────
    batting_rows = _fetch_csv(_BATTING_URL)
    for row in batting_rows:
        team_raw = row.get("Team", row.get("team", ""))
        if not team_raw:
            continue
        abbr = _abbrev(team_raw)
        # FanGraphs BABIP column header variants
        for col in ("BABIP", "babip"):
            raw = row.get(col, "")
            if raw:
                babip_by_team[abbr] = _parse_pct(raw)
                break

    # ── Pitching CSV — LOB% ───────────────────────────────────────────
    pitching_rows = _fetch_csv(_PITCHING_URL)
    for row in pitching_rows:
        team_raw = row.get("Team", row.get("team", ""))
        if not team_raw:
            continue
        abbr = _abbrev(team_raw)
        # FanGraphs strand-rate column variants
        for col in ("LOB%", "lob%", "LOB", "Strand%", "strand%"):
            raw = row.get(col, "")
            if raw:
                lob_by_team[abbr] = _parse_pct(raw)
                break

    # ── Build output ──────────────────────────────────────────────────
    all_teams = set(babip_by_team) | set(lob_by_team)
    result: dict[str, dict] = {}

    for abbr in all_teams:
        babip = babip_by_team.get(abbr, 0.295)
        lob   = lob_by_team.get(abbr, 0.720)

        luck_score = 0.0
        # Offense: BABIP
        if babip > 0.320:
            luck_score += 2   # Over-performing; expect regression
        elif babip < 0.270:
            luck_score -= 2   # Under-performing; expect recovery

        # Pitching: LOB%
        if lob > 0.78:
            luck_score += 2   # Stranding too many; regression coming
        elif lob < 0.68:
            luck_score -= 2   # Strand rate unsustainably low

        result[abbr] = {
            "babip":      round(babip, 3),
            "lob_pct":    round(lob, 3),
            "luck_score": luck_score,
        }

    if not result:
        logger.info("luck_metrics_collector: no data returned (FanGraphs unavailable)")
    else:
        logger.info("luck_metrics_collector: fetched %d teams", len(result))

    return result
