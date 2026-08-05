"""
Lineup Monitor — Tier 3, Layer 24.
Scrapes confirmed lineups from RotoWire and/or MLB Stats API.
Scores lineup completeness (0-10) to detect day-of scratches or TBD slots.
"""
from __future__ import annotations
import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

import os as _os
# Blocked-site scrapers (RotoWire/Covers/InsideThePen/Savant) are unreachable
# from Streamlit Cloud IPs and just burn the startup timeout before we fall back
# to the MLB StatsAPI anyway. Skip them by default; the MLB API path returns real
# data fast. Set USE_BLOCKED_SCRAPERS=1 to re-enable (e.g. running locally).
def _use_blocked_scrapers() -> bool:
    return _os.getenv("USE_BLOCKED_SCRAPERS", "").lower() in ("1", "true", "yes")


_ROTOWIRE_URL = "https://www.rotowire.com/baseball/daily-lineups.php"
_MLB_SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&date={date}&hydrate=lineups"
)

_TEAM_NAME_MAP: dict[str, str] = {
    # RotoWire full → abbreviation
    "Angels": "LAA", "Astros": "HOU", "Athletics": "OAK", "Blue Jays": "TOR",
    "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
    "D-backs": "ARI", "Diamondbacks": "ARI", "Dodgers": "LAD", "Giants": "SF",
    "Guardians": "CLE", "Indians": "CLE", "Mariners": "SEA", "Marlins": "MIA",
    "Mets": "NYM", "Nationals": "WSH", "Orioles": "BAL", "Padres": "SD",
    "Phillies": "PHI", "Pirates": "PIT", "Rangers": "TEX", "Rays": "TB",
    "Red Sox": "BOS", "Reds": "CIN", "Rockies": "COL", "Royals": "KC",
    "Tigers": "DET", "Twins": "MIN", "White Sox": "CWS", "Yankees": "NYY",
}


def _abbrev(name: str) -> str:
    n = name.strip()
    if n in _TEAM_NAME_MAP:
        return _TEAM_NAME_MAP[n]
    if len(n) <= 3:
        return n.upper()
    for full, abbr in _TEAM_NAME_MAP.items():
        if full.lower() in n.lower() or n.lower() in full.lower():
            return abbr
    return n[:3].upper()


def _score_lineup(batters: list[str]) -> dict:
    """
    Given an ordered list of batter strings, compute value_score and counts.
    Lineup slot indices: 0=leadoff, 2=3-hole, 3=cleanup.
    """
    total_slots = 9
    tbd_count = sum(1 for b in batters if b.strip().upper() in ("TBD", "", "?"))
    missing_spots = max(0, total_slots - len(batters))
    confirmed = len([b for b in batters if b.strip().upper() not in ("TBD", "", "?")])
    lineup_confirmed = (confirmed >= 9 and tbd_count == 0 and missing_spots == 0)

    score = 5.0
    # +2 if cleanup hitter (slot 3 or 4, 0-indexed 2 or 3) is confirmed
    cleanup_confirmed = False
    if len(batters) >= 4:
        cleanup = batters[2] if len(batters) > 2 else ""
        if cleanup.strip().upper() not in ("TBD", "", "?"):
            cleanup_confirmed = True
    if cleanup_confirmed:
        score += 2.0

    # +2 if leadoff (slot 1) is confirmed
    leadoff_confirmed = False
    if batters and batters[0].strip().upper() not in ("TBD", "", "?"):
        leadoff_confirmed = True
    if leadoff_confirmed:
        score += 2.0

    # -3 if 2+ TBD slots
    effective_tbd = tbd_count + missing_spots
    if effective_tbd >= 2:
        score -= 3.0

    return {
        "lineup_confirmed": lineup_confirmed,
        "value_score":      max(0.0, min(10.0, round(score, 1))),
        "missing_spots":    missing_spots,
        "tbd_count":        tbd_count,
        "batters":          batters,
    }


def _fetch_rotowire(date_str: str) -> dict[str, dict]:
    """
    Scrape RotoWire daily-lineups page.
    Returns {team_abbrev: {lineup_confirmed, value_score, missing_spots, tbd_count}}.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("lineup_monitor: BeautifulSoup not installed; skipping RotoWire scrape")
        return {}

    result: dict[str, dict] = {}
    try:
        resp = requests.get(
            _ROTOWIRE_URL,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # RotoWire wraps each team lineup in a div with class "lineup__box"
        for box in soup.select(".lineup__box"):
            # Team name from header
            team_header = box.select_one(".lineup__team-name, .lineup__team")
            if not team_header:
                continue
            team_name = team_header.get_text(strip=True)
            abbr = _abbrev(team_name)

            # Batter rows: class "lineup__player"
            batter_els = box.select(".lineup__player")
            batters: list[str] = []
            for el in batter_els:
                name = el.get_text(strip=True)
                if name:
                    batters.append(name)

            # If we found at least one batter, score the lineup
            if batters or box.select(".lineup__confirmed"):
                info = _score_lineup(batters)
                # RotoWire confirms lineups with a specific badge
                confirmed_badge = box.select_one(".lineup__status-confirmed, .lineup-confirmed")
                if confirmed_badge:
                    info["lineup_confirmed"] = True
                result[abbr] = info

    except Exception as exc:
        logger.warning("lineup_monitor: RotoWire scrape failed: %s", exc)

    return result


def _fetch_mlb_api(date_str: str) -> dict[str, dict]:
    """
    MLB Stats API hydrate=lineups endpoint is not reachable from cloud hosting.
    RotoWire is the primary lineup source; this path is disabled to avoid
    an 8-second connect timeout on every boot.
    """
    return {}


def get_lineups(date_str: str | None = None) -> dict[str, dict]:
    """
    Fetch day-of lineups from RotoWire (primary) and MLB Stats API (secondary).
    Merges results, preferring confirmed data.

    Returns:
        {team_abbrev: {lineup_confirmed: bool, value_score: float,
                       missing_spots: int, tbd_count: int}}
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # MLB StatsAPI is primary (real, cloud-reachable). RotoWire only when explicitly enabled.
    roto_data = _fetch_rotowire(date_str) if _use_blocked_scrapers() else {}
    mlb_data  = _fetch_mlb_api(date_str)

    # Merge: prefer MLB API confirmed lineups; fill gaps with RotoWire
    merged: dict[str, dict] = {}
    all_teams = set(roto_data) | set(mlb_data)
    for abbr in all_teams:
        mlb_entry  = mlb_data.get(abbr, {})
        roto_entry = roto_data.get(abbr, {})

        # If MLB API has a confirmed lineup, prefer it
        if mlb_entry.get("lineup_confirmed"):
            merged[abbr] = mlb_entry
        elif roto_entry:
            merged[abbr] = roto_entry
        else:
            merged[abbr] = mlb_entry

    if not merged:
        logger.info("lineup_monitor: no lineup data available for %s", date_str)
    else:
        confirmed_count = sum(1 for v in merged.values() if v.get("lineup_confirmed"))
        logger.info(
            "lineup_monitor: %d teams fetched, %d confirmed lineups",
            len(merged), confirmed_count,
        )

    return merged
