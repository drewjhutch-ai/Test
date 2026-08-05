"""
Bullpen fatigue collector.
Attempts InsideThePen HTML scrape, then falls back to MLB Stats API recent game logs.
Returns fatigue score 0-10 per team, where 7+ = fatigued (top arms threw 35+ pitches last 3 days).
Cached module-level with ~6-hour TTL.
"""
from __future__ import annotations
import logging
import time

import requests

logger = logging.getLogger(__name__)

_CACHE: dict[str, dict] = {}
_CACHE_TS: float = 0.0
_TTL: float = 6 * 3600

_ITP_URL = "https://insidethepen.com/bullpen-usage.html"
_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
_GAMELOG_URL = (
    "https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
    "?group=pitching&type=gameLog&limit=4"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


# ------------------------------------------------------------------ #
#  InsideThePen scrape (primary)                                       #
# ------------------------------------------------------------------ #

def _scrape_insidethepen() -> dict[str, dict]:
    """
    Scrape InsideThePen bullpen usage table.
    Returns {team_abbrev_upper: {fatigue_score, arms_available, details}}.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("bullpen_fatigue_collector: BeautifulSoup not available")
        return {}

    try:
        resp = requests.get(_ITP_URL, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        logger.warning("bullpen_fatigue_collector: InsideThePen fetch failed: %s", exc)
        return {}

    result: dict[str, dict] = {}
    try:
        table = soup.find("table")
        if not table:
            return {}
        rows = table.find_all("tr")
        for row in rows[1:]:
            cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if len(cols) < 3:
                continue
            team = cols[0].strip().upper()
            if not team or team in ("TEAM", ""):
                continue
            # Try to extract pitches_last_3d and arms counts from available columns
            # InsideThePen layout varies; extract all numeric values
            nums = []
            for c in cols[1:]:
                try:
                    nums.append(float(c.replace("%", "").strip()))
                except (ValueError, TypeError):
                    pass
            if not nums:
                continue
            # Heuristic: first large number = total pitches; last = arms available
            pitches_3d = nums[0] if nums else 0.0
            arms_available = int(nums[-1]) if len(nums) > 1 else 3
            fatigue_score = _compute_fatigue_score(pitches_3d)
            result[team] = {
                "fatigue_score":    fatigue_score,
                "arms_available":   max(0, min(7, arms_available)),
                "pitches_last_3d":  pitches_3d,
                "details":          f"{pitches_3d:.0f} pen pitches last 3 days",
            }
    except Exception as exc:
        logger.warning("bullpen_fatigue_collector: InsideThePen parse error: %s", exc)
        return {}

    return result


def _compute_fatigue_score(pitches_last_3d: float) -> float:
    """
    Scale pitches thrown in last 3 days to a 0-10 fatigue score.
    Reference: top arms throwing 35+ pitches = score 7+.
    """
    # 0 pitches → 0, 50+ pitches → 10
    score = min(10.0, (pitches_last_3d / 50.0) * 10.0)
    return round(score, 2)


# ------------------------------------------------------------------ #
#  MLB Stats API fallback                                              #
# ------------------------------------------------------------------ #

def _fetch_via_mlb_api() -> dict[str, dict]:
    """
    Fallback: use MLB Stats API game log endpoint to estimate bullpen usage.
    Fetches last 4 game pitching logs per team and sums reliever pitches.
    """
    try:
        teams_resp = requests.get(_TEAMS_URL, timeout=12)
        teams_resp.raise_for_status()
        teams = teams_resp.json().get("teams", [])
    except Exception as exc:
        logger.warning("bullpen_fatigue_collector: teams API failed: %s", exc)
        return {}

    result: dict[str, dict] = {}
    for team in teams:
        team_id   = team.get("id")
        team_abbr = team.get("abbreviation", "").upper()
        if not team_id or not team_abbr:
            continue
        try:
            url  = _GAMELOG_URL.format(team_id=team_id)
            resp = requests.get(url, timeout=12)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("bullpen_fatigue_collector: game log fetch failed for %s: %s", team_abbr, exc)
            continue

        total_pen_pitches = 0.0
        arms_seen: set = set()
        splits = data.get("stats", [{}])[0].get("splits", [])
        for split in splits:
            stat = split.get("stat", {})
            # numberOfPitches is for all pitchers; we estimate relievers as total - SP approx
            pitches = _safe_float(stat.get("numberOfPitches"), 0.0)
            ip      = _safe_float(stat.get("inningsPitched"), 0.0)
            # Roughly: if pitcher went < 4 IP in a game they're a reliever
            if ip < 4.0:
                total_pen_pitches += pitches
                player_id = split.get("player", {}).get("id")
                if player_id:
                    arms_seen.add(player_id)

        fatigue_score  = _compute_fatigue_score(total_pen_pitches)
        arms_available = max(0, 7 - len(arms_seen))
        result[team_abbr] = {
            "fatigue_score":   fatigue_score,
            "arms_available":  arms_available,
            "pitches_last_3d": total_pen_pitches,
            "details":         f"~{total_pen_pitches:.0f} reliever pitches last 3-4 games (API est.)",
        }

    return result


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def get_bullpen_fatigue() -> dict[str, dict]:
    """
    Returns {team_abbrev: {fatigue_score, arms_available, pitches_last_3d, details}}.
    fatigue_score 0-10; 7+ = fatigued.
    arms_available: estimated count of fresh arms (< 25 pitches last 3 days).
    Refreshes every 6 hours.
    """
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE and (now - _CACHE_TS) < _TTL:
        return _CACHE

    # MLB StatsAPI is primary (real, cloud-reachable). InsideThePen (blocked)
    # only when explicitly enabled — otherwise it just burns the timeout.
    import os as _os
    _use_scrapers = _os.getenv("USE_BLOCKED_SCRAPERS", "").lower() in ("1", "true", "yes")
    data = _scrape_insidethepen() if _use_scrapers else {}
    if not data:
        data = _fetch_via_mlb_api()

    if data:
        _CACHE = data
        _CACHE_TS = now
        logger.info("bullpen_fatigue_collector: loaded fatigue data for %d teams", len(data))
    else:
        logger.warning("bullpen_fatigue_collector: all sources failed; returning stale or empty cache")

    return _CACHE
