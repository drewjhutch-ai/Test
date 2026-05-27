"""
Umpire tendencies collector.
Scrapes historical O/U data from Covers and fetches today's umpire assignments
from the MLB Stats API.
Cached module-level with ~6-hour TTL.
"""
from __future__ import annotations
import logging
import re
import time
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

_COVERS_CACHE: dict[str, dict] = {}
_COVERS_TS: float = 0.0

_TODAY_CACHE: dict[str, dict] = {}   # {game_id: {umpire_name, over_pct, under_pct, under_lean}}
_TODAY_TS: float = 0.0

_TTL: float = 6 * 3600

_COVERS_URL = "https://www.covers.com/sport/baseball/mlb/statistics/umpires/overunder/2025"
_SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&date={date}&hydrate=officials"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _safe_float(val: str | None, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(str(val).replace("%", "").strip())
    except (ValueError, TypeError):
        return default


# ------------------------------------------------------------------ #
#  Covers scrape                                                       #
# ------------------------------------------------------------------ #

def _scrape_covers() -> dict[str, dict]:
    """
    Scrape Covers umpire O/U table.
    Returns {umpire_name_lower: {over_pct, under_pct, total_games}}
    Falls back to empty dict on any failure.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("umpire_collector: BeautifulSoup not available")
        return {}

    try:
        resp = requests.get(_COVERS_URL, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        logger.warning("umpire_collector: Covers scrape failed: %s", exc)
        return {}

    result: dict[str, dict] = {}
    try:
        table = soup.find("table")
        if not table:
            logger.warning("umpire_collector: no table found on Covers page")
            return {}
        rows = table.find_all("tr")
        for row in rows[1:]:  # skip header
            cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if len(cols) < 4:
                continue
            # Expected columns: Umpire | Games | Over% | Under% (order may vary)
            # Try to identify columns by content patterns
            name = cols[0].strip()
            if not name or name.lower() in ("umpire", "name"):
                continue
            # Find numeric columns
            nums = []
            for c in cols[1:]:
                try:
                    nums.append(float(c.replace("%", "").strip()))
                except (ValueError, TypeError):
                    pass
            if len(nums) < 3:
                continue
            # nums[0]=games, nums[1]=over_pct, nums[2]=under_pct (typical Covers layout)
            total_games = int(nums[0]) if nums[0] > 1 else 0
            over_pct    = nums[1] / 100 if nums[1] > 1 else nums[1]
            under_pct   = nums[2] / 100 if nums[2] > 1 else nums[2]
            result[name.lower()] = {
                "umpire_name": name,
                "over_pct":    round(over_pct, 3),
                "under_pct":   round(under_pct, 3),
                "total_games": total_games,
            }
    except Exception as exc:
        logger.warning("umpire_collector: parse error: %s", exc)
        return {}

    return result


def get_umpire_tendencies() -> dict[str, dict]:
    """
    Returns {umpire_name_lower: {umpire_name, over_pct, under_pct, total_games}}.
    Refreshes every 6 hours.
    """
    global _COVERS_CACHE, _COVERS_TS
    now = time.time()
    if _COVERS_CACHE and (now - _COVERS_TS) < _TTL:
        return _COVERS_CACHE

    data = _scrape_covers()
    if data:
        _COVERS_CACHE = data
        _COVERS_TS = now
        logger.info("umpire_collector: loaded tendencies for %d umpires", len(data))
    else:
        logger.warning("umpire_collector: empty scrape result; using stale cache")

    return _COVERS_CACHE


# ------------------------------------------------------------------ #
#  Today's umpire assignments                                          #
# ------------------------------------------------------------------ #

def get_todays_umpires(date_str: str | None = None) -> dict[str, dict]:
    """
    Returns {game_id: {umpire_name, over_pct, under_pct, under_lean}}.
    under_lean: True if under_pct >= 0.58 (tight zone).
    Refreshes every 6 hours.
    """
    global _TODAY_CACHE, _TODAY_TS
    now = time.time()
    if _TODAY_CACHE and (now - _TODAY_TS) < _TTL:
        return _TODAY_CACHE

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    tendencies = get_umpire_tendencies()

    try:
        url = _SCHEDULE_URL.format(date=date_str)
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        schedule = resp.json()
    except Exception as exc:
        logger.warning("umpire_collector: MLB schedule fetch failed: %s", exc)
        return _TODAY_CACHE

    result: dict[str, dict] = {}
    dates = schedule.get("dates", [])
    for day in dates:
        for game in day.get("games", []):
            game_id  = str(game.get("gamePk", ""))
            officials = game.get("officials", [])
            hp_ump = None
            for off in officials:
                off_type = off.get("officialType", "")
                if "home plate" in off_type.lower() or "hp" in off_type.lower():
                    hp_ump = off.get("official", {}).get("fullName", "").strip()
                    break
            # Fallback: first official listed
            if not hp_ump and officials:
                hp_ump = officials[0].get("official", {}).get("fullName", "").strip()

            if not hp_ump or not game_id:
                continue

            tend = tendencies.get(hp_ump.lower(), {})
            over_pct  = tend.get("over_pct",  0.50)
            under_pct = tend.get("under_pct", 0.50)
            result[game_id] = {
                "umpire_name": hp_ump,
                "over_pct":    over_pct,
                "under_pct":   under_pct,
                "under_lean":  under_pct >= 0.58,
                "over_lean":   over_pct  >= 0.58,
            }

    if result:
        _TODAY_CACHE = result
        _TODAY_TS = now
        logger.info("umpire_collector: assigned umpires to %d games", len(result))
    else:
        logger.warning("umpire_collector: no umpire assignments found for %s", date_str)

    return _TODAY_CACHE
