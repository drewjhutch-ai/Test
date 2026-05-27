"""
Umpire tendencies collector.
Today's umpire assignments from MLB Stats API (fast, reliable).
Historical O/U tendencies from Covers.com (best-effort; skipped if blocked).
Cached module-level with ~6-hour TTL.
"""
from __future__ import annotations
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

_COVERS_CACHE: dict[str, dict] = {}
_COVERS_TS: float = 0.0

_TODAY_CACHE: dict[str, dict] = {}
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
#  Covers scrape (best-effort, 8s hard cap)                           #
# ------------------------------------------------------------------ #

def _scrape_covers() -> dict[str, dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {}

    try:
        resp = requests.get(_COVERS_URL, headers=_HEADERS, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        logger.debug("umpire_collector: Covers scrape failed: %s", exc)
        return {}

    result: dict[str, dict] = {}
    try:
        table = soup.find("table")
        if not table:
            return {}
        for row in table.find_all("tr")[1:]:
            cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if len(cols) < 4:
                continue
            name = cols[0].strip()
            if not name or name.lower() in ("umpire", "name"):
                continue
            nums = []
            for c in cols[1:]:
                try:
                    nums.append(float(c.replace("%", "").strip()))
                except (ValueError, TypeError):
                    pass
            if len(nums) < 3:
                continue
            total_games = int(nums[0]) if nums[0] > 1 else 0
            over_pct    = nums[1] / 100 if nums[1] > 1 else nums[1]
            under_pct   = nums[2] / 100 if nums[2] > 1 else nums[2]
            result[name.lower()] = {
                "umpire_name": name,
                "over_pct":    round(over_pct, 3),
                "under_pct":   round(under_pct, 3),
                "total_games": total_games,
            }
    except Exception:
        return {}

    return result


def _get_covers_cached() -> dict[str, dict]:
    """Return Covers cache, refreshing in background if stale. Never blocks > 8s."""
    global _COVERS_CACHE, _COVERS_TS
    now = time.time()
    if _COVERS_CACHE and (now - _COVERS_TS) < _TTL:
        return _COVERS_CACHE

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_scrape_covers)
            data = fut.result(timeout=8)
        if data:
            _COVERS_CACHE = data
            _COVERS_TS = now
            logger.info("umpire_collector: loaded tendencies for %d umpires", len(data))
    except FuturesTimeout:
        logger.debug("umpire_collector: Covers scrape timed out — using stale/empty cache")
    except Exception as exc:
        logger.debug("umpire_collector: Covers error: %s", exc)

    return _COVERS_CACHE


# ------------------------------------------------------------------ #
#  Today's umpire assignments (MLB Stats API — always fast)           #
# ------------------------------------------------------------------ #

def get_todays_umpires(date_str: str | None = None) -> dict[str, dict]:
    """
    Returns {game_id: {umpire_name, over_pct, under_pct, under_lean}}.
    MLB schedule fetch runs first (reliable). Covers O/U data is enrichment
    only — umpire names are returned even when Covers is unavailable.
    """
    global _TODAY_CACHE, _TODAY_TS
    now = time.time()
    if _TODAY_CACHE and (now - _TODAY_TS) < _TTL:
        return _TODAY_CACHE

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Step 1: fetch today's assignments from MLB Stats API (fast, no block risk)
    try:
        url  = _SCHEDULE_URL.format(date=date_str)
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        schedule = resp.json()
    except Exception as exc:
        logger.warning("umpire_collector: MLB schedule fetch failed: %s", exc)
        return _TODAY_CACHE

    # Build umpire-name map first so Covers enrichment is optional
    assignments: dict[str, str] = {}  # game_id → hp_ump_name
    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            game_id  = str(game.get("gamePk", ""))
            officials = game.get("officials", [])
            hp_ump = None
            for off in officials:
                if "home plate" in off.get("officialType", "").lower():
                    hp_ump = off.get("official", {}).get("fullName", "").strip()
                    break
            if not hp_ump and officials:
                hp_ump = officials[0].get("official", {}).get("fullName", "").strip()
            if hp_ump and game_id:
                assignments[game_id] = hp_ump

    if not assignments:
        logger.warning("umpire_collector: no umpire assignments for %s", date_str)
        return _TODAY_CACHE

    # Step 2: try to enrich with Covers historical O/U (best-effort, 8s cap)
    tendencies = _get_covers_cached()

    result: dict[str, dict] = {}
    for game_id, hp_ump in assignments.items():
        tend      = tendencies.get(hp_ump.lower(), {})
        over_pct  = tend.get("over_pct",  0.50)
        under_pct = tend.get("under_pct", 0.50)
        result[game_id] = {
            "umpire_name": hp_ump,
            "over_pct":    over_pct,
            "under_pct":   under_pct,
            "under_lean":  under_pct >= 0.58,
            "over_lean":   over_pct  >= 0.58,
        }

    _TODAY_CACHE = result
    _TODAY_TS    = now
    logger.info("umpire_collector: assigned umpires to %d games", len(result))
    return _TODAY_CACHE


def get_umpire_tendencies() -> dict[str, dict]:
    """Public alias for Covers cache (used by other modules)."""
    return _get_covers_cached()
