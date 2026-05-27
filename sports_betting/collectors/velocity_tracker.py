"""
Velocity & spin drop tracker — fetches per-pitcher fastball velocity from Baseball Savant.
Cached module-level with ~6-hour TTL.
"""
from __future__ import annotations
import csv
import io
import logging
import time

import requests
from .savant_headers import SAVANT_HEADERS

logger = logging.getLogger(__name__)

_CACHE: dict = {}
_CACHE_TS: float = 0.0
_TTL: float = 6 * 3600

# Four-seam fastball arsenal stats (has season avg and recent velo columns)
_URL = (
    "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
    "?type=pitcher&pitchType=FF&year=2026&team=&min=10&csv=true"
)

_HEADERS = SAVANT_HEADERS


def _safe_float(val: str | None, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def get_velocity_data() -> dict[str, dict]:
    """
    Returns dict keyed by player full name (and last name for fuzzy match).
    Each value:
      {
        season_velo: float,   # season avg fastball velocity
        recent_velo: float,   # last-7d or recent avg fastball velocity (same as season if no split)
        velo_drop:   float,   # season_velo - recent_velo  (positive = decline)
        season_spin: float,   # season avg spin rate
        recent_spin: float,   # recent spin rate
        spin_drop_pct: float, # (season_spin - recent_spin) / season_spin  (positive = decline)
      }
    """
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE and (now - _CACHE_TS) < _TTL:
        return _CACHE

    try:
        resp = requests.get(_URL, headers=_HEADERS, timeout=12)
        if resp.status_code == 403:
            logger.warning("velocity_tracker: Baseball Savant returned 403")
            return _CACHE  # return stale
        resp.raise_for_status()
        text = resp.content.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        rows = list(csv.DictReader(io.StringIO(text)))
    except Exception as exc:
        logger.warning("velocity_tracker: fetch failed: %s", exc)
        return _CACHE

    result: dict[str, dict] = {}
    for row in rows:
        last  = row.get("last_name", "").strip()
        first = row.get("first_name", "").strip()
        if not last:
            continue
        full = f"{first} {last}".strip() if first else last

        # Savant column names for arsenal stats
        # Typical columns: avg_speed (season), avg_spin (season)
        # There may not be a direct last-7d split in the CSV — if absent we leave velo_drop=0
        season_velo = _safe_float(row.get("avg_speed") or row.get("release_speed"), 0.0)
        # Some exports have "avg_speed_7d" or similar; try a few names
        recent_velo = _safe_float(
            row.get("avg_speed_7d") or row.get("release_speed_recent") or row.get("avg_speed"),
            season_velo,
        )
        season_spin = _safe_float(row.get("avg_spin") or row.get("release_spin_rate"), 0.0)
        recent_spin = _safe_float(
            row.get("avg_spin_7d") or row.get("release_spin_rate_recent") or row.get("avg_spin"),
            season_spin,
        )

        velo_drop = round(season_velo - recent_velo, 2)
        spin_drop_pct = 0.0
        if season_spin > 0:
            spin_drop_pct = round((season_spin - recent_spin) / season_spin, 4)

        entry = {
            "season_velo":    season_velo,
            "recent_velo":    recent_velo,
            "velo_drop":      velo_drop,
            "season_spin":    season_spin,
            "recent_spin":    recent_spin,
            "spin_drop_pct":  spin_drop_pct,
        }
        result[full] = entry
        if last:
            result[last] = entry

    if result:
        _CACHE = result
        _CACHE_TS = now
        logger.info("velocity_tracker: loaded %d pitcher velocity rows", len(rows))
    else:
        logger.warning("velocity_tracker: no rows loaded; returning stale or empty cache")

    return _CACHE
