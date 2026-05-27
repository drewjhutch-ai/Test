"""
Velocity & spin drop tracker.
MLB Stats API has no pitch velocity data, so this module returns zero-filled
entries keyed by pitcher name. Layer 14 handles velocity_season == 0 gracefully
with a "data unavailable" note.
Cached module-level with ~6-hour TTL.
"""
from __future__ import annotations
import logging
import time

import requests

logger = logging.getLogger(__name__)

_CACHE: dict = {}
_CACHE_TS: float = 0.0
_TTL: float = 6 * 3600

_URL = (
    "https://statsapi.mlb.com/api/v1/stats"
    "?stats=season&group=pitching&gameType=R&season=2026"
    "&playerPool=ALL&limit=500&sportId=1"
)
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def get_velocity_data() -> dict[str, dict]:
    """
    Returns dict keyed by pitcher full name and last name.
    All velocity/spin fields are 0.0 (unavailable from MLB Stats API).
    Layer 14 handles velocity_season == 0 with a neutral "unavailable" note.
    """
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE and (now - _CACHE_TS) < _TTL:
        return _CACHE

    try:
        resp = requests.get(_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        splits = resp.json()["stats"][0]["splits"]
    except Exception as exc:
        logger.warning("velocity_tracker: fetch failed: %s", exc)
        return _CACHE

    result: dict[str, dict] = {}
    _ZERO = {
        "season_velo":   0.0,
        "recent_velo":   0.0,
        "velo_drop":     0.0,
        "season_spin":   0.0,
        "recent_spin":   0.0,
        "spin_drop_pct": 0.0,
    }

    for sp in splits:
        full = sp.get("player", {}).get("fullName", "").strip()
        if not full:
            continue
        ip = _safe_float(sp.get("stat", {}).get("inningsPitched"), 0)
        if ip < 5:
            continue
        result[full] = _ZERO
        last = full.split()[-1]
        if last not in result:
            result[last] = _ZERO

    if result:
        _CACHE = result
        _CACHE_TS = now
        logger.info(
            "velocity_tracker: loaded %d pitcher entries (velocity unavailable — MLB Stats API; zeros set)",
            len(result),
        )
    else:
        logger.warning("velocity_tracker: no rows loaded; returning stale or empty cache")

    return _CACHE
