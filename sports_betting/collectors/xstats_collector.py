"""
xStats collector — fetches pitcher xERA/xFIP/SIERA and team xwOBA from MLB Stats API.
Also provides CSW% and Stuff+ proxies derived from K%, BB%, and strike%.
Cached module-level with ~6-hour TTL.
"""
from __future__ import annotations
import logging
import math
import time

import requests

logger = logging.getLogger(__name__)

_PITCHER_CACHE: dict = {}
_PITCHER_CACHE_TS: float = 0.0

_BATTER_CACHE: dict = {}
_BATTER_CACHE_TS: float = 0.0

_CSW_CACHE: dict = {}
_CSW_CACHE_TS: float = 0.0

_STUFF_CACHE: dict = {}
_STUFF_CACHE_TS: float = 0.0

# Shared pitcher splits cache (avoids re-fetching for each function)
_SPLITS_CACHE: list = []
_SPLITS_CACHE_TS: float = 0.0

_TTL: float = 6 * 3600  # 6 hours

_PITCHER_URL = (
    "https://statsapi.mlb.com/api/v1/stats"
    "?stats=season&group=pitching&gameType=R&season=2026"
    "&playerPool=ALL&limit=500&sportId=1"
)
_BATTER_URL = (
    "https://statsapi.mlb.com/api/v1/stats"
    "?stats=season&group=hitting&gameType=R&season=2026"
    "&playerPool=ALL&limit=1000&sportId=1"
)

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def _parse_ip(ip_str) -> float:
    """Convert innings pitched string like '34.1' to decimal innings."""
    try:
        s = str(ip_str)
        if "." in s:
            whole, frac = s.split(".", 1)
            return float(whole) + int(frac) / 3.0
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _fetch_pitcher_splits() -> list:
    """Fetch pitcher season splits from MLB Stats API. Cached 6 hours."""
    global _SPLITS_CACHE, _SPLITS_CACHE_TS
    now = time.time()
    if _SPLITS_CACHE and (now - _SPLITS_CACHE_TS) < _TTL:
        return _SPLITS_CACHE

    try:
        resp = requests.get(_PITCHER_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        splits = data["stats"][0]["splits"]
        _SPLITS_CACHE = splits
        _SPLITS_CACHE_TS = now
        logger.info("xstats_collector: fetched %d pitcher splits", len(splits))
        return splits
    except Exception as exc:
        logger.warning("xstats_collector: pitcher splits fetch failed: %s", exc)
        return _SPLITS_CACHE


def _fetch_batter_splits() -> list:
    """Fetch batter season splits from MLB Stats API."""
    try:
        resp = requests.get(_BATTER_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data["stats"][0]["splits"]
    except Exception as exc:
        logger.warning("xstats_collector: batter splits fetch failed: %s", exc)
        return []


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def get_pitcher_xstats() -> dict[str, dict]:
    """
    Returns dict keyed by pitcher full name and last name.
    Each value: {xera, xfip, siera, era}

    Proxies:
      xera  = ERA (actual from API)
      xfip  = FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + 3.10, clamped [1.0, 9.0]
      siera = ERA (no groundball data available)
    Only includes pitchers with IP >= 5.
    """
    global _PITCHER_CACHE, _PITCHER_CACHE_TS
    now = time.time()
    if _PITCHER_CACHE and (now - _PITCHER_CACHE_TS) < _TTL:
        return _PITCHER_CACHE

    splits = _fetch_pitcher_splits()
    result: dict[str, dict] = {}

    for split in splits:
        player = split.get("player", {})
        stat = split.get("stat", {})
        full_name = player.get("fullName", "").strip()
        if not full_name:
            continue

        bf = _safe_float(stat.get("battersFaced"), 0.0)
        if bf < 1:
            continue

        ip = _parse_ip(stat.get("inningsPitched", "0"))
        if ip < 5:
            continue

        era = _safe_float(stat.get("era"), 4.50)
        hr  = _safe_float(stat.get("homeRuns"), 0.0)
        bb  = _safe_float(stat.get("baseOnBalls"), 0.0)
        hbp = _safe_float(stat.get("hitByPitch"), 0.0)
        k   = _safe_float(stat.get("strikeOuts"), 0.0)

        if ip >= 5:
            fip = (13 * hr + 3 * (bb + hbp) - 2 * k) / ip + 3.10
            fip = max(1.0, min(9.0, fip))
        else:
            fip = 4.50

        entry = {
            "xera":  round(era, 2),
            "xfip":  round(fip, 2),
            "siera": round(era, 2),
            "era":   round(era, 2),
        }
        result[full_name] = entry
        last = full_name.split()[-1]
        if last not in result:
            result[last] = entry

    if result:
        _PITCHER_CACHE = result
        _PITCHER_CACHE_TS = now
        logger.info("xstats_collector: loaded %d pitcher xstat entries", len(splits))
    else:
        logger.warning("xstats_collector: no pitcher rows fetched; returning stale or empty cache")

    return _PITCHER_CACHE


def get_team_xwoba() -> dict[str, dict]:
    """
    Returns dict keyed by team abbreviation.
    Each value: {xwoba, woba}
    xwoba = team avg OPS * 0.38 (rough wOBA conversion proxy).
    """
    global _BATTER_CACHE, _BATTER_CACHE_TS
    now = time.time()
    if _BATTER_CACHE and (now - _BATTER_CACHE_TS) < _TTL:
        return _BATTER_CACHE

    splits = _fetch_batter_splits()

    team_accum: dict[str, dict] = {}
    for split in splits:
        team = split.get("team", {}).get("abbreviation", "").strip().upper()
        stat = split.get("stat", {})
        if not team:
            continue
        ops = _safe_float(stat.get("ops"), 0.0)
        obp = _safe_float(stat.get("obp"), 0.0)
        if ops == 0.0 and obp == 0.0:
            continue
        if team not in team_accum:
            team_accum[team] = {"ops_sum": 0.0, "obp_sum": 0.0, "count": 0}
        team_accum[team]["ops_sum"] += ops
        team_accum[team]["obp_sum"] += obp
        team_accum[team]["count"]   += 1

    result: dict[str, dict] = {}
    for team, acc in team_accum.items():
        n = max(1, acc["count"])
        avg_ops = acc["ops_sum"] / n
        avg_obp = acc["obp_sum"] / n
        xwoba = round(avg_ops * 0.38, 3)
        woba  = round(avg_obp, 3)
        result[team] = {"xwoba": xwoba, "woba": woba}

    if result:
        _BATTER_CACHE = result
        _BATTER_CACHE_TS = now
        logger.info("xstats_collector: loaded xwOBA for %d teams", len(result))
    else:
        logger.warning("xstats_collector: no batter rows fetched; returning stale or empty cache")

    return _BATTER_CACHE


def get_pitcher_csw() -> dict[str, float]:
    """
    CSW% proxy from MLB Stats API.
    Uses strikePercentage / 100 as CSW proxy.
    Returns dict keyed by pitcher name → CSW rate (0.0–1.0 float).
    Cached 6-hour TTL.
    """
    global _CSW_CACHE, _CSW_CACHE_TS
    now = time.time()
    if _CSW_CACHE and (now - _CSW_CACHE_TS) < _TTL:
        return _CSW_CACHE

    splits = _fetch_pitcher_splits()
    result: dict[str, float] = {}

    for split in splits:
        player = split.get("player", {})
        stat   = split.get("stat", {})
        full_name = player.get("fullName", "").strip()
        if not full_name:
            continue

        ip = _parse_ip(stat.get("inningsPitched", "0"))
        if ip < 5:
            continue

        bf = _safe_float(stat.get("battersFaced"), 0.0)
        if bf < 1:
            continue

        strike_pct_raw = _safe_float(stat.get("strikePercentage"), -1.0)
        if strike_pct_raw < 0:
            continue

        # strikePercentage may be stored as 0–100 or 0.0–1.0
        csw_rate = strike_pct_raw / 100.0 if strike_pct_raw > 1.0 else strike_pct_raw
        result[full_name] = round(csw_rate, 4)
        last = full_name.split()[-1]
        if last not in result:
            result[last] = round(csw_rate, 4)

    if result:
        _CSW_CACHE = result
        _CSW_CACHE_TS = now
        logger.info("get_pitcher_csw: loaded CSW proxy for %d pitchers", len(splits))
    else:
        logger.warning("get_pitcher_csw: no rows parsed; returning stale or empty cache")

    return _CSW_CACHE


def get_stuff_plus() -> dict[str, float]:
    """
    Stuff+ proxy from MLB Stats API.
    Normalize K%-BB% to 100=avg, 10 points per standard deviation above avg.
    Returns dict keyed by pitcher name → float (100 = average).
    Cached 6-hour TTL.
    """
    global _STUFF_CACHE, _STUFF_CACHE_TS
    now = time.time()
    if _STUFF_CACHE and (now - _STUFF_CACHE_TS) < _TTL:
        return _STUFF_CACHE

    splits = _fetch_pitcher_splits()

    # First pass: collect K%-BB% values
    raw_vals: list[tuple[str, str, float]] = []
    for split in splits:
        player = split.get("player", {})
        stat   = split.get("stat", {})
        full_name = player.get("fullName", "").strip()
        if not full_name:
            continue

        ip = _parse_ip(stat.get("inningsPitched", "0"))
        if ip < 5:
            continue

        bf  = _safe_float(stat.get("battersFaced"), 0.0)
        if bf < 1:
            continue

        k   = _safe_float(stat.get("strikeOuts"), 0.0)
        bb  = _safe_float(stat.get("baseOnBalls"), 0.0)
        k_pct  = k  / bf if bf > 0 else 0.0
        bb_pct = bb / bf if bf > 0 else 0.0
        k_minus_bb = k_pct - bb_pct
        last = full_name.split()[-1]
        raw_vals.append((full_name, last, k_minus_bb))

    result: dict[str, float] = {}
    if raw_vals:
        values = [v for _, _, v in raw_vals]
        avg_val = sum(values) / len(values)
        # Compute std deviation
        variance = sum((v - avg_val) ** 2 for v in values) / max(1, len(values))
        std_val = math.sqrt(variance) if variance > 0 else 0.05

        for full_name, last, k_minus_bb in raw_vals:
            z = (k_minus_bb - avg_val) / std_val if std_val > 0 else 0.0
            stuff = round(100.0 + z * 10.0, 1)
            result[full_name] = stuff
            if last not in result:
                result[last] = stuff

    if result:
        _STUFF_CACHE = result
        _STUFF_CACHE_TS = now
        logger.info("get_stuff_plus: loaded Stuff+ proxy for %d pitchers", len(splits))
    else:
        logger.warning("get_stuff_plus: no rows parsed; returning stale or empty cache")

    return _STUFF_CACHE
