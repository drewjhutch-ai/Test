"""
Luck Metrics Collector — Tier 3, Layer 22.
Fetches team luck indicators from Baseball Savant:
  - Batting luck: team avg (xBA - BA) gap — negative gap = over-performing luck
  - Pitching luck: team avg (xBA_against - BA_against) gap
Returns a luck_score for each team indicating whether recent results are
luck-inflated (+) or luck-deflated (-).
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

# Batter expected stats — xba vs ba gap indicates BABIP luck
_BATTER_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=batter&year=2026&position=&team=&min=25&csv=true"
)
# Pitcher expected stats — xba_against vs ba_against gap indicates strand/BABIP luck
_PITCHER_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=pitcher&year=2026&position=&team=&min=25&csv=true"
)

_HEADERS = SAVANT_HEADERS


def _safe_float(val: str | None, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def _fetch_csv(url: str) -> list[dict]:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        if resp.status_code == 403:
            logger.warning("luck_metrics_collector: Savant returned 403 for %s", url)
            return []
        resp.raise_for_status()
        text = resp.content.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        return list(csv.DictReader(io.StringIO(text)))
    except Exception as exc:
        logger.warning("luck_metrics_collector: CSV fetch failed (%s): %s", url[:60], exc)
        return []


def get_luck_metrics() -> dict[str, dict]:
    """
    Compute team luck scores from Baseball Savant xBA vs BA gaps.

    Returns:
        {team_abbrev: {babip_proxy, xba_gap_batting, xba_gap_pitching, luck_score}}
        luck_score range -4 to +4:
            positive = luck-inflated (expect regression)
            negative = luck-deflated (expect improvement)
    """
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE and (now - _CACHE_TS) < _TTL:
        return _CACHE

    # ── Batting: accumulate (xba - ba) gap per team ───────────────────
    bat_gap: dict[str, list[float]] = {}
    for row in _fetch_csv(_BATTER_URL):
        team = row.get("team_name_abbrev", row.get("team", "")).strip().upper()
        if not team:
            continue
        xba = _safe_float(row.get("xba"), -1.0)
        ba  = _safe_float(row.get("ba"),  -1.0)
        if xba < 0 or ba < 0:
            continue
        # negative gap (xba < ba) means batter is over-performing luck
        bat_gap.setdefault(team, []).append(xba - ba)

    # ── Pitching: accumulate (xba_against - ba_against) gap per team ──
    pit_gap: dict[str, list[float]] = {}
    for row in _fetch_csv(_PITCHER_URL):
        team = row.get("team_name_abbrev", row.get("team", "")).strip().upper()
        if not team:
            continue
        xba = _safe_float(row.get("xba"), -1.0)
        ba  = _safe_float(row.get("ba"),  -1.0)
        if xba < 0 or ba < 0:
            continue
        # negative gap (xba < ba) means pitcher has been unlucky (balls falling in)
        pit_gap.setdefault(team, []).append(xba - ba)

    all_teams = set(bat_gap) | set(pit_gap)
    result: dict[str, dict] = {}

    for team in all_teams:
        b_vals = bat_gap.get(team, [])
        p_vals = pit_gap.get(team, [])

        avg_bat_gap = sum(b_vals) / len(b_vals) if b_vals else 0.0
        avg_pit_gap = sum(p_vals) / len(p_vals) if p_vals else 0.0

        luck_score = 0.0
        # Batting luck: xba < ba (avg_bat_gap < 0) means batters over-performing → +luck
        if avg_bat_gap < -0.015:
            luck_score += 2   # offensive over-performance; expect regression
        elif avg_bat_gap > 0.015:
            luck_score -= 2   # offensive under-performance; expect recovery

        # Pitching luck: xba < ba_against (avg_pit_gap < 0) means pitchers unlucky
        if avg_pit_gap < -0.015:
            luck_score -= 2   # pitching worse than skill; expect improvement
        elif avg_pit_gap > 0.015:
            luck_score += 2   # pitching luckier than skill; expect regression

        # babip: convert xba gap to a BABIP-like value (league avg ~0.295)
        babip_est = round(0.295 - avg_bat_gap, 3)
        result[team] = {
            "babip":              babip_est,
            "lob_pct":            0.720,       # not available from Savant; use league avg
            "xba_gap_batting":    round(avg_bat_gap, 4),
            "xba_gap_pitching":   round(avg_pit_gap, 4),
            "luck_score":         luck_score,
        }

    if not result:
        logger.warning("luck_metrics_collector: no data returned (Savant unavailable)")
    else:
        logger.info("luck_metrics_collector: fetched %d teams", len(result))

    if result:
        _CACHE = result
        _CACHE_TS = now

    return _CACHE
