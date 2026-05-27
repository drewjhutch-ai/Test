"""
Luck Metrics Collector — Tier 3, Layer 22.
Computes real team BABIP from MLB Stats API component stats.
BABIP = (H - HR) / (AB - K - HR + SF)
Returns luck_score for each team: positive = luck-inflated, negative = luck-deflated.
"""
from __future__ import annotations
import logging
import time

import requests

logger = logging.getLogger(__name__)

_CACHE: dict = {}
_CACHE_TS: float = 0.0
_TTL: float = 6 * 3600

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


def _compute_babip(h: float, hr: float, ab: float, k: float, sf: float) -> float:
    """BABIP = (H - HR) / (AB - K - HR + SF). Returns -1.0 if denominator < 10."""
    denom = ab - k - hr + sf
    if denom < 10:
        return -1.0
    return (h - hr) / denom


def get_luck_metrics() -> dict[str, dict]:
    """
    Compute team BABIP from MLB Stats API component hitting stats.

    Returns:
        {team_abbrev: {babip, lob_pct, luck_score}}
        luck_score range -4 to +4:
            positive = luck-inflated (expect regression)
            negative = luck-deflated (expect improvement)
    """
    global _CACHE, _CACHE_TS
    now = time.time()
    if _CACHE and (now - _CACHE_TS) < _TTL:
        return _CACHE

    try:
        resp = requests.get(_BATTER_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        splits = resp.json()["stats"][0]["splits"]
    except Exception as exc:
        logger.warning("luck_metrics_collector: fetch failed: %s", exc)
        return _CACHE

    # Accumulate component stats per team
    team_acc: dict[str, dict] = {}
    for sp in splits:
        team = sp.get("team", {}).get("abbreviation", "").strip().upper()
        if not team:
            continue
        stat = sp.get("stat", {})
        h   = _safe_float(stat.get("hits"))
        hr  = _safe_float(stat.get("homeRuns"))
        ab  = _safe_float(stat.get("atBats"))
        k   = _safe_float(stat.get("strikeOuts"))
        sf  = _safe_float(stat.get("sacFlies"))
        if ab < 1:
            continue
        if team not in team_acc:
            team_acc[team] = {"h": 0.0, "hr": 0.0, "ab": 0.0, "k": 0.0, "sf": 0.0}
        team_acc[team]["h"]  += h
        team_acc[team]["hr"] += hr
        team_acc[team]["ab"] += ab
        team_acc[team]["k"]  += k
        team_acc[team]["sf"] += sf

    result: dict[str, dict] = {}
    for team, acc in team_acc.items():
        babip = _compute_babip(acc["h"], acc["hr"], acc["ab"], acc["k"], acc["sf"])
        if babip < 0:
            continue

        luck_score = 0.0
        if babip > 0.320:
            luck_score += 2   # over-performing; expect regression
        elif babip < 0.270:
            luck_score -= 2   # under-performing; expect improvement

        result[team] = {
            "babip":      round(babip, 3),
            "lob_pct":    0.720,   # not computable from available stats; use league avg
            "luck_score": luck_score,
        }

    if result:
        _CACHE = result
        _CACHE_TS = now
        logger.info("luck_metrics_collector: computed BABIP for %d teams", len(result))
    else:
        logger.warning("luck_metrics_collector: no data returned")

    return _CACHE
