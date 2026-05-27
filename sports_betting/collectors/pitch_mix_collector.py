"""
Pitch mix change detector — compares season-level vs recent (last 14 days)
pitch type usage for starting pitchers.

Detects:
  - new_pitch      : pitch type went from 0% season → >5% recent
  - primary_dropped: primary pitch usage dropped >10pp recently
  - pitch_abandoned: pitch type went from >15% season → <5% recent

Data sources:
  - Season:  Baseball Savant pitch arsenal leaderboard (csv)
  - Recent:  Baseball Savant Statcast search (last 14 days, aggregated by name+type)
"""
from __future__ import annotations
import csv
import io
import logging
import time
from datetime import datetime, timedelta

import requests
from .savant_headers import SAVANT_HEADERS

logger = logging.getLogger(__name__)

_SEASON_CACHE: dict = {}
_SEASON_CACHE_TS: float = 0.0

_RECENT_CACHE: dict = {}
_RECENT_CACHE_TS: float = 0.0

_TTL: float = 6 * 3600  # 6 hours

_HEADERS = SAVANT_HEADERS

_ARSENAL_URL = (
    "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
    "?type=pitcher&pitchType=&year=2026&team=&min=10&csv=true"
)


def _recent_search_url() -> str:
    today = datetime.utcnow().date()
    cutoff = today - timedelta(days=14)
    return (
        "https://baseballsavant.mlb.com/statcast_search/csv"
        "?all=true&type=details&player_type=pitcher"
        f"&game_date_gt={cutoff}&game_date_lt={today}"
        "&pitch_type=&team=&position=1&hfSea=2026|"
        "&min_pitches=0&min_results=0&group_by=name_type"
        "&sort_col=pitches&player_event_sort=api_p_release_speed"
        "&sort_order=desc&min_pas=0"
    )


def _safe_float(val: str | None, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "null") else default
    except (ValueError, TypeError):
        return default


def _fetch_csv(url: str) -> list[dict]:
    """Fetch a CSV URL, return list of dicts. Returns [] on failure."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        if resp.status_code == 403:
            logger.warning("pitch_mix_collector: 403 for %s", url[:80])
            return []
        resp.raise_for_status()
        text = resp.content.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        return list(csv.DictReader(io.StringIO(text)))
    except Exception as exc:
        logger.warning("pitch_mix_collector: fetch failed (%s): %s", url[:80], exc)
        return []


def _get_season_arsenal() -> dict[str, dict[str, float]]:
    """
    Returns {pitcher_name: {pitch_type: usage_pct, ...}} for season-level data.
    Usage is stored as 0.0–100.0 float (percentage).
    """
    global _SEASON_CACHE, _SEASON_CACHE_TS
    now = time.time()
    if _SEASON_CACHE and (now - _SEASON_CACHE_TS) < _TTL:
        return _SEASON_CACHE

    rows = _fetch_csv(_ARSENAL_URL)
    result: dict[str, dict[str, float]] = {}

    for row in rows:
        # Columns vary; try common names
        name = (row.get("player_name") or row.get("PlayerName") or row.get("name") or "").strip()
        pitch_type = (
            row.get("pitch_type") or row.get("pitch_name") or row.get("PitchType") or ""
        ).strip().upper()
        if not name or not pitch_type:
            continue

        # Usage percentage column
        usage_raw = (
            row.get("pitch_usage") or row.get("usage") or row.get("usage_pct")
            or row.get("pitch_percent") or "0"
        )
        usage_str = str(usage_raw).replace("%", "")
        usage_pct = _safe_float(usage_str, 0.0)
        # Normalize: if value looks like 0.28 (decimal), convert to percent
        if usage_pct <= 1.0 and usage_pct > 0:
            usage_pct *= 100.0

        if name not in result:
            result[name] = {}
        result[name][pitch_type] = usage_pct

    if result:
        _SEASON_CACHE = result
        _SEASON_CACHE_TS = now
        logger.info("pitch_mix_collector: season arsenal loaded for %d pitchers", len(result))
    else:
        logger.warning("pitch_mix_collector: season arsenal empty; using stale cache")

    return _SEASON_CACHE


def _get_recent_arsenal() -> dict[str, dict[str, float]]:
    """
    Returns {pitcher_name: {pitch_type: usage_pct}} for last 14 days.
    Aggregates pitch counts by pitcher+type, then computes usage%.
    """
    global _RECENT_CACHE, _RECENT_CACHE_TS
    now = time.time()
    if _RECENT_CACHE and (now - _RECENT_CACHE_TS) < _TTL:
        return _RECENT_CACHE

    rows = _fetch_csv(_recent_search_url())

    # Accumulate pitch counts
    # group_by=name_type should give pre-aggregated rows, but we handle both formats
    counts: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}

    for row in rows:
        name = (row.get("player_name") or row.get("pitcher_name") or "").strip()
        pitch_type = (
            row.get("pitch_type") or row.get("pitch_name") or ""
        ).strip().upper()
        if not name or not pitch_type:
            continue

        n = int(_safe_float(row.get("pitches") or row.get("n") or "1", 1.0))
        if name not in counts:
            counts[name] = {}
            totals[name] = 0
        counts[name][pitch_type] = counts[name].get(pitch_type, 0) + n
        totals[name] += n

    result: dict[str, dict[str, float]] = {}
    for name, pitch_counts in counts.items():
        total = max(1, totals[name])
        result[name] = {
            pt: round(cnt / total * 100.0, 1)
            for pt, cnt in pitch_counts.items()
        }

    if result:
        _RECENT_CACHE = result
        _RECENT_CACHE_TS = now
        logger.info("pitch_mix_collector: recent arsenal loaded for %d pitchers", len(result))
    else:
        logger.warning("pitch_mix_collector: recent arsenal empty; using stale cache")

    return _RECENT_CACHE


def get_pitch_mix_changes() -> dict[str, dict]:
    """
    Compare season vs recent pitch usage for all pitchers.

    Returns dict keyed by pitcher name →
        {
          mix_change  : bool
          change_type : "new_pitch" | "primary_dropped" | "pitch_abandoned" | None
          details     : str  (human-readable description)
        }

    Returns {} on failure (both fetches empty). Cached 6-hour TTL.
    """
    season_data = _get_season_arsenal()
    recent_data = _get_recent_arsenal()

    if not season_data and not recent_data:
        return {}

    result: dict[str, dict] = {}

    all_names = set(season_data.keys()) | set(recent_data.keys())

    for name in all_names:
        season_mix = season_data.get(name, {})
        recent_mix = recent_data.get(name, {})

        if not season_mix or not recent_mix:
            # Insufficient data for comparison
            result[name] = {
                "mix_change": False,
                "change_type": None,
                "details": "insufficient data for mix comparison",
            }
            continue

        # Find primary pitch (highest usage in season)
        primary_pitch = max(season_mix, key=lambda p: season_mix[p], default=None)
        primary_season_pct = season_mix.get(primary_pitch, 0.0) if primary_pitch else 0.0
        primary_recent_pct = recent_mix.get(primary_pitch, 0.0) if primary_pitch else 0.0

        change_type: str | None = None
        details: str = "no significant pitch mix change"
        mix_change = False

        # Check for new pitch (0% season → >5% recent)
        for pt, recent_pct in recent_mix.items():
            season_pct = season_mix.get(pt, 0.0)
            if season_pct == 0.0 and recent_pct > 5.0:
                change_type = "new_pitch"
                details = (
                    f"New pitch type {pt} added: 0% season → {recent_pct:.1f}% recent"
                )
                mix_change = True
                break

        # Check for primary pitch dropped >10pp (if no new pitch found yet)
        if not mix_change and primary_pitch:
            drop = primary_season_pct - primary_recent_pct
            if drop > 10.0:
                change_type = "primary_dropped"
                details = (
                    f"Primary pitch {primary_pitch} usage dropped "
                    f"{primary_season_pct:.1f}% → {primary_recent_pct:.1f}% "
                    f"(↓{drop:.1f}pp)"
                )
                mix_change = True

        # Check for abandoned pitch (>15% season → <5% recent)
        if not mix_change:
            for pt, season_pct in season_mix.items():
                if season_pct > 15.0:
                    recent_pct = recent_mix.get(pt, 0.0)
                    if recent_pct < 5.0:
                        change_type = "pitch_abandoned"
                        details = (
                            f"Pitch {pt} abandoned: {season_pct:.1f}% season → "
                            f"{recent_pct:.1f}% recent"
                        )
                        mix_change = True
                        break

        result[name] = {
            "mix_change":  mix_change,
            "change_type": change_type,
            "details":     details,
        }

    logger.info("pitch_mix_collector: analyzed %d pitchers for mix changes", len(result))
    return result
