"""
FanGraphs + Baseball Savant collector via pybaseball.
Provides live SIERA, xFIP, FIP, barrel rate, SwStr% for every pitcher.
Data is cached daily so the servers aren't hammered on every run.
This replaces the hardcoded estimated stats in the model.
"""
import logging
import json
import os
from datetime import datetime, date
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:
    today = date.today().isoformat()
    return CACHE_DIR / f"{name}_{today}.json"


def _load_cache(name: str) -> dict | None:
    path = _cache_path(name)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(name: str, data: dict):
    path = _cache_path(name)
    with open(path, "w") as f:
        json.dump(data, f)
    # Clean up yesterday's cache files
    for old in CACHE_DIR.glob(f"{name}_*.json"):
        if old != path:
            try:
                old.unlink()
            except Exception:
                pass


def get_pitcher_stats_fangraphs(season: int = None) -> dict[str, dict]:
    """
    Pull full pitcher stats from FanGraphs via pybaseball.
    Returns dict keyed by last name (lowercase) for fuzzy matching.
    Cached daily.
    """
    cached = _load_cache("fangraphs_pitching")
    if cached:
        logger.info("FanGraphs pitching stats loaded from cache (%d pitchers)", len(cached))
        return cached

    if season is None:
        season = datetime.now().year

    try:
        from pybaseball import pitching_stats
        logger.info("Fetching FanGraphs pitching stats for %d season...", season)

        df = pitching_stats(season, qual=1)  # qual=1 = any IP, gets everyone

        stats = {}
        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            if not name:
                continue

            entry = {
                "name": name,
                "team": str(row.get("Team", "")),
                "era":   _safe_float(row.get("ERA")),
                "fip":   _safe_float(row.get("FIP")),
                "xfip":  _safe_float(row.get("xFIP")),
                "siera": _safe_float(row.get("SIERA")),
                "whip":  _safe_float(row.get("WHIP")),
                "k9":    _safe_float(row.get("K/9")),
                "bb9":   _safe_float(row.get("BB/9")),
                "kbb":   _safe_float(row.get("K/BB")),
                "swstr": _safe_float(row.get("SwStr%")),
                "ip":    _safe_float(row.get("IP")),
                "gs":    int(row.get("GS", 0) or 0),
                "war":   _safe_float(row.get("WAR")),
            }

            # Key by lowercase last name for matching
            last = name.split()[-1].lower()
            full_lower = name.lower().replace(" ", "_")
            stats[full_lower] = entry
            # Also store by last name if no collision
            if last not in stats:
                stats[last] = entry

        _save_cache("fangraphs_pitching", stats)
        logger.info("FanGraphs: loaded %d pitchers", len(df))
        return stats

    except ImportError:
        logger.warning("pybaseball not installed. Run: pip install pybaseball")
        return {}
    except Exception as e:
        logger.error("FanGraphs fetch failed: %s", e)
        return {}


def get_statcast_pitcher_metrics(season: int = None) -> dict[str, dict]:
    """
    Pull Statcast metrics from Baseball Savant via pybaseball.
    Provides barrel rate, hard hit %, exit velocity, spin rate.
    Cached daily.
    """
    cached = _load_cache("statcast_pitchers")
    if cached:
        logger.info("Statcast pitcher metrics loaded from cache (%d pitchers)", len(cached))
        return cached

    if season is None:
        season = datetime.now().year

    try:
        from pybaseball import statcast_pitcher_exitvelo_barrels
        logger.info("Fetching Statcast pitcher metrics for %d...", season)

        df = statcast_pitcher_exitvelo_barrels(season)

        stats = {}
        for _, row in df.iterrows():
            name = str(row.get("last_name, first_name", "")).strip()
            # Savant format is "Last, First" — flip it
            if "," in name:
                parts = name.split(",")
                name = f"{parts[1].strip()} {parts[0].strip()}"

            if not name:
                continue

            entry = {
                "name": name,
                "barrel_rate":     _safe_float(row.get("barrel_batted_rate")),
                "hard_hit_pct":    _safe_float(row.get("hard_hit_percent")),
                "avg_exit_velo":   _safe_float(row.get("avg_hit_speed")),
                "avg_launch_angle":_safe_float(row.get("avg_hit_angle")),
                "xba":             _safe_float(row.get("xba")),
                "xslg":            _safe_float(row.get("xslg")),
                "xwoba":           _safe_float(row.get("xwoba")),
            }

            full_lower = name.lower().replace(" ", "_")
            last = name.split()[-1].lower()
            stats[full_lower] = entry
            if last not in stats:
                stats[last] = entry

        _save_cache("statcast_pitchers", stats)
        logger.info("Statcast: loaded metrics for %d pitchers", len(df))
        return stats

    except Exception as e:
        logger.warning("Statcast pitcher metrics fetch failed: %s", e)
        return {}


def get_statcast_batter_metrics(season: int = None) -> dict[str, dict]:
    """
    Pull Statcast batter metrics — exit velocity, barrel rate, xwOBA.
    Used for HR prop 5-factor check.
    """
    cached = _load_cache("statcast_batters")
    if cached:
        return cached

    if season is None:
        season = datetime.now().year

    try:
        from pybaseball import statcast_batter_exitvelo_barrels
        df = statcast_batter_exitvelo_barrels(season)

        stats = {}
        for _, row in df.iterrows():
            name = str(row.get("last_name, first_name", "")).strip()
            if "," in name:
                parts = name.split(",")
                name = f"{parts[1].strip()} {parts[0].strip()}"
            if not name:
                continue

            entry = {
                "name": name,
                "barrel_rate":   _safe_float(row.get("barrel_batted_rate")),
                "hard_hit_pct":  _safe_float(row.get("hard_hit_percent")),
                "avg_exit_velo": _safe_float(row.get("avg_hit_speed")),
                "xwoba":         _safe_float(row.get("xwoba")),
                "xba":           _safe_float(row.get("xba")),
            }
            full_lower = name.lower().replace(" ", "_")
            last = name.split()[-1].lower()
            stats[full_lower] = entry
            if last not in stats:
                stats[last] = entry

        _save_cache("statcast_batters", stats)
        return stats
    except Exception as e:
        logger.warning("Statcast batter metrics fetch failed: %s", e)
        return {}


def lookup_pitcher(name: str, fg_stats: dict, sc_stats: dict) -> dict:
    """
    Look up a pitcher by name across FanGraphs and Statcast data.
    Uses fuzzy last-name matching as fallback.
    Returns merged stats dict or empty dict if not found.
    """
    if not name or name == "TBD":
        return {}

    # Try exact full name match first
    full_lower = name.lower().replace(" ", "_")
    fg = fg_stats.get(full_lower, {})
    sc = sc_stats.get(full_lower, {})

    # Fall back to last name
    if not fg:
        last = name.split()[-1].lower()
        fg = fg_stats.get(last, {})
    if not sc:
        last = name.split()[-1].lower()
        sc = sc_stats.get(last, {})

    if not fg and not sc:
        logger.debug("No FanGraphs/Statcast data found for pitcher: %s", name)
        return {}

    merged = {**fg, **sc}
    merged["data_source"] = "fangraphs+statcast"
    return merged


def enrich_pitcher_profile(pitcher_profile, fg_stats: dict, sc_stats: dict):
    """
    Update a PitcherProfile in-place with real FanGraphs + Statcast data.
    Only overwrites fields where real data exists — keeps estimates otherwise.
    """
    data = lookup_pitcher(pitcher_profile.name, fg_stats, sc_stats)
    if not data:
        return pitcher_profile

    if data.get("era"):
        pitcher_profile.era = data["era"]
    if data.get("fip"):
        pitcher_profile.fip = data["fip"]
    if data.get("xfip"):
        pitcher_profile.xfip = data["xfip"]
    if data.get("siera"):
        pitcher_profile.siera = data["siera"]
    if data.get("whip"):
        pitcher_profile.whip = data["whip"]
    if data.get("k9"):
        pitcher_profile.k9 = data["k9"]
    if data.get("bb9"):
        pitcher_profile.bb9 = data["bb9"]
    if data.get("kbb"):
        pitcher_profile.kbb_ratio = data["kbb"]
    if data.get("swstr"):
        pitcher_profile.swstr_pct = data["swstr"] / 100 if data["swstr"] > 1 else data["swstr"]
    if data.get("barrel_rate"):
        pitcher_profile.barrel_rate = data["barrel_rate"] / 100 if data["barrel_rate"] > 1 else data["barrel_rate"]

    # Mark as confirmed from 2 real sources
    pitcher_profile.confirmed_sources = 2

    logger.debug(
        "Enriched %s: ERA %.2f | FIP %.2f | SIERA %.2f | xFIP %.2f | SwStr %.1f%%",
        pitcher_profile.name,
        pitcher_profile.era, pitcher_profile.fip,
        pitcher_profile.siera, pitcher_profile.xfip,
        (pitcher_profile.swstr_pct or 0) * 100,
    )
    return pitcher_profile


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if pd.isna(f) else round(f, 3)
    except (TypeError, ValueError):
        return None
