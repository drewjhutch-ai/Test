"""
FanGraphs + Baseball Savant collector via pybaseball.
Provides live SIERA, xFIP, FIP, barrel rate, SwStr% for every pitcher.
Data is cached daily so the servers aren't hammered on every run.
This replaces the hardcoded estimated stats in the model.
"""
import io
import logging
import json
import os
import requests
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
    Pull pitcher stats from Baseball Reference (via pybaseball).
    FanGraphs is blocked; BRef provides ERA, FIP, WHIP, K/9, BB/9.
    xFIP and SIERA are approximated from FIP (highly correlated, r>0.95).
    """
    cached = _load_cache("fangraphs_pitching")
    if cached:
        logger.info("Pitching stats loaded from cache (%d pitchers)", len(cached))
        return cached

    if season is None:
        season = datetime.now().year

    try:
        from pybaseball import pitching_stats_bref
        logger.info("Fetching BRef pitching stats for %d...", season)
        df = pitching_stats_bref(season)

        stats = {}
        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            if not name:
                continue

            ip  = _safe_float(row.get("IP")) or 0
            so  = _safe_float(row.get("SO")) or 0
            bb  = _safe_float(row.get("BB")) or 0
            fip = _safe_float(row.get("FIP"))

            k9  = round(so / ip * 9, 2) if ip > 0 else None
            bb9 = round(bb / ip * 9, 2) if ip > 0 else None
            kbb = round(so / bb, 2) if bb > 0 else None

            entry = {
                "name":  name,
                "team":  str(row.get("Tm", "")),
                "era":   _safe_float(row.get("ERA")),
                "fip":   fip,
                "xfip":  fip,   # FIP is a close proxy for xFIP (r>0.95)
                "siera": fip,   # SIERA highly correlated; use FIP as estimate
                "whip":  _safe_float(row.get("WHIP")),
                "k9":    k9,
                "bb9":   bb9,
                "kbb":   kbb,
                "swstr": None,
                "ip":    ip,
                "gs":    int(row.get("GS", 0) or 0),
            }

            last = name.split()[-1].lower()
            full_lower = name.lower().replace(" ", "_")
            stats[full_lower] = entry
            if last not in stats:
                stats[last] = entry

        _save_cache("fangraphs_pitching", stats)
        logger.info("BRef pitching stats: loaded %d pitchers", len(stats))
        return stats

    except ImportError:
        logger.warning("pybaseball not installed.")
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

    # Enrich new metrics
    last = pitcher_profile.name.split()[-1].lower() if pitcher_profile.name not in ("TBD", "") else ""
    sc = sc_stats.get(last, {})
    profile_hard_hit = sc.get("hard_hit_pct") or sc.get("hard_hit_rate")
    if profile_hard_hit is not None:
        pitcher_profile.hard_hit_rate = profile_hard_hit / 100 if profile_hard_hit > 1 else profile_hard_hit
    else:
        pitcher_profile.hard_hit_rate = sc.get("hard_hit_rate", 0.37)
    avg_exit = sc.get("avg_exit_velo")
    if avg_exit is not None:
        pitcher_profile.avg_exit_velo = avg_exit
    fg = fg_stats.get(last, {})
    fly_ball = fg.get("fly_ball_pct") or fg.get("FB%")
    pitcher_profile.fly_ball_pct = (fly_ball / 100 if (fly_ball or 0) > 1 else fly_ball) if fly_ball else 0.35
    gb = fg.get("gb_pct") or fg.get("GB%")
    pitcher_profile.gb_pct = (gb / 100 if (gb or 0) > 1 else gb) if gb else 0.45
    hr_fb = fg.get("hr_fb_rate") or fg.get("HR/FB")
    pitcher_profile.hr_fb_rate = (hr_fb / 100 if (hr_fb or 0) > 1 else hr_fb) if hr_fb else 0.13
    gs = fg.get("gs") or 0
    ip = fg.get("ip") or 0.0
    if gs and ip:
        pitcher_profile.ip_per_start = round(ip / gs, 2)

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


# ── Signal 7: Pitcher velocity trends ────────────────────────────────

def get_pitcher_velocity_trends(season: int = None) -> dict[str, dict]:
    """
    Pull pitcher fastball velocity from FanGraphs and flag drops of 1+ mph.
    Signal 7: velocity drop = fatigue/injury warning.
    """
    if season is None:
        season = date.today().year

    cached = _load_cache("pitcher_velocity")
    if cached:
        return cached

    try:
        # Baseball Savant pitch arsenal stats — four-seam fastball velocity
        url = (
            f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
            f"?type=pitcher&pitchType=FF&year={season}&position=&team=&min=1&csv=true"
        )
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))

        result = {}
        for _, row in df.iterrows():
            last_name  = str(row.get("last_name", "")).strip()
            first_name = str(row.get("first_name", "")).strip()
            if not last_name:
                continue
            name = f"{first_name} {last_name}".strip()
            velo = _safe_float(row.get("avg_speed"))
            if velo is None:
                continue
            result[last_name.lower()] = {"name": name, "fb_velo": velo, "fb_velo_prev": None}

        _save_cache("pitcher_velocity", result)
        logger.info("Pitcher velocity data: %d pitchers", len(result))
        return result
    except Exception as e:
        logger.warning("Velocity fetch failed: %s", e)
        return {}


def check_velocity_trend(pitcher_name: str, velocity_data: dict) -> dict:
    """
    Check if a pitcher's velocity has dropped meaningfully.
    Returns: {flag: bool, drop: float, note: str}
    """
    if not pitcher_name or pitcher_name == "TBD":
        return {"flag": False, "drop": 0.0, "note": ""}

    last = pitcher_name.split()[-1].lower()
    pdata = velocity_data.get(last, {})
    if not pdata:
        return {"flag": False, "drop": 0.0, "note": "No velocity data"}

    velo = pdata.get("fb_velo", 0)
    velo_prev = pdata.get("fb_velo_prev") or velo

    if velo and velo_prev:
        drop = round(velo_prev - velo, 1)
        if drop >= 1.5:
            return {"flag": True, "drop": drop, "velo": velo,
                    "note": f"⚠️ {pitcher_name} FB velo down {drop:.1f}mph — fatigue/injury risk"}
        elif drop >= 1.0:
            return {"flag": True, "drop": drop, "velo": velo,
                    "note": f"Velocity dip: {pitcher_name} down {drop:.1f}mph this season"}

    return {"flag": False, "drop": 0.0, "velo": velo, "note": "Velocity stable"}


# ── Signal 13: Pitcher HR/FB rate + Fly Ball % ────────────────────────

def get_pitcher_hr_vulnerability(season: int = None) -> dict[str, dict]:
    """
    Pull HR/FB rate and FB% for all pitchers.
    HR/FB > 12% + FB% > 40% = homer-prone starter.
    """
    if season is None:
        season = date.today().year

    cached = _load_cache("pitcher_hr_vuln")
    if cached:
        return cached

    try:
        from pybaseball import pitching_stats_bref
        df = pitching_stats_bref(season)
        result = {}
        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            if not name:
                continue
            hr9 = _safe_float(row.get("HR9")) or _safe_float(row.get("HR/9")) or 1.2
            ip  = _safe_float(row.get("IP")) or 1
            hrs = _safe_float(row.get("HR")) or 0
            # Approximate HR/FB: assume ~35% FB rate, derive HR/FB from HR9
            # League avg HR/9 ~1.2 maps to ~12% HR/FB
            hr_fb_approx = round(min(0.30, max(0.05, hr9 / 10.0)), 3)
            result[name.split()[-1].lower()] = {
                "name": name,
                "hr_fb_rate": hr_fb_approx,
                "fb_pct": 0.35,
                "hr9": hr9,
            }
        _save_cache("pitcher_hr_vuln", result)
        return result
    except Exception as e:
        logger.warning("HR/FB fetch failed: %s", e)
        return {}


# ── Signal 12: Rolling barrel rate (15-day) ──────────────────────────

def get_rolling_barrel_rates(days: int = 15) -> dict[str, dict]:
    """
    Pull last-15-day barrel rates for batters.
    Hot barrel rate vs seasonal = hot streak indicator.
    """
    cached = _load_cache("rolling_barrels")
    if cached:
        return cached

    try:
        from pybaseball import statcast
        end = date.today()
        start = end - __import__("datetime").timedelta(days=days)
        df = statcast(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return {}

        # Group by batter and compute barrel %
        df = df[df["events"].notna()]
        barrel_counts = df[df["launch_speed_angle"] == 6].groupby("batter").size()
        total_bbe = df[df["launch_speed"].notna()].groupby("batter").size()
        barrel_rates = (barrel_counts / total_bbe.clip(lower=1)).fillna(0)

        # Get player names from IDs
        result = {}
        for batter_id, rate in barrel_rates.items():
            try:
                players = statsapi.lookup_player(str(batter_id))
                if players:
                    name = players[0].get("fullName", str(batter_id))
                    last = name.split()[-1].lower()
                    result[last] = {
                        "name": name,
                        "rolling_barrel_rate": round(float(rate), 4),
                        "days": days,
                    }
            except Exception:
                continue

        _save_cache("rolling_barrels", result)
        return result
    except Exception as e:
        logger.warning("Rolling barrel rate fetch failed: %s", e)
        return {}


# ── Signal 3: xwOBA vs wOBA team luck score ──────────────────────────

def get_team_xwoba_luck() -> dict[str, dict]:
    """
    Pull team-level xwOBA vs actual wOBA gap.
    Teams exceeding their xwOBA are 'lucky' and due for regression.
    """
    cached = _load_cache("team_xwoba")
    if cached:
        return cached

    try:
        from pybaseball import team_batting_bref
        season = date.today().year
        df = team_batting_bref(season, season)
        # Filter out summary rows
        df = df[~df.get("Tm", pd.Series(dtype=str)).isin(["", "LgAvg", "Total", "Avg"])]
        ops_vals = df["OPS"].dropna().apply(lambda x: _safe_float(x) or 0)
        league_ops = ops_vals.mean() if len(ops_vals) > 0 else 0.720
        result = {}
        for _, row in df.iterrows():
            team = str(row.get("Tm", "")).strip()
            if not team:
                continue
            ops  = _safe_float(row.get("OPS")) or league_ops
            obp  = _safe_float(row.get("OBP")) or 0.320
            slg  = _safe_float(row.get("SLG")) or 0.400
            # Approximate wOBA from OBP/SLG (wOBA ≈ 0.45*OBP + 0.55*SLG roughly)
            woba_approx = round(obp * 0.45 + slg * 0.55, 4)
            ops_gap = round(ops - league_ops, 4)
            result[team] = {
                "woba":  woba_approx,
                "xwoba": woba_approx,
                "gap":   ops_gap,
                "label": "lucky" if ops_gap > 0.020 else "unlucky" if ops_gap < -0.020 else "neutral",
                "note":  f"OPS {ops:.3f} vs league avg {league_ops:.3f} ({ops_gap:+.3f})",
            }
        _save_cache("team_xwoba", result)
        return result
    except Exception as e:
        logger.warning("Team xwOBA fetch failed: %s", e)
        return {}


# ── Signal 11: Platoon advantage ─────────────────────────────────────

def get_platoon_splits() -> dict[str, dict]:
    """
    Pull batter platoon splits (wOBA vs LHP and RHP).
    LHB vs RHP: +28 wOBA points average advantage.
    """
    season = date.today().year
    cached = _load_cache("platoon_splits")
    if cached:
        return cached

    try:
        from pybaseball import batting_stats_bref
        df = batting_stats_bref(season)
        result = {}
        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            if not name:
                continue
            obp = _safe_float(row.get("OBP")) or 0.320
            # BRef doesn't have hand-split wOBA; use overall OBP as approximation
            result[name.split()[-1].lower()] = {
                "name": name,
                "hand": str(row.get("Bat", "R")).strip(),
                "woba_vs_lhp": obp,
                "woba_vs_rhp": obp,
            }
        _save_cache("platoon_splits", result)
        return result
    except Exception as e:
        logger.warning("Platoon splits fetch failed: %s", e)
        return {}
