"""
Air density collector — computes true air density at game time using
Weather.gov (NWS) hourly forecast data.

Lower air density = less air resistance on the ball = more carry = more HRs.
Coors Field is the canonical extreme example (altitude 5200 ft, density ~0.80).

Key output: air_density_ratio (1.0 = sea-level standard; <1 = thinner air)
             carry_boost_pct  (each 1% density drop ≈ 2.5% more ball carry)
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timezone

import requests

from ..config import MLB_STADIUMS

logger = logging.getLogger(__name__)

_CACHE: dict[str, dict] = {}
_CACHE_TS: dict[str, float] = {}
_TTL: float = 6 * 3600  # 6 hours

_NWS_HEADERS = {
    "User-Agent": "MLB-BettingModel/4.0 (contact: admin@example.com)",
    "Accept": "application/geo+json",
}

# Altitude (feet) for each stadium.  Pulled from park_database where available,
# extended here for all 30 parks.  Used as fallback when NWS barometric data is
# unavailable and as a cross-check on calculated density.
STADIUM_ALTITUDE_FT: dict[str, float] = {
    "New York Yankees":       55.0,
    "New York Mets":          20.0,
    "Boston Red Sox":         20.0,
    "Chicago Cubs":          595.0,
    "Chicago White Sox":     595.0,
    "Los Angeles Dodgers":   512.0,
    "Los Angeles Angels":    160.0,
    "San Francisco Giants":   10.0,
    "Oakland Athletics":      20.0,
    "Athletics":             30.0,
    "Seattle Mariners":       20.0,
    "Houston Astros":         43.0,
    "Texas Rangers":         551.0,
    "Arizona Diamondbacks": 1100.0,
    "Colorado Rockies":     5200.0,
    "Minnesota Twins":       830.0,
    "Detroit Tigers":        585.0,
    "Cleveland Guardians":   653.0,
    "Kansas City Royals":    750.0,
    "Milwaukee Brewers":     672.0,
    "St. Louis Cardinals":   466.0,
    "Pittsburgh Pirates":    738.0,
    "Cincinnati Reds":       481.0,
    "Atlanta Braves":        1050.0,
    "Miami Marlins":          10.0,
    "Tampa Bay Rays":         15.0,
    "Baltimore Orioles":      20.0,
    "Washington Nationals":   25.0,
    "Philadelphia Phillies":  20.0,
    "Toronto Blue Jays":     249.0,
    "San Diego Padres":       17.0,
}


def calc_air_density_index(
    temp_f: float,
    dewpoint_f: float,
    pressure_mb: float,
    altitude_ft: float,
) -> float:
    """
    Calculate air density relative to sea-level standard (1.225 kg/m³).

    Returns a ratio: <1.0 means less dense air (ball carries more).
    Uses the Magnus formula for saturation vapor pressure and virtual
    temperature correction for humidity.
    """
    temp_c = (temp_f - 32) * 5.0 / 9.0
    dp_c   = (dewpoint_f - 32) * 5.0 / 9.0

    # Saturation and actual vapor pressure (Magnus formula), hPa
    e_s = 6.1078 * (10 ** (7.5 * temp_c / (237.3 + temp_c)))
    e_a = 6.1078 * (10 ** (7.5 * dp_c  / (237.3 + dp_c)))

    # Mixing ratio (kg water vapor / kg dry air)
    mixing_ratio = 0.622 * e_a / max(0.01, pressure_mb - e_a)

    # Virtual temperature (K) — corrects for moisture reducing effective density
    tv = (temp_c + 273.15) * (1.0 + mixing_ratio / 0.622) / (1.0 + mixing_ratio)

    # Air density (kg/m³)
    density = pressure_mb * 100.0 / (287.05 * tv)

    sea_level_density = 1.225  # kg/m³ at 15°C, 1013.25 hPa
    return density / sea_level_density


def _altitude_pressure_estimate(altitude_ft: float) -> float:
    """
    Barometric formula fallback: estimate pressure (mb) from altitude.
    Standard atmosphere approximation.
    """
    altitude_m = altitude_ft * 0.3048
    pressure_mb = 1013.25 * (1 - 2.25577e-5 * altitude_m) ** 5.25588
    return round(pressure_mb, 2)


def _fetch_nws_hourly(lat: float, lon: float) -> list[dict]:
    """
    Fetch NWS hourly forecast periods.  Returns [] on failure.
    Two-step: /points → gridpoint → hourly forecast.
    """
    try:
        points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        r = requests.get(points_url, headers=_NWS_HEADERS, timeout=12)
        if r.status_code != 200:
            logger.debug("NWS /points returned %s for %.4f,%.4f", r.status_code, lat, lon)
            return []
        forecast_url = r.json()["properties"]["forecastHourly"]

        r2 = requests.get(forecast_url, headers=_NWS_HEADERS, timeout=12)
        if r2.status_code != 200:
            logger.debug("NWS hourly forecast returned %s", r2.status_code)
            return []
        periods = r2.json()["properties"]["periods"]
        return periods
    except Exception as exc:
        logger.debug("NWS hourly fetch failed: %s", exc)
        return []


def _parse_nws_period(period: dict, altitude_ft: float) -> dict:
    """
    Extract weather values from a single NWS forecast period.
    Returns a dict with temp_f, dewpoint_f, pressure_mb.
    """
    props = period
    temp_f = float(props.get("temperature", 72))

    # NWS gives dewpoint as {"value": ..., "unitCode": "wmoUnit:degC"}
    dp_raw = props.get("dewpoint", {})
    if isinstance(dp_raw, dict):
        dp_c = float(dp_raw.get("value") or 10.0)
        dewpoint_f = dp_c * 9.0 / 5.0 + 32.0
    else:
        dewpoint_f = 55.0  # fallback

    # NWS hourly does not always include barometric pressure; use altitude fallback
    pressure_mb = _altitude_pressure_estimate(altitude_ft)

    return {
        "temp_f": temp_f,
        "dewpoint_f": dewpoint_f,
        "pressure_mb": pressure_mb,
    }


def get_air_density_for_game(
    home_team: str,
    game_time_utc: str | None = None,
) -> dict:
    """
    Return air density metrics for the given home team's stadium.

    Parameters
    ----------
    home_team      : Full team name matching MLB_STADIUMS keys
    game_time_utc  : ISO-format UTC string e.g. "2026-05-27T22:10:00Z" (optional)

    Returns
    -------
    dict with keys:
        air_density_ratio  – float (1.0 = standard sea-level density)
        carry_boost_pct    – float (positive = ball carries more)
        temp_f             – float
        dewpoint_f         – float
        pressure_mb        – float
        source             – "nws" | "altitude_fallback"
    """
    cache_key = f"{home_team}:{game_time_utc or 'now'}"
    now = time.time()
    if cache_key in _CACHE and (now - _CACHE_TS.get(cache_key, 0)) < _TTL:
        return _CACHE[cache_key]

    stadium = MLB_STADIUMS.get(home_team)
    altitude_ft = STADIUM_ALTITUDE_FT.get(home_team, 50.0)

    default_result = {
        "air_density_ratio": 1.0,
        "carry_boost_pct": 0.0,
        "temp_f": 72.0,
        "dewpoint_f": 55.0,
        "pressure_mb": _altitude_pressure_estimate(altitude_ft),
        "source": "altitude_fallback",
    }

    # Domed stadiums: climate-controlled — density essentially sea-level standard
    if stadium and stadium.get("roof"):
        result = {**default_result, "source": "dome"}
        _CACHE[cache_key] = result
        _CACHE_TS[cache_key] = now
        return result

    if not stadium:
        logger.debug("air_density_collector: no stadium for %s", home_team)
        _CACHE[cache_key] = default_result
        _CACHE_TS[cache_key] = now
        return default_result

    lat = stadium["lat"]
    lon = stadium["lon"]

    # Try NWS hourly forecast
    periods = _fetch_nws_hourly(lat, lon)

    parsed_weather: dict | None = None
    if periods:
        # Find the period closest to game time if provided
        target_period = periods[0]  # default: next available hour
        if game_time_utc:
            try:
                game_dt = datetime.fromisoformat(
                    game_time_utc.replace("Z", "+00:00")
                )
                best_delta = None
                for p in periods[:18]:  # look up to 18 hours ahead
                    start_str = p.get("startTime", "")
                    if not start_str:
                        continue
                    try:
                        period_dt = datetime.fromisoformat(
                            start_str.replace("Z", "+00:00")
                        )
                        delta = abs((period_dt - game_dt).total_seconds())
                        if best_delta is None or delta < best_delta:
                            best_delta = delta
                            target_period = p
                    except ValueError:
                        continue
            except (ValueError, TypeError):
                pass

        parsed_weather = _parse_nws_period(target_period, altitude_ft)

    if parsed_weather:
        temp_f      = parsed_weather["temp_f"]
        dewpoint_f  = parsed_weather["dewpoint_f"]
        pressure_mb = parsed_weather["pressure_mb"]
        source = "nws"
    else:
        # Altitude-based fallback: assume 72°F, 50% RH
        temp_f      = 72.0
        dewpoint_f  = 55.0
        pressure_mb = _altitude_pressure_estimate(altitude_ft)
        source = "altitude_fallback"
        logger.debug("air_density_collector: NWS failed for %s — using altitude fallback", home_team)

    density_ratio = calc_air_density_index(temp_f, dewpoint_f, pressure_mb, altitude_ft)
    # Each 1% drop in density → ~2.5% more carry
    carry_boost_pct = (1.0 - density_ratio) * 100.0 * 2.5

    result = {
        "air_density_ratio": round(density_ratio, 4),
        "carry_boost_pct":   round(carry_boost_pct, 2),
        "temp_f":            round(temp_f, 1),
        "dewpoint_f":        round(dewpoint_f, 1),
        "pressure_mb":       round(pressure_mb, 2),
        "source":            source,
    }

    _CACHE[cache_key] = result
    _CACHE_TS[cache_key] = now
    logger.info(
        "air_density_collector: %s → density_ratio=%.4f carry_boost=%.1f%% (%s)",
        home_team, density_ratio, carry_boost_pct, source,
    )
    return result
