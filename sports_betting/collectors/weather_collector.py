"""
Weather data collector for MLB stadiums.
Weather significantly affects run totals (wind, cold temps, rain).
"""
import logging
import requests
from ..config import WEATHER_API_KEY, WEATHER_BASE, MLB_STADIUMS
from ..database import get_db

logger = logging.getLogger(__name__)


def get_game_weather(team_name: str, game_id: str) -> dict:
    """Fetch weather for a team's stadium and calculate impact score."""
    stadium = MLB_STADIUMS.get(team_name)
    if not stadium:
        logger.warning("No stadium data for team: %s", team_name)
        return {"weather_impact_score": 0, "roof_closed": False}

    # Domed stadiums aren't weather-affected
    if stadium.get("roof"):
        with get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO weather_data
                (game_id, conditions, roof_closed, weather_impact_score)
                VALUES (?, 'Dome/Retractable Roof', 1, 0)
            """, (game_id,))
        return {
            "conditions": "Dome/Retractable Roof",
            "roof_closed": True,
            "weather_impact_score": 0,
        }

    if not WEATHER_API_KEY:
        logger.warning("No WEATHER_API_KEY set. Skipping weather fetch.")
        return {"weather_impact_score": 0, "roof_closed": False}

    try:
        url = f"{WEATHER_BASE}/forecast"
        params = {
            "lat": stadium["lat"],
            "lon": stadium["lon"],
            "appid": WEATHER_API_KEY,
            "units": "imperial",
            "cnt": 8,  # Next 24 hours in 3hr blocks
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Use the closest forecast to game time (simplified: first entry)
        forecast = data["list"][0]
        weather = {
            "temperature": forecast["main"]["temp"],
            "feels_like": forecast["main"]["feels_like"],
            "humidity": forecast["main"]["humidity"],
            "wind_speed": forecast["wind"]["speed"],
            "wind_direction": forecast["wind"]["deg"],
            "wind_gust": forecast["wind"].get("gust", 0),
            "conditions": forecast["weather"][0]["description"],
            "precipitation_chance": forecast.get("pop", 0) * 100,
            "visibility": data.get("visibility", 10000) / 1609.34,  # meters to miles
            "roof_closed": False,
        }
        weather["weather_impact_score"] = _compute_weather_impact(weather, stadium)

        _store_weather(game_id, weather)
        logger.info("Weather fetched for %s: %.0f°F, %s, wind %.1f mph",
                    team_name, weather["temperature"], weather["conditions"],
                    weather["wind_speed"])
        return weather

    except Exception as e:
        logger.error("Weather fetch failed for %s: %s", team_name, e)
        return {"weather_impact_score": 0, "roof_closed": False}


def _compute_weather_impact(weather: dict, stadium: dict) -> float:
    """
    Score weather impact on run scoring (-1.0 to +1.0).
    Positive = more runs expected, negative = fewer runs.
    Factors: wind (direction matters at Wrigley), temp, precipitation.
    """
    score = 0.0

    # Temperature impact: cold suppresses offense
    temp = weather["temperature"]
    if temp < 40:
        score -= 0.4
    elif temp < 50:
        score -= 0.25
    elif temp < 60:
        score -= 0.1
    elif temp > 85:
        score += 0.05  # Hot air slightly benefits hitters

    # Wind impact: out to CF/RF boosts offense, in suppresses
    wind_speed = weather["wind_speed"]
    wind_dir = weather["wind_direction"]
    if wind_speed > 15:
        # Blowing out (roughly E-SE for most stadiums facing west/NW)
        # Simplified: wind direction 45-135 degrees often blows out to LF/CF
        if 45 <= wind_dir <= 135:
            score += wind_speed * 0.025  # Strong out wind, big boost
        elif 225 <= wind_dir <= 315:
            score -= wind_speed * 0.020  # Wind in, suppress scoring
        else:
            score += wind_speed * 0.005  # Cross wind, minor effect

    # Precipitation suppresses scoring and increases postponement risk
    precip_pct = weather["precipitation_chance"]
    if precip_pct > 70:
        score -= 0.3
    elif precip_pct > 40:
        score -= 0.15
    elif precip_pct > 20:
        score -= 0.05

    # Humidity: high humidity at warm temps can affect ball carry
    if weather["humidity"] > 80 and temp > 75:
        score += 0.05

    return round(max(-1.0, min(1.0, score)), 3)


def _store_weather(game_id: str, weather: dict):
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO weather_data
            (game_id, temperature, feels_like, humidity, wind_speed, wind_direction,
             wind_gust, conditions, precipitation_chance, visibility, roof_closed,
             weather_impact_score)
            VALUES (:game_id, :temperature, :feels_like, :humidity, :wind_speed,
                    :wind_direction, :wind_gust, :conditions, :precipitation_chance,
                    :visibility, :roof_closed, :weather_impact_score)
        """, {"game_id": game_id, **weather})


def get_stored_weather(game_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM weather_data WHERE game_id=?", (game_id,)
        ).fetchone()
        return dict(row) if row else None


def classify_weather_impact(score: float) -> str:
    if score >= 0.3:
        return "STRONG OVER"
    elif score >= 0.15:
        return "MILD OVER"
    elif score <= -0.3:
        return "STRONG UNDER"
    elif score <= -0.15:
        return "MILD UNDER"
    return "NEUTRAL"
