"""
MLB data collector using the official MLB StatsAPI.
Pulls team stats, pitcher stats, roster data, and game logs.
"""
import logging
import json
from datetime import datetime, timedelta
import statsapi
import requests
from ..config import MLB_BASE_URL
from ..database import get_db

logger = logging.getLogger(__name__)


def get_todays_games() -> list[dict]:
    """Fetch today's MLB schedule with probable pitchers."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        schedule = statsapi.schedule(date=today, sportId=1)
        games = []
        for g in schedule:
            games.append({
                "game_id": f"mlb_{g['game_id']}",
                "home_team": g["home_name"],
                "away_team": g["away_name"],
                "game_date": today,
                "game_time": g.get("game_datetime", ""),
                "venue": g.get("venue_name", ""),
                "status": g.get("status", "scheduled"),
                "home_probable_pitcher": g.get("home_probable_pitcher", ""),
                "away_probable_pitcher": g.get("away_probable_pitcher", ""),
                "home_score": g.get("home_score"),
                "away_score": g.get("away_score"),
            })
        logger.info("Fetched %d games for %s", len(games), today)
        return games
    except Exception as e:
        logger.error("Failed to fetch schedule: %s", e)
        return []


def get_team_standings() -> list[dict]:
    """Get current MLB standings with win/loss records."""
    try:
        standings = statsapi.standings_data(leagueId="103,104")
        result = []
        for div_id, div_data in standings.items():
            for team in div_data["teams"]:
                result.append({
                    "team_id": team["team_id"],
                    "team_name": team["name"],
                    "wins": team["w"],
                    "losses": team["l"],
                    "pct": float(team["pct"]),
                    "gb": team["gb"],
                    "streak": team.get("streak", ""),
                    "last10": team.get("last10", ""),
                    "home": team.get("home", ""),
                    "away": team.get("away", ""),
                    "division": div_data["div_name"],
                })
        return result
    except Exception as e:
        logger.error("Failed to fetch standings: %s", e)
        return []


def get_team_stats(team_id: int, season: int = None) -> dict:
    """Fetch comprehensive team batting and pitching stats."""
    if season is None:
        season = datetime.now().year
    try:
        stats_url = f"{MLB_BASE_URL}/teams/{team_id}/stats"
        params = {
            "stats": "season",
            "group": "hitting,pitching",
            "season": season,
        }
        resp = requests.get(stats_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = {"team_id": team_id, "season": season}
        for stat_group in data.get("stats", []):
            group_name = stat_group["group"]["displayName"].lower()
            splits = stat_group.get("splits", [])
            if splits:
                stats = splits[0].get("stat", {})
                result[group_name] = stats
        return result
    except Exception as e:
        logger.error("Failed to fetch team stats for %d: %s", team_id, e)
        return {}


def get_pitcher_stats(player_id: int, season: int = None) -> dict:
    """Fetch detailed pitcher stats including recent form."""
    if season is None:
        season = datetime.now().year
    try:
        stats_url = f"{MLB_BASE_URL}/people/{player_id}/stats"
        params = {
            "stats": "season,gameLog",
            "group": "pitching",
            "season": season,
        }
        resp = requests.get(stats_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = {"player_id": player_id, "season": season}
        for stat_group in data.get("stats", []):
            stat_type = stat_group["type"]["displayName"]
            splits = stat_group.get("splits", [])
            if stat_type == "season" and splits:
                result["season_stats"] = splits[0].get("stat", {})
            elif stat_type == "gameLog":
                result["game_log"] = [s.get("stat", {}) for s in splits[:10]]
        return result
    except Exception as e:
        logger.error("Failed to fetch pitcher stats for %d: %s", player_id, e)
        return {}


def get_last_n_games(team_id: int, n: int = 10) -> list[dict]:
    """Get a team's last N game results for streak analysis."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    try:
        schedule = statsapi.schedule(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            teamId=team_id,
        )
        completed = [g for g in schedule if g.get("status") == "Final"]
        return completed[-n:]
    except Exception as e:
        logger.error("Failed to fetch game log for team %d: %s", team_id, e)
        return []


def compute_team_streak(team_id: int, team_name: str) -> dict:
    """Compute current win/loss streak and recent form."""
    games = get_last_n_games(team_id, n=15)
    if not games:
        return {"streak_type": "unknown", "streak_count": 0, "last_10": "N/A"}

    results = []
    for g in games:
        if g["home_id"] == team_id:
            results.append("W" if g["home_score"] > g["away_score"] else "L")
        else:
            results.append("W" if g["away_score"] > g["home_score"] else "L")

    # Current streak
    streak_type = results[-1]
    streak_count = 0
    for r in reversed(results):
        if r == streak_type:
            streak_count += 1
        else:
            break

    # Last 10
    last_10 = results[-10:]
    wins_10 = last_10.count("W")

    return {
        "streak_type": streak_type,
        "streak_count": streak_count,
        "last_10": f"{wins_10}-{10 - wins_10}",
        "results": results,
    }


def compute_pitcher_recent_era(game_log: list[dict], last_n: int = 5) -> float:
    """Calculate a pitcher's ERA over their last N starts."""
    if not game_log:
        return None
    recent = game_log[:last_n]
    total_er = sum(float(g.get("earnedRuns", 0)) for g in recent)
    total_ip = sum(float(g.get("inningsPitched", 0)) for g in recent)
    if total_ip == 0:
        return None
    return round((total_er / total_ip) * 9, 2)


def get_head_to_head(home_team_id: int, away_team_id: int, seasons: int = 2) -> dict:
    """Get head-to-head record between two teams."""
    current_year = datetime.now().year
    h2h_wins = {home_team_id: 0, away_team_id: 0}

    for season in range(current_year - seasons + 1, current_year + 1):
        try:
            url = f"{MLB_BASE_URL}/schedule"
            params = {
                "teamId": home_team_id,
                "opponentId": away_team_id,
                "season": season,
                "gameType": "R",
                "sportId": 1,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for date_entry in data.get("dates", []):
                for game in date_entry.get("games", []):
                    if game.get("status", {}).get("abstractGameState") == "Final":
                        teams = game.get("teams", {})
                        home = teams.get("home", {})
                        away = teams.get("away", {})
                        if home.get("score", 0) > away.get("score", 0):
                            h2h_wins[home["team"]["id"]] = h2h_wins.get(home["team"]["id"], 0) + 1
                        else:
                            h2h_wins[away["team"]["id"]] = h2h_wins.get(away["team"]["id"], 0) + 1
        except Exception as e:
            logger.warning("H2H lookup failed for season %d: %s", season, e)

    total = sum(h2h_wins.values())
    return {
        "home_wins": h2h_wins.get(home_team_id, 0),
        "away_wins": h2h_wins.get(away_team_id, 0),
        "total_games": total,
    }


def get_team_id_map() -> dict[str, int]:
    """Build a map of team name -> team ID."""
    try:
        resp = requests.get(f"{MLB_BASE_URL}/teams", params={"sportId": 1}, timeout=10)
        resp.raise_for_status()
        teams = resp.json().get("teams", [])
        return {t["name"]: t["id"] for t in teams}
    except Exception as e:
        logger.error("Failed to fetch team ID map: %s", e)
        return {}


def get_ballpark_factors() -> dict[str, dict]:
    """Park factors — affects run environment and totals significantly."""
    # Based on multi-year historical park factor data
    # Scale: 100 = neutral, >100 = hitter-friendly, <100 = pitcher-friendly
    return {
        "Colorado Rockies": {"runs": 115, "hr": 121, "hits": 112, "note": "High altitude, extreme hitter park"},
        "Cincinnati Reds": {"runs": 107, "hr": 112, "hits": 106, "note": "Hitter friendly"},
        "Boston Red Sox": {"runs": 105, "hr": 103, "hits": 108, "note": "Fenway hitter friendly"},
        "Chicago Cubs": {"runs": 104, "hr": 107, "hits": 103, "note": "Wind-dependent"},
        "Philadelphia Phillies": {"runs": 103, "hr": 106, "hits": 102, "note": "Slight hitter park"},
        "New York Yankees": {"runs": 103, "hr": 112, "hits": 101, "note": "Short porch RF"},
        "Atlanta Braves": {"runs": 102, "hr": 105, "hits": 101, "note": "Slightly hitter friendly"},
        "Los Angeles Angels": {"runs": 101, "hr": 103, "hits": 101, "note": "Near neutral"},
        "Houston Astros": {"runs": 100, "hr": 100, "hits": 100, "note": "Neutral (roof)"},
        "New York Mets": {"runs": 99, "hr": 97, "hits": 99, "note": "Slightly pitcher friendly"},
        "Los Angeles Dodgers": {"runs": 98, "hr": 96, "hits": 98, "note": "Pitcher friendly"},
        "San Francisco Giants": {"runs": 96, "hr": 91, "hits": 97, "note": "Wind, cold nights"},
        "Oakland Athletics": {"runs": 96, "hr": 94, "hits": 97, "note": "Large foul territory"},
        "San Diego Padres": {"runs": 95, "hr": 93, "hits": 96, "note": "Marine layer, pitcher park"},
        "Seattle Mariners": {"runs": 95, "hr": 93, "hits": 96, "note": "Marine layer (roof retractable)"},
        "Tampa Bay Rays": {"runs": 94, "hr": 92, "hits": 95, "note": "Dome, pitcher park"},
        "Miami Marlins": {"runs": 94, "hr": 91, "hits": 95, "note": "Dome, pitcher park"},
        "Minnesota Twins": {"runs": 98, "hr": 100, "hits": 98, "note": "Near neutral"},
        "Detroit Tigers": {"runs": 97, "hr": 97, "hits": 97, "note": "Near neutral"},
        "Cleveland Guardians": {"runs": 97, "hr": 95, "hits": 98, "note": "Slight pitcher park"},
        "Kansas City Royals": {"runs": 98, "hr": 96, "hits": 99, "note": "Near neutral"},
        "Milwaukee Brewers": {"runs": 99, "hr": 101, "hits": 99, "note": "Roof, near neutral"},
        "St. Louis Cardinals": {"runs": 100, "hr": 99, "hits": 100, "note": "Neutral"},
        "Pittsburgh Pirates": {"runs": 99, "hr": 99, "hits": 99, "note": "Near neutral"},
        "Chicago White Sox": {"runs": 99, "hr": 100, "hits": 99, "note": "Near neutral"},
        "Baltimore Orioles": {"runs": 101, "hr": 105, "hits": 100, "note": "Slight hitter park"},
        "Washington Nationals": {"runs": 100, "hr": 101, "hits": 100, "note": "Neutral"},
        "Toronto Blue Jays": {"runs": 101, "hr": 103, "hits": 101, "note": "Dome, slight hitter"},
        "Arizona Diamondbacks": {"runs": 102, "hr": 104, "hits": 101, "note": "Heat, dome, hitter friendly"},
        "Texas Rangers": {"runs": 99, "hr": 100, "hits": 99, "note": "Neutral (new dome)"},
    }
