import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# DB path: default is inside the repo data/ folder so it lives alongside the
# project.  Override with DB_PATH env var to use a mounted volume or external
# path.  On first run we also migrate data from the old ~/.sports_betting path
# if it exists there but not at the new default location.
_DEFAULT_DB = DATA_DIR / "betting.db"
_LEGACY_DB  = Path.home() / ".sports_betting" / "betting.db"

if not _DEFAULT_DB.exists() and _LEGACY_DB.exists():
    import shutil
    try:
        shutil.copy2(str(_LEGACY_DB), str(_DEFAULT_DB))
    except Exception:
        pass

# API Keys
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY", "")

# Database
DB_PATH = os.getenv("DB_PATH", str(_DEFAULT_DB))

# Risk Management
BANKROLL = float(os.getenv("BANKROLL", "1000"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", "0.05"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.03"))

# The Odds API config
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "baseball_mlb"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h,spreads,totals"
ODDS_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet", "pinnacle"]

# MLB StatsAPI
MLB_BASE_URL = "https://statsapi.mlb.com/api/v1"

# Weather API
WEATHER_BASE = "https://api.openweathermap.org/data/2.5"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# MLB Stadiums with coordinates for weather lookups
MLB_STADIUMS = {
    "New York Yankees": {"lat": 40.8296, "lon": -73.9262, "name": "Yankee Stadium", "roof": False},
    "New York Mets": {"lat": 40.7571, "lon": -73.8458, "name": "Citi Field", "roof": False},
    "Boston Red Sox": {"lat": 42.3467, "lon": -71.0972, "name": "Fenway Park", "roof": False},
    "Chicago Cubs": {"lat": 41.9484, "lon": -87.6553, "name": "Wrigley Field", "roof": False},
    "Chicago White Sox": {"lat": 41.8300, "lon": -87.6339, "name": "Guaranteed Rate Field", "roof": False},
    "Los Angeles Dodgers": {"lat": 34.0739, "lon": -118.2400, "name": "Dodger Stadium", "roof": False},
    "Los Angeles Angels": {"lat": 33.8003, "lon": -117.8827, "name": "Angel Stadium", "roof": False},
    "San Francisco Giants": {"lat": 37.7786, "lon": -122.3893, "name": "Oracle Park", "roof": False},
    "Oakland Athletics": {"lat": 37.7516, "lon": -122.2005, "name": "Oakland Coliseum", "roof": False},
    "Athletics":         {"lat": 38.5799, "lon": -121.5026, "name": "Sutter Health Park", "roof": False},
    "Seattle Mariners": {"lat": 47.5914, "lon": -122.3325, "name": "T-Mobile Park", "roof": True},
    "Houston Astros": {"lat": 29.7573, "lon": -95.3555, "name": "Minute Maid Park", "roof": True},
    "Texas Rangers": {"lat": 32.7473, "lon": -97.0822, "name": "Globe Life Field", "roof": True},
    "Arizona Diamondbacks": {"lat": 33.4453, "lon": -112.0667, "name": "Chase Field", "roof": True},
    "Colorado Rockies": {"lat": 39.7560, "lon": -104.9942, "name": "Coors Field", "roof": False},
    "Minnesota Twins": {"lat": 44.9817, "lon": -93.2778, "name": "Target Field", "roof": False},
    "Detroit Tigers": {"lat": 42.3390, "lon": -83.0485, "name": "Comerica Park", "roof": False},
    "Cleveland Guardians": {"lat": 41.4962, "lon": -81.6852, "name": "Progressive Field", "roof": False},
    "Kansas City Royals": {"lat": 39.0517, "lon": -94.4803, "name": "Kauffman Stadium", "roof": False},
    "Milwaukee Brewers": {"lat": 43.0280, "lon": -87.9712, "name": "American Family Field", "roof": True},
    "St. Louis Cardinals": {"lat": 38.6226, "lon": -90.1928, "name": "Busch Stadium", "roof": False},
    "Pittsburgh Pirates": {"lat": 40.4469, "lon": -80.0057, "name": "PNC Park", "roof": False},
    "Cincinnati Reds": {"lat": 39.0979, "lon": -84.5082, "name": "Great American Ball Park", "roof": False},
    "Atlanta Braves": {"lat": 33.8908, "lon": -84.4678, "name": "Truist Park", "roof": False},
    "Miami Marlins": {"lat": 25.7781, "lon": -80.2197, "name": "LoanDepot Park", "roof": True},
    "Tampa Bay Rays": {"lat": 27.7683, "lon": -82.6534, "name": "Tropicana Field", "roof": True},
    "Baltimore Orioles": {"lat": 39.2838, "lon": -76.6218, "name": "Oriole Park", "roof": False},
    "Washington Nationals": {"lat": 38.8730, "lon": -77.0074, "name": "Nationals Park", "roof": False},
    "Philadelphia Phillies": {"lat": 39.9061, "lon": -75.1665, "name": "Citizens Bank Park", "roof": False},
    "Toronto Blue Jays": {"lat": 43.6414, "lon": -79.3894, "name": "Rogers Centre", "roof": True},
    "San Diego Padres": {"lat": 32.7076, "lon": -117.1570, "name": "Petco Park", "roof": False},
}

# Sharp money thresholds
SHARP_LINE_MOVE_THRESHOLD = 0.5   # cents on moneyline
SHARP_TICKET_PCT_THRESHOLD = 0.30  # sharp side if <30% of tickets but >55% of money
SHARP_MONEY_PCT_THRESHOLD = 0.55

# Streak thresholds
HOT_STREAK_WINS = 5
COLD_STREAK_LOSSES = 4
PITCHER_HOT_ERA_THRESHOLD = 2.50
PITCHER_COLD_ERA_THRESHOLD = 5.00
