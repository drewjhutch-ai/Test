"""
Odds collector using The Odds API.
Pulls live lines from DraftKings and multiple books simultaneously.
Tracks opening lines vs. current lines for movement analysis.
"""
import logging
import requests
from datetime import datetime
from ..config import (
    ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT, ODDS_REGIONS,
    ODDS_BOOKS, SHARP_LINE_MOVE_THRESHOLD
)
from ..database import (
    get_db, insert_odds_snapshot, get_latest_odds,
    save_arb_opportunity, save_sharp_play
)

logger = logging.getLogger(__name__)


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return round(1 + american / 100, 4)
    return round(1 + 100 / abs(american), 4)


def decimal_to_implied_prob(decimal: float) -> float:
    """Convert decimal odds to implied probability (no vig)."""
    return round(1 / decimal, 4)


def american_to_implied_prob(american: int) -> float:
    return decimal_to_implied_prob(american_to_decimal(american))


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove the bookmaker's vig to get fair probabilities."""
    total = prob_a + prob_b
    return round(prob_a / total, 4), round(prob_b / total, 4)


def get_live_odds() -> list[dict]:
    """Fetch live odds from The Odds API for all configured books."""
    if not ODDS_API_KEY:
        logger.warning("No ODDS_API_KEY set. Using mock data.")
        return _mock_odds()

    try:
        url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": ODDS_REGIONS,
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "bookmakers": ",".join(ODDS_BOOKS),
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()

        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.info("Odds API request succeeded. Remaining quota: %s", remaining)

        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            logger.error("Invalid Odds API key")
        elif e.response.status_code == 429:
            logger.error("Odds API rate limit exceeded")
        else:
            logger.error("Odds API HTTP error: %s", e)
        return []
    except Exception as e:
        logger.error("Failed to fetch odds: %s", e)
        return []


def parse_and_store_odds(raw_odds: list[dict]) -> list[dict]:
    """Parse raw odds API response, store snapshots, detect movements."""
    parsed_games = []

    for event in raw_odds:
        game_id = f"odds_{event['id']}"
        home_team = event["home_team"]
        away_team = event["away_team"]
        commence_time = event.get("commence_time", "")

        game_entry = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "game_date": commence_time[:10] if commence_time else "",
            "game_time": commence_time,
            "venue": "",
            "status": "scheduled",
            "odds_by_book": {},
        }

        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker["key"]
            book_odds = {"h2h": {}, "spreads": {}, "totals": {}}

            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}

                if market_key == "h2h":
                    book_odds["h2h"] = {
                        "home_price": outcomes.get(home_team),
                        "away_price": outcomes.get(away_team),
                    }
                elif market_key == "spreads":
                    for o in market.get("outcomes", []):
                        if o["name"] == home_team:
                            book_odds["spreads"]["home_line"] = o.get("point")
                            book_odds["spreads"]["home_price"] = o.get("price")
                        elif o["name"] == away_team:
                            book_odds["spreads"]["away_line"] = o.get("point")
                            book_odds["spreads"]["away_price"] = o.get("price")
                elif market_key == "totals":
                    for o in market.get("outcomes", []):
                        if o["name"] == "Over":
                            book_odds["totals"]["over_price"] = o.get("price")
                            book_odds["totals"]["total"] = o.get("point")
                        elif o["name"] == "Under":
                            book_odds["totals"]["under_price"] = o.get("price")

            game_entry["odds_by_book"][book_key] = book_odds

            # Store snapshot for each market
            for market_key, odds in book_odds.items():
                if odds:
                    snapshot = {
                        "game_id": game_id,
                        "book": book_key,
                        "market": market_key,
                        "home_line": odds.get("home_line"),
                        "away_line": odds.get("away_line"),
                        "home_price": odds.get("home_price"),
                        "away_price": odds.get("away_price"),
                        "over_price": odds.get("over_price"),
                        "under_price": odds.get("under_price"),
                        "total": odds.get("total"),
                    }
                    insert_odds_snapshot(snapshot)
                    _detect_line_movement(game_id, book_key, market_key, snapshot)

        parsed_games.append(game_entry)

    logger.info("Parsed and stored odds for %d games", len(parsed_games))
    return parsed_games


def _detect_line_movement(game_id: str, book: str, market: str, current: dict):
    """Compare current odds to previous snapshot and flag significant moves."""
    previous = get_latest_odds(game_id, book, market)
    if not previous:
        return

    movements = []
    if market == "h2h":
        for side, price_key in [("home", "home_price"), ("away", "away_price")]:
            old_price = previous.get(price_key)
            new_price = current.get(price_key)
            if old_price and new_price and old_price != new_price:
                move = abs(new_price - old_price)
                if move >= SHARP_LINE_MOVE_THRESHOLD:
                    movements.append({
                        "side": side,
                        "old_price": old_price,
                        "new_price": new_price,
                        "move": move,
                    })

    elif market == "spreads":
        old_line = previous.get("home_line")
        new_line = current.get("home_line")
        if old_line and new_line and old_line != new_line:
            movements.append({
                "side": "spread",
                "old_price": old_line,
                "new_price": new_line,
                "move": abs(new_line - old_line),
            })

    elif market == "totals":
        old_total = previous.get("total")
        new_total = current.get("total")
        if old_total and new_total and old_total != new_total:
            movements.append({
                "side": "total",
                "old_price": old_total,
                "new_price": new_total,
                "move": abs(new_total - old_total),
            })

    for m in movements:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO line_movements
                (game_id, book, market, old_value, new_value, old_price, new_price, side, sharp_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, book, market,
                m.get("old_price"), m.get("new_price"),
                m.get("old_price"), m.get("new_price"),
                m["side"], 1
            ))
        logger.info("Line movement detected: %s %s %s -> %s (%s)",
                    game_id, market, m["old_price"], m["new_price"], m["side"])


def get_consensus_odds(game_id: str, market: str = "h2h") -> dict:
    """Get consensus (average) odds across all books for a game."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT book, home_price, away_price, total, over_price, under_price
            FROM odds_snapshots
            WHERE game_id=? AND market=?
            AND snapshot_time >= datetime('now', '-2 hours')
            GROUP BY book
            HAVING MAX(snapshot_time)
        """, (game_id, market)).fetchall()

    if not rows:
        return {}

    books_data = [dict(r) for r in rows]
    home_prices = [r["home_price"] for r in books_data if r.get("home_price")]
    away_prices = [r["away_price"] for r in books_data if r.get("away_price")]

    result = {"books_count": len(books_data)}
    if home_prices:
        result["consensus_home_price"] = round(sum(home_prices) / len(home_prices), 1)
    if away_prices:
        result["consensus_away_price"] = round(sum(away_prices) / len(away_prices), 1)
    return result


def get_draftkings_odds(game_id: str, market: str = "h2h") -> dict | None:
    """Get the latest DraftKings line specifically."""
    return get_latest_odds(game_id, "draftkings", market)


def _mock_odds() -> list[dict]:
    """Mock odds data for testing without an API key."""
    today = datetime.now().strftime("%Y-%m-%dT19:05:00Z")
    return [
        {
            "id": "mock_game_1",
            "sport_key": "baseball_mlb",
            "sport_title": "MLB",
            "commence_time": today,
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "New York Yankees", "price": -140},
                                {"name": "Boston Red Sox", "price": 120},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -110, "point": 8.5},
                                {"name": "Under", "price": -110, "point": 8.5},
                            ],
                        },
                    ],
                },
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "New York Yankees", "price": -145},
                                {"name": "Boston Red Sox", "price": 122},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -108, "point": 8.5},
                                {"name": "Under", "price": -112, "point": 8.5},
                            ],
                        },
                    ],
                },
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "New York Yankees", "price": -138},
                                {"name": "Boston Red Sox", "price": 128},
                            ],
                        },
                    ],
                },
            ],
        }
    ]
