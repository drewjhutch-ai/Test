"""
Odds collector.
Primary source: ESPN public scoreboard API (free, no key, no cloud blocks).
Fallback: The Odds API (requires ODDS_API_KEY; 500 req/month on free tier).
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

_ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
_ESPN_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Normalise ESPN provider names → book keys used elsewhere in the model
_ESPN_BOOK_MAP = {
    "draftkings": "draftkings",
    "fanduel":    "fanduel",
    "betmgm":     "betmgm",
    "caesars":    "williamhill_us",
    "pointsbet":  "pointsbet",
    "barstool":   "barstool_sportsbook",
    "betrivers":  "betrivers",
}


def american_to_decimal(american: int) -> float:
    if american > 0:
        return round(1 + american / 100, 4)
    return round(1 + 100 / abs(american), 4)


def decimal_to_implied_prob(decimal: float) -> float:
    return round(1 / decimal, 4)


def american_to_implied_prob(american: int) -> float:
    return decimal_to_implied_prob(american_to_decimal(american))


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    return round(prob_a / total, 4), round(prob_b / total, 4)


# ------------------------------------------------------------------ #
#  ESPN primary source                                                 #
# ------------------------------------------------------------------ #

def _fetch_espn_odds() -> list[dict]:
    """
    Pull moneylines and totals from ESPN's public scoreboard API.
    No key required, not blocked on cloud IPs.
    Returns data in the same shape parse_and_store_odds expects.
    """
    try:
        resp = requests.get(_ESPN_URL, headers=_ESPN_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("ESPN odds fetch failed: %s", exc)
        return []

    events = []
    for event in data.get("events", []):
        comp = (event.get("competitions") or [{}])[0]

        home_team = away_team = ""
        for side in comp.get("competitors", []):
            name = side.get("team", {}).get("displayName", "")
            if side.get("homeAway") == "home":
                home_team = name
            else:
                away_team = name

        if not home_team or not away_team:
            continue

        bookmakers = []
        for odds in comp.get("odds", []):
            provider_name = odds.get("provider", {}).get("name", "espn").lower()
            # Normalise to a known book key; fall back to sanitised provider name
            book_key = _ESPN_BOOK_MAP.get(
                provider_name.replace(" ", "").replace("'", ""),
                provider_name.replace(" ", "_"),
            )

            home_ml  = odds.get("homeTeamOdds", {}).get("moneyLine")
            away_ml  = odds.get("awayTeamOdds", {}).get("moneyLine")
            ou_line  = odds.get("overUnder")
            over_odds  = odds.get("overOdds",  -110)
            under_odds = odds.get("underOdds", -110)

            markets = []
            if home_ml and away_ml:
                markets.append({
                    "key": "h2h",
                    "outcomes": [
                        {"name": home_team, "price": int(home_ml)},
                        {"name": away_team, "price": int(away_ml)},
                    ],
                })
            if ou_line:
                markets.append({
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over",  "price": int(over_odds  or -110), "point": ou_line},
                        {"name": "Under", "price": int(under_odds or -110), "point": ou_line},
                    ],
                })

            if markets:
                bookmakers.append({"key": book_key, "markets": markets})

        # If ESPN doesn't carry any provider line, synthesise a neutral "espn" entry
        # so the game still appears in the odds pipeline
        if not bookmakers:
            bookmakers.append({
                "key": "espn",
                "markets": [],
            })

        events.append({
            "id": str(event.get("id", "")),
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": comp.get("date", ""),
            "bookmakers": bookmakers,
        })

    logger.info("ESPN odds: %d games", len(events))
    return events


# ------------------------------------------------------------------ #
#  Public interface                                                     #
# ------------------------------------------------------------------ #

def _fetch_odds_api() -> list[dict]:
    """Fetch full-slate odds from The Odds API (needs ODDS_API_KEY)."""
    if not ODDS_API_KEY:
        return []
    try:
        url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds"
        params = {
            "apiKey":     ODDS_API_KEY,
            "regions":    ODDS_REGIONS,
            "markets":    "h2h,spreads,totals",
            "oddsFormat": "american",
            "bookmakers": ",".join(ODDS_BOOKS),
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        data = resp.json()
        logger.info("Odds API: %d games. Remaining quota: %s", len(data), remaining)
        return data
    except requests.exceptions.HTTPError as e:
        code = getattr(e.response, "status_code", "?")
        logger.error("Odds API HTTP %s (invalid key / quota / rate limit)", code)
        return []
    except Exception as e:
        logger.error("Odds API fetch failed: %s", e)
        return []


def _count_h2h(games: list[dict]) -> int:
    """How many games actually carry a two-way moneyline from any book."""
    n = 0
    for g in games:
        for bk in g.get("bookmakers", []):
            has = any(
                m.get("key") == "h2h" and len(m.get("outcomes", [])) >= 2
                for m in bk.get("markets", [])
            )
            if has:
                n += 1
                break
    return n


def get_live_odds() -> list[dict]:
    """
    Fetch live odds, preferring the source with the BEST moneyline coverage.

    The Odds API (when ODDS_API_KEY is set) covers the full slate with real
    books, so it is used as primary. ESPN's free scoreboard often carries
    moneylines for only a few games early in the day, so it is used only when
    there's no key — and only if it actually returns more h2h coverage than we'd
    otherwise have. Without any odds we do NOT fabricate mock lines (the model
    now refuses to bet without a real market).
    """
    # --- Primary: The Odds API (full coverage) when a key is configured ---
    if ODDS_API_KEY:
        api = _fetch_odds_api()
        if _count_h2h(api) > 0:
            return api
        logger.warning("Odds API returned no usable h2h — falling back to ESPN.")

    # --- ESPN (free, partial coverage) ---
    espn = _fetch_espn_odds()
    espn_h2h = _count_h2h(espn)
    if espn_h2h > 0:
        logger.info("Using ESPN odds: %d/%d games have a moneyline.", espn_h2h, len(espn))
        return espn

    logger.warning(
        "No usable moneylines from any source (ODDS_API_KEY set: %s). "
        "Games will be skipped as 'no market'. Set ODDS_API_KEY for full coverage.",
        bool(ODDS_API_KEY),
    )
    return espn or []



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
