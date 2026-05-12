"""
Arbitrage and positive expected value opportunity detector.
Finds discrepancies across sportsbooks that guarantee profit
or significantly exceed true probability.
"""
import logging
from itertools import combinations
from ..database import save_arb_opportunity
from ..collectors.odds_collector import american_to_decimal, decimal_to_implied_prob

logger = logging.getLogger(__name__)


def find_arbitrage(parsed_games: list[dict]) -> list[dict]:
    """
    Scan all games for true arbitrage opportunities across books.
    Arbitrage exists when 1/odds_a + 1/odds_b < 1.0
    """
    arb_opportunities = []

    for game in parsed_games:
        game_id = game["game_id"]
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        odds_by_book = game.get("odds_by_book", {})

        if len(odds_by_book) < 2:
            continue

        arbs = _check_h2h_arb(game_id, home_team, away_team, odds_by_book)
        arbs += _check_totals_arb(game_id, odds_by_book)

        arb_opportunities.extend(arbs)

    if arb_opportunities:
        logger.info("Found %d arbitrage opportunities", len(arb_opportunities))

    return arb_opportunities


def _check_h2h_arb(game_id: str, home: str, away: str, odds_by_book: dict) -> list[dict]:
    """Find H2H arbitrage across all book pairs."""
    arbs = []
    book_list = list(odds_by_book.keys())

    for book_a, book_b in combinations(book_list, 2):
        h2h_a = odds_by_book[book_a].get("h2h", {})
        h2h_b = odds_by_book[book_b].get("h2h", {})

        if not h2h_a or not h2h_b:
            continue

        home_a = h2h_a.get("home_price")
        away_a = h2h_a.get("away_price")
        home_b = h2h_b.get("home_price")
        away_b = h2h_b.get("away_price")

        if not all([home_a, away_a, home_b, away_b]):
            continue

        # Best prices: max decimal odds for each side
        best_home_price = max(home_a, home_b)
        best_home_book = book_a if home_a >= home_b else book_b
        best_away_price = max(away_a, away_b)
        best_away_book = book_a if away_a >= away_b else book_b

        if best_home_book == best_away_book:
            continue  # Same book can't arb

        home_dec = american_to_decimal(best_home_price)
        away_dec = american_to_decimal(best_away_price)

        arb_pct = (1 / home_dec) + (1 / away_dec)
        profit_pct = (1 - arb_pct) * 100

        if arb_pct < 1.0:
            arb = {
                "game_id": game_id,
                "market": "h2h",
                "book_a": best_home_book,
                "book_b": best_away_book,
                "side_a": home,
                "side_b": away,
                "price_a": best_home_price,
                "price_b": best_away_price,
                "arb_pct": round(arb_pct, 4),
                "profit_pct": round(profit_pct, 3),
                "home_team": home,
                "away_team": away,
            }
            arbs.append(arb)
            save_arb_opportunity(arb)
            logger.info(
                "ARB FOUND: %s vs %s | %s (%.0f) @ %s + %s (%.0f) @ %s = %.2f%% profit",
                home, away,
                home, best_home_price, best_home_book,
                away, best_away_price, best_away_book,
                profit_pct
            )

    return arbs


def _check_totals_arb(game_id: str, odds_by_book: dict) -> list[dict]:
    """Find totals arbitrage where books have different lines that create a gap."""
    arbs = []
    book_list = list(odds_by_book.keys())

    for book_a, book_b in combinations(book_list, 2):
        tot_a = odds_by_book[book_a].get("totals", {})
        tot_b = odds_by_book[book_b].get("totals", {})

        if not tot_a or not tot_b:
            continue

        total_a = tot_a.get("total")
        total_b = tot_b.get("total")

        # Different totals create a middle opportunity
        if total_a and total_b and total_a != total_b:
            over_price = tot_a.get("over_price") if total_a > total_b else tot_b.get("over_price")
            under_price = tot_b.get("under_price") if total_a > total_b else tot_a.get("under_price")
            over_book = book_a if total_a > total_b else book_b
            under_book = book_b if total_a > total_b else book_a
            high_total = max(total_a, total_b)
            low_total = min(total_a, total_b)

            if over_price and under_price:
                over_dec = american_to_decimal(over_price)
                under_dec = american_to_decimal(under_price)
                arb_pct = (1 / over_dec) + (1 / under_dec)
                profit_pct = (1 - arb_pct) * 100

                if arb_pct < 1.0:
                    arb = {
                        "game_id": game_id,
                        "market": "totals_middle",
                        "book_a": over_book,
                        "book_b": under_book,
                        "side_a": f"Over {low_total}",
                        "side_b": f"Under {high_total}",
                        "price_a": over_price,
                        "price_b": under_price,
                        "arb_pct": round(arb_pct, 4),
                        "profit_pct": round(profit_pct, 3),
                        "middle_window": high_total - low_total,
                    }
                    arbs.append(arb)
                    save_arb_opportunity(arb)

    return arbs


def find_middle_opportunities(parsed_games: list[dict], min_middle_size: float = 1.0) -> list[dict]:
    """
    Middles: bet both sides of a spread where the line has moved,
    creating a window where BOTH bets can win.
    """
    middles = []

    for game in parsed_games:
        game_id = game["game_id"]
        odds_by_book = game.get("odds_by_book", {})
        book_list = list(odds_by_book.keys())

        for book_a, book_b in combinations(book_list, 2):
            spread_a = odds_by_book[book_a].get("spreads", {})
            spread_b = odds_by_book[book_b].get("spreads", {})

            if not spread_a or not spread_b:
                continue

            home_line_a = spread_a.get("home_line")
            home_line_b = spread_b.get("home_line")

            if home_line_a is None or home_line_b is None:
                continue

            diff = abs(home_line_a - home_line_b)
            if diff >= min_middle_size:
                # There's a middle window
                high_spread = max(home_line_a, home_line_b)
                low_spread = min(home_line_a, home_line_b)

                home_price_a = spread_a.get("home_price", -110)
                away_price_b = spread_b.get("away_price", -110)

                middle = {
                    "game_id": game_id,
                    "market": "spreads_middle",
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "book_a": book_a,
                    "book_b": book_b,
                    "side_a": f"Home +{abs(low_spread)} @ {book_a}",
                    "side_b": f"Away +{abs(high_spread)} @ {book_b}",
                    "price_a": home_price_a,
                    "price_b": away_price_b,
                    "middle_window": diff,
                    "middle_probability": _estimate_middle_probability(diff),
                }
                middles.append(middle)

    return middles


def _estimate_middle_probability(window_size: float) -> float:
    """Estimate probability of hitting the middle based on window size."""
    # Historical data: ~10% per full point of spread movement
    return round(min(0.90, window_size * 0.10), 3)


def calculate_optimal_arb_stakes(total_bankroll: float, arb: dict) -> dict:
    """Calculate exact bet amounts to guarantee equal profit on both sides."""
    price_a = arb["price_a"]
    price_b = arb["price_b"]

    dec_a = american_to_decimal(price_a)
    dec_b = american_to_decimal(price_b)

    # Optimal allocation: stake_a / stake_b = dec_b / dec_a
    ratio = dec_b / dec_a
    stake_a = total_bankroll * ratio / (1 + ratio)
    stake_b = total_bankroll - stake_a

    profit = stake_a * dec_a - total_bankroll

    return {
        "stake_a": round(stake_a, 2),
        "stake_b": round(stake_b, 2),
        "guaranteed_profit": round(profit, 2),
        "roi": round(profit / total_bankroll * 100, 3),
    }
