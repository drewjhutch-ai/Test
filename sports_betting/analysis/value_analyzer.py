"""
Expected value and value bet finder.
Compares model-derived probabilities to bookmaker lines
and identifies +EV opportunities across all markets.
"""
import logging
import json
from ..config import MIN_EDGE, BANKROLL, MAX_BET_PCT
from ..database import save_value_bet
from ..collectors.odds_collector import american_to_implied_prob, remove_vig

logger = logging.getLogger(__name__)


def kelly_criterion(edge: float, odds: int, fraction: float = 0.25) -> float:
    """
    Fractional Kelly Criterion for bet sizing.
    Uses 1/4 Kelly for safety (reduces variance significantly).
    Returns fraction of bankroll to bet.
    """
    dec_odds = _american_to_decimal(odds)
    b = dec_odds - 1  # Net profit per unit staked
    p = edge + (1 / dec_odds)  # Our estimated win probability
    q = 1 - p

    full_kelly = (b * p - q) / b
    return max(0.0, round(full_kelly * fraction, 4))


def find_value_bets(game: dict, model_prediction: dict) -> list[dict]:
    """
    Compare model win probabilities against bookmaker lines.
    Flag any bet where our edge exceeds MIN_EDGE threshold.
    """
    value_bets = []
    game_id = game["game_id"]
    home_win_prob = model_prediction.get("home_win_prob", 0.5)
    away_win_prob = model_prediction.get("away_win_prob", 0.5)
    predicted_total = model_prediction.get("predicted_total")
    confidence = model_prediction.get("confidence", 0.5)

    odds_by_book = game.get("odds_by_book", {})

    for book, markets in odds_by_book.items():
        # H2H value check
        h2h = markets.get("h2h", {})
        if h2h:
            home_price = h2h.get("home_price")
            away_price = h2h.get("away_price")

            if home_price:
                implied = american_to_implied_prob(home_price)
                edge = home_win_prob - implied
                if edge >= MIN_EDGE:
                    bet = _build_value_bet(
                        game_id, book, "h2h", "home",
                        home_price, home_win_prob, implied, edge,
                        confidence, model_prediction.get("factors", {})
                    )
                    value_bets.append(bet)
                    save_value_bet(bet)

            if away_price:
                implied = american_to_implied_prob(away_price)
                edge = away_win_prob - implied
                if edge >= MIN_EDGE:
                    bet = _build_value_bet(
                        game_id, book, "h2h", "away",
                        away_price, away_win_prob, implied, edge,
                        confidence, model_prediction.get("factors", {})
                    )
                    value_bets.append(bet)
                    save_value_bet(bet)

        # Totals value check
        totals = markets.get("totals", {})
        if totals and predicted_total:
            book_total = totals.get("total")
            over_price = totals.get("over_price")
            under_price = totals.get("under_price")

            if book_total and over_price and under_price:
                # Estimate over/under probabilities from predicted total vs book total
                total_diff = predicted_total - book_total
                over_model_prob, under_model_prob = _estimate_total_probs(total_diff)

                if over_model_prob:
                    over_implied = american_to_implied_prob(over_price)
                    over_edge = over_model_prob - over_implied
                    if over_edge >= MIN_EDGE:
                        bet = _build_value_bet(
                            game_id, book, "totals", "over",
                            over_price, over_model_prob, over_implied, over_edge,
                            confidence * 0.85, model_prediction.get("factors", {})
                        )
                        value_bets.append(bet)
                        save_value_bet(bet)

                if under_model_prob:
                    under_implied = american_to_implied_prob(under_price)
                    under_edge = under_model_prob - under_implied
                    if under_edge >= MIN_EDGE:
                        bet = _build_value_bet(
                            game_id, book, "totals", "under",
                            under_price, under_model_prob, under_implied, under_edge,
                            confidence * 0.85, model_prediction.get("factors", {})
                        )
                        value_bets.append(bet)
                        save_value_bet(bet)

    return value_bets


def _build_value_bet(
    game_id, book, market, side, price, model_prob, implied_prob, edge,
    confidence, factors
) -> dict:
    kelly_frac = kelly_criterion(edge, price)
    kelly_frac = min(kelly_frac, MAX_BET_PCT)  # Cap at max bet %
    rec_bet = round(BANKROLL * kelly_frac, 2)

    confidence_label = "HIGH" if confidence >= 0.65 else "MEDIUM" if confidence >= 0.50 else "LOW"

    return {
        "game_id": game_id,
        "book": book,
        "market": market,
        "side": side,
        "book_price": price,
        "model_probability": round(model_prob, 4),
        "implied_probability": round(implied_prob, 4),
        "edge": round(edge, 4),
        "kelly_fraction": kelly_frac,
        "recommended_bet": rec_bet,
        "confidence": confidence_label,
        "factors": json.dumps(factors),
    }


def _estimate_total_probs(total_diff: float) -> tuple[float, float]:
    """
    Estimate over/under probability based on difference between
    model predicted total and book total.
    Simplified sigmoid: bigger diff = higher confidence.
    """
    import math
    if abs(total_diff) < 0.25:
        return 0.52, 0.48  # Slight lean
    # Sigmoid-ish mapping
    over_prob = 1 / (1 + math.exp(-total_diff * 0.8))
    return round(over_prob, 4), round(1 - over_prob, 4)


def _american_to_decimal(american: int) -> float:
    if american > 0:
        return 1 + american / 100
    return 1 + 100 / abs(american)


def calculate_closing_line_value(opening_price: int, closing_price: int, bet_price: int) -> float:
    """
    Closing Line Value (CLV): the gold standard for long-term betting edge.
    Positive CLV = you got better odds than the market closed at.
    """
    bet_prob = american_to_implied_prob(bet_price)
    closing_prob = american_to_implied_prob(closing_price)
    return round(closing_prob - bet_prob, 4)


def grade_bet(predicted_side_won: bool, edge: float, clv: float | None = None) -> dict:
    """Grade a historical bet for model performance tracking."""
    grade = {
        "result": "WIN" if predicted_side_won else "LOSS",
        "edge_was_positive": edge > 0,
        "clv": clv,
        "is_positive_clv": clv > 0 if clv is not None else None,
        "notes": [],
    }
    if edge > 0 and predicted_side_won:
        grade["notes"].append("Edge identified and bet won — model confirmed")
    elif edge > 0 and not predicted_side_won:
        grade["notes"].append("Edge identified, bet lost — expected variance")
    elif edge <= 0:
        grade["notes"].append("Warning: negative edge bet — review model")
    return grade


def compute_expected_value(probability: float, decimal_odds: float) -> float:
    """EV = (probability * profit) - ((1-probability) * stake)"""
    return round(probability * (decimal_odds - 1) - (1 - probability), 4)
