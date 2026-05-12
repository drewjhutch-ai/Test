"""
Auto-retraining scheduler.
Monitors the DB for newly completed games and triggers model retraining
when enough fresh data has accumulated. This is what makes the system
continuously evolving — the model improves after every game day.
"""
import logging
from datetime import datetime, timedelta
from ..database import get_db
from .prediction_model import MLBPredictor

logger = logging.getLogger(__name__)

RETRAIN_THRESHOLD = 10   # Retrain after this many new completed games
MIN_TRAINING_GAMES = 50  # Absolute minimum to train


def get_completed_games_since_last_train() -> int:
    """Count games completed since the last model training run."""
    with get_db() as conn:
        last_train = conn.execute("""
            SELECT MAX(updated_at) as last_train FROM model_performance
        """).fetchone()
        last_train_time = (last_train["last_train"] if last_train and last_train["last_train"]
                           else "2000-01-01")

        count = conn.execute("""
            SELECT COUNT(*) as cnt FROM games
            WHERE status='Final'
            AND updated_at >= ?
        """, (last_train_time,)).fetchone()
        return count["cnt"] if count else 0


def maybe_retrain(predictor: MLBPredictor, force: bool = False) -> dict | None:
    """
    Check if retraining is warranted and run it if so.
    Returns training metrics if retrain occurred, None otherwise.
    """
    new_games = get_completed_games_since_last_train()

    if not force and new_games < RETRAIN_THRESHOLD:
        logger.debug("Only %d new games since last train (need %d). Skipping.",
                     new_games, RETRAIN_THRESHOLD)
        return None

    logger.info("Triggering model retrain: %d new completed games.", new_games)
    training_data = predictor.collect_training_data()

    if len(training_data) < MIN_TRAINING_GAMES:
        logger.info("Not enough total training data yet (%d games, need %d).",
                    len(training_data), MIN_TRAINING_GAMES)
        return None

    metrics = predictor.train(training_data)
    logger.info("Retrain complete: %s", metrics)
    return metrics


def update_bet_results(predictor: MLBPredictor):
    """
    After games complete, mark value bets as won/lost
    and compute actual profit/loss for ROI tracking.
    """
    with get_db() as conn:
        pending_bets = conn.execute("""
            SELECT vb.*, g.home_score, g.away_score, g.status
            FROM value_bets vb
            JOIN games g ON vb.game_id = g.game_id
            WHERE vb.result IS NULL
            AND g.status = 'Final'
            AND g.home_score IS NOT NULL
        """).fetchall()

    updated = 0
    for bet in pending_bets:
        bet = dict(bet)
        home_score = bet.get("home_score", 0)
        away_score = bet.get("away_score", 0)
        side = bet.get("side")
        market = bet.get("market")
        price = bet.get("book_price", -110)

        won = _determine_bet_result(bet, home_score, away_score)
        if won is None:
            continue

        if won:
            if price > 0:
                profit = bet.get("recommended_bet", 0) * price / 100
            else:
                profit = bet.get("recommended_bet", 0) * 100 / abs(price)
        else:
            profit = -(bet.get("recommended_bet", 0))

        with get_db() as conn:
            conn.execute("""
                UPDATE value_bets
                SET result=?, profit_loss=?
                WHERE id=?
            """, ("WIN" if won else "LOSS", round(profit, 2), bet["id"]))
        updated += 1

    if updated:
        logger.info("Graded %d completed bets.", updated)
    return updated


def _determine_bet_result(bet: dict, home_score: int, away_score: int) -> bool | None:
    """Determine if a bet won based on game result."""
    side = bet.get("side", "")
    market = bet.get("market", "")

    if market == "h2h":
        if side == "home":
            return home_score > away_score
        elif side == "away":
            return away_score > home_score

    elif market == "totals":
        total = home_score + away_score
        book_total = bet.get("book_price")  # stored differently — look up from DB
        if book_total:
            if side == "over":
                return total > book_total
            elif side == "under":
                return total < book_total

    return None


def compute_roi_summary() -> dict:
    """Calculate overall ROI and performance metrics on graded bets."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_bets,
                SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(recommended_bet) as total_staked,
                SUM(profit_loss) as total_profit,
                AVG(edge) as avg_edge,
                AVG(CASE WHEN result='WIN' THEN 1.0 ELSE 0.0 END) as hit_rate
            FROM value_bets
            WHERE result IS NOT NULL
        """).fetchone()

    if not row or not row["total_bets"]:
        return {"status": "no_graded_bets"}

    r = dict(row)
    total_staked = r.get("total_staked") or 0
    total_profit = r.get("total_profit") or 0
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    return {
        "total_bets": r["total_bets"],
        "wins": r["wins"],
        "losses": r["losses"],
        "hit_rate": round((r.get("hit_rate") or 0), 4),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 3),
        "avg_edge": round((r.get("avg_edge") or 0), 4),
    }
