"""
Hot/cold streak detection for teams and pitchers.
Streaks are one of the most reliable short-term predictors.
"""
import logging
from ..config import (
    HOT_STREAK_WINS, COLD_STREAK_LOSSES,
    PITCHER_HOT_ERA_THRESHOLD, PITCHER_COLD_ERA_THRESHOLD
)

logger = logging.getLogger(__name__)


def analyze_team_streak(streak_data: dict) -> dict:
    """Classify a team's current streak with confidence and betting implications."""
    streak_type = streak_data.get("streak_type", "unknown")
    streak_count = streak_data.get("streak_count", 0)
    last_10 = streak_data.get("last_10", "N/A")
    results = streak_data.get("results", [])

    classification = {
        "streak_type": streak_type,
        "streak_count": streak_count,
        "last_10": last_10,
        "is_hot": False,
        "is_cold": False,
        "momentum_score": 0.0,
        "betting_lean": "NEUTRAL",
        "notes": [],
    }

    if streak_type == "W":
        if streak_count >= HOT_STREAK_WINS:
            classification["is_hot"] = True
            classification["betting_lean"] = "FAVOR"
            classification["notes"].append(f"HOT: Won {streak_count} straight")
        elif streak_count >= 3:
            classification["notes"].append(f"Winning streak: {streak_count} games")
    elif streak_type == "L":
        if streak_count >= COLD_STREAK_LOSSES:
            classification["is_cold"] = True
            classification["betting_lean"] = "FADE"
            classification["notes"].append(f"COLD: Lost {streak_count} straight")
        elif streak_count >= 3:
            classification["notes"].append(f"Losing streak: {streak_count} games")

    # Last-10 record analysis
    if last_10 != "N/A":
        try:
            parts = last_10.split("-")
            wins_10 = int(parts[0])
            if wins_10 >= 8:
                classification["is_hot"] = True
                classification["betting_lean"] = "FAVOR"
                classification["notes"].append(f"Elite recent form: {last_10} last 10")
            elif wins_10 <= 2:
                classification["is_cold"] = True
                classification["betting_lean"] = "FADE"
                classification["notes"].append(f"Poor recent form: {last_10} last 10")
        except (ValueError, IndexError):
            pass

    # Momentum score: weighted recency (more recent = more weight)
    if results:
        weights = [1.5 ** i for i in range(len(results))]
        weights = weights[::-1]  # Most recent gets highest weight
        total_weight = sum(weights)
        win_weight = sum(w for r, w in zip(results, weights) if r == "W")
        classification["momentum_score"] = round(win_weight / total_weight, 3)

    return classification


def analyze_pitcher_streak(pitcher_stats: dict) -> dict:
    """Evaluate a starting pitcher's recent form."""
    season_stats = pitcher_stats.get("season_stats", {})
    game_log = pitcher_stats.get("game_log", [])

    era = float(season_stats.get("era", 4.50) or 4.50)
    whip = float(season_stats.get("whip", 1.30) or 1.30)
    recent_era = _compute_recent_era(game_log, 5)
    recent_whip = _compute_recent_whip(game_log, 5)

    classification = {
        "season_era": era,
        "season_whip": whip,
        "last_5_era": recent_era,
        "last_5_whip": recent_whip,
        "is_hot": False,
        "is_cold": False,
        "trend": "STABLE",
        "betting_lean": "NEUTRAL",
        "notes": [],
    }

    if recent_era is not None:
        if recent_era <= PITCHER_HOT_ERA_THRESHOLD:
            classification["is_hot"] = True
            classification["trend"] = "HOT"
            classification["betting_lean"] = "FAVOR"
            classification["notes"].append(f"Pitcher HOT: {recent_era} ERA last 5 starts")
        elif recent_era >= PITCHER_COLD_ERA_THRESHOLD:
            classification["is_cold"] = True
            classification["trend"] = "COLD"
            classification["betting_lean"] = "FADE"
            classification["notes"].append(f"Pitcher COLD: {recent_era} ERA last 5 starts")

        # Trend vs season ERA
        if recent_era < era - 1.0:
            classification["notes"].append(f"ERA improving ({era:.2f} season -> {recent_era:.2f} recent)")
        elif recent_era > era + 1.0:
            classification["notes"].append(f"ERA deteriorating ({era:.2f} season -> {recent_era:.2f} recent)")

    # Strikeout trend
    recent_k9 = _compute_recent_k9(game_log, 5)
    season_k9 = float(season_stats.get("strikeoutsPer9Inn", 8.0) or 8.0)
    if recent_k9 and recent_k9 > season_k9 + 1.5:
        classification["notes"].append(f"K rate surging: {recent_k9:.1f} K/9 vs {season_k9:.1f} season")
    elif recent_k9 and recent_k9 < season_k9 - 1.5:
        classification["notes"].append(f"K rate dropping: {recent_k9:.1f} K/9 vs {season_k9:.1f} season")

    return classification


def _compute_recent_era(game_log: list[dict], last_n: int) -> float | None:
    if not game_log:
        return None
    recent = game_log[:last_n]
    total_er = sum(float(g.get("earnedRuns", 0) or 0) for g in recent)
    total_ip = sum(float(g.get("inningsPitched", 0) or 0) for g in recent)
    if total_ip < 1:
        return None
    return round((total_er / total_ip) * 9, 2)


def _compute_recent_whip(game_log: list[dict], last_n: int) -> float | None:
    if not game_log:
        return None
    recent = game_log[:last_n]
    total_walks = sum(int(g.get("baseOnBalls", 0) or 0) for g in recent)
    total_hits = sum(int(g.get("hits", 0) or 0) for g in recent)
    total_ip = sum(float(g.get("inningsPitched", 0) or 0) for g in recent)
    if total_ip < 1:
        return None
    return round((total_walks + total_hits) / total_ip, 3)


def _compute_recent_k9(game_log: list[dict], last_n: int) -> float | None:
    if not game_log:
        return None
    recent = game_log[:last_n]
    total_k = sum(int(g.get("strikeOuts", 0) or 0) for g in recent)
    total_ip = sum(float(g.get("inningsPitched", 0) or 0) for g in recent)
    if total_ip < 1:
        return None
    return round(total_k / total_ip * 9, 2)


def compute_combined_team_advantage(
    home_streak: dict,
    away_streak: dict,
    home_pitcher: dict | None,
    away_pitcher: dict | None,
) -> dict:
    """
    Combine team and pitcher streaks into a matchup advantage score.
    Returns a score from -1 (strong away advantage) to +1 (strong home advantage).
    """
    score = 0.0
    notes = []

    # Team streak contributions
    home_momentum = home_streak.get("momentum_score", 0.5)
    away_momentum = away_streak.get("momentum_score", 0.5)
    score += (home_momentum - away_momentum) * 0.4

    if home_streak.get("is_hot"):
        score += 0.15
        notes.append(f"Home team hot ({home_streak.get('streak_count')} W streak)")
    if home_streak.get("is_cold"):
        score -= 0.15
        notes.append(f"Home team cold ({home_streak.get('streak_count')} L streak)")
    if away_streak.get("is_hot"):
        score -= 0.15
        notes.append(f"Away team hot ({away_streak.get('streak_count')} W streak)")
    if away_streak.get("is_cold"):
        score += 0.15
        notes.append(f"Away team cold ({away_streak.get('streak_count')} L streak)")

    # Pitcher streak contributions
    if home_pitcher:
        if home_pitcher.get("is_hot"):
            score += 0.12
            notes.append(f"Home SP hot ({home_pitcher.get('last_5_era')} ERA)")
        elif home_pitcher.get("is_cold"):
            score -= 0.12
            notes.append(f"Home SP cold ({home_pitcher.get('last_5_era')} ERA)")

    if away_pitcher:
        if away_pitcher.get("is_hot"):
            score -= 0.12
            notes.append(f"Away SP hot ({away_pitcher.get('last_5_era')} ERA)")
        elif away_pitcher.get("is_cold"):
            score += 0.12
            notes.append(f"Away SP cold ({away_pitcher.get('last_5_era')} ERA)")

    score = max(-1.0, min(1.0, score))

    lean = "NEUTRAL"
    if score >= 0.2:
        lean = "HOME"
    elif score <= -0.2:
        lean = "AWAY"

    return {
        "advantage_score": round(score, 3),
        "lean": lean,
        "confidence": "HIGH" if abs(score) >= 0.4 else "MEDIUM" if abs(score) >= 0.2 else "LOW",
        "notes": notes,
    }
