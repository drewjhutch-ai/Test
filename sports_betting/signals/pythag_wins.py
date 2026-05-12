"""
Signal 2: Pythagorean Win% & Luck Score
Teams 4+ games above/below their run-differential expected record are regression candidates.
One of the most reliable signals in baseball analytics.
"""
from __future__ import annotations
import math
import logging

logger = logging.getLogger(__name__)

PYTHAG_EXP = 1.83  # Bill James exponent for baseball


def compute_pythag_wpct(runs_scored: float, runs_allowed: float) -> float:
    """Expected win% based on run differential. Pythagoras formula."""
    if runs_scored <= 0 and runs_allowed <= 0:
        return 0.500
    rs = max(0.1, runs_scored)
    ra = max(0.1, runs_allowed)
    return round(rs ** PYTHAG_EXP / (rs ** PYTHAG_EXP + ra ** PYTHAG_EXP), 4)


def compute_luck_score(
    actual_wins: int, actual_losses: int,
    runs_scored: float, runs_allowed: float,
) -> dict:
    """
    Compare actual W-L to Pythagorean expected W-L.
    Positive luck score = team is outperforming their run differential (lucky, expect regression).
    Negative luck score = team is underperforming (unlucky, expect bounce-back).
    """
    total_games = actual_wins + actual_losses
    if total_games < 10:
        return {"luck_score": 0, "label": "insufficient_data", "note": "Need 10+ games", "pythag_wins": 0, "actual_wins": actual_wins}

    pythag_wpct = compute_pythag_wpct(runs_scored, runs_allowed)
    pythag_wins = round(pythag_wpct * total_games, 1)
    luck_score = round(actual_wins - pythag_wins, 1)

    if luck_score >= 5:
        label = "very_lucky"
        note = f"⚠️ Running {luck_score:.1f} wins LUCKY — regression expected. Fade in close games."
    elif luck_score >= 3:
        label = "lucky"
        note = f"Running {luck_score:.1f} wins above expectation — slight fade signal"
    elif luck_score <= -5:
        label = "very_unlucky"
        note = f"⭐ Running {abs(luck_score):.1f} wins UNLUCKY — buy-low opportunity. Back in value spots."
    elif luck_score <= -3:
        label = "unlucky"
        note = f"Running {abs(luck_score):.1f} wins below expectation — back signal"
    else:
        label = "neutral"
        note = f"W-L aligned with run differential (diff: {luck_score:+.1f})"

    return {
        "actual_wins": actual_wins,
        "actual_losses": actual_losses,
        "pythag_wins": pythag_wins,
        "pythag_wpct": pythag_wpct,
        "luck_score": luck_score,
        "label": label,
        "note": note,
    }


def get_luck_scores_for_game(
    home_team: str, away_team: str,
    home_standing: dict, away_standing: dict,
) -> dict:
    """Return luck scores for both teams."""
    def _extract(standing):
        return {
            "wins": standing.get("wins", 0) or 0,
            "losses": standing.get("losses", 0) or 0,
            "runs_scored": standing.get("runs_scored", 0) or 0,
            "runs_allowed": standing.get("runs_allowed", 0) or 0,
        }

    h = _extract(home_standing)
    a = _extract(away_standing)

    home_luck = compute_luck_score(h["wins"], h["losses"], h["runs_scored"], h["runs_allowed"])
    away_luck = compute_luck_score(a["wins"], a["losses"], a["runs_scored"], a["runs_allowed"])

    return {
        "home": {**home_luck, "team": home_team},
        "away": {**away_luck, "team": away_team},
    }
