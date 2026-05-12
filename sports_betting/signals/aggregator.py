"""
Signal Aggregator
Single entry point that runs all 13 signals for every game and returns
a unified signals dict. Called once per daily run.
"""
from __future__ import annotations
import logging
from datetime import date

logger = logging.getLogger(__name__)


def run_all_signals(
    games: list[dict],
    standings: dict,
    team_id_map: dict,
) -> dict[str, dict]:
    """
    Run all signals for all games.
    Returns: {game_key: {signal_name: result_dict}}
    game_key format: "Away Team @ Home Team"
    """
    from .bullpen_fatigue import get_bullpen_fatigue_for_game
    from .pythag_wins import get_luck_scores_for_game
    from .umpire_tracker import get_todays_umpires, get_umpire_tendency, get_totals_lean
    from .travel_tracker import get_travel_signals_for_game
    from .opener_detector import detect_opener_game

    # Fetch umpire assignments once for all games
    umpire_map = {}
    try:
        umpire_map = get_todays_umpires()
    except Exception as e:
        logger.debug("Umpire map fetch failed: %s", e)

    results = {}

    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        home_sp = game.get("home_probable_pitcher", "TBD")
        away_sp = game.get("away_probable_pitcher", "TBD")
        game_key = f"{away} @ {home}"

        signals = {}

        # Signal 1: Bullpen fatigue
        try:
            signals["bullpen"] = get_bullpen_fatigue_for_game(home, away, team_id_map)
        except Exception as e:
            logger.debug("Bullpen signal failed for %s: %s", game_key, e)
            signals["bullpen"] = {"home": {"score": 0, "note": "unavailable"}, "away": {"score": 0, "note": "unavailable"}}

        # Signal 2: Pythagorean luck scores
        try:
            home_stand = standings.get(home, {})
            away_stand = standings.get(away, {})
            signals["pythag"] = get_luck_scores_for_game(home, away, home_stand, away_stand)
        except Exception as e:
            logger.debug("Pythag signal failed: %s", e)
            signals["pythag"] = {"home": {"luck_score": 0, "label": "neutral"}, "away": {"luck_score": 0, "label": "neutral"}}

        # Signal 4: Umpire tendency
        try:
            ump_name = umpire_map.get(game_key, "")
            ump_data = get_umpire_tendency(ump_name)
            signals["umpire"] = {
                **ump_data,
                "lean": get_totals_lean(ump_data),
                "game": game_key,
            }
        except Exception as e:
            logger.debug("Umpire signal failed: %s", e)
            signals["umpire"] = {"name": "Unknown", "lean": "NEUTRAL", "over_rate": 0.5, "note": "unavailable"}

        # Signal 6: Travel fatigue
        try:
            signals["travel"] = get_travel_signals_for_game(home, away, team_id_map)
        except Exception as e:
            logger.debug("Travel signal failed: %s", e)
            signals["travel"] = {"home": {"travel_flag": False}, "away": {"travel_flag": False}}

        # Signal 9: Opener detection
        try:
            signals["opener"] = detect_opener_game(home_sp, away_sp, home, away)
        except Exception as e:
            logger.debug("Opener signal failed: %s", e)
            signals["opener"] = {"home_opener": False, "away_opener": False, "total_adjustment": 0.0, "flags": []}

        results[game_key] = signals

    return results


def build_signal_factors(
    game_key: str,
    backing_team: str,
    home_team: str,
    away_team: str,
    all_signals: dict,
) -> list[str]:
    """
    Convert signal results into human-readable factor strings for layer_11.
    Called from _build_factor_list in daily_runner.py.
    """
    factors = []
    signals = all_signals.get(game_key, {})
    is_home = backing_team == home_team

    # Signal 1: Bullpen fatigue
    bullpen = signals.get("bullpen", {})
    opp_side = "away" if is_home else "home"
    opp_bullpen = bullpen.get(opp_side, {})
    if opp_bullpen.get("score", 0) >= 0.60:
        factors.append(f"Opposing bullpen fatigued: {opp_bullpen.get('note','')}")
    elif opp_bullpen.get("score", 0) >= 0.40:
        factors.append(f"Opposing bullpen mildly stretched ({opp_bullpen.get('fatigue_level','')})")

    # Signal 2: Pythagorean luck
    pythag = signals.get("pythag", {})
    backing_pythag = pythag.get("home" if is_home else "away", {})
    opp_pythag = pythag.get("away" if is_home else "home", {})

    if backing_pythag.get("label") in ("unlucky", "very_unlucky"):
        factors.append(f"Pythagorean edge: {backing_team} running {abs(backing_pythag.get('luck_score',0)):.1f} wins UNLUCKY — buy-low value")
    if opp_pythag.get("label") in ("lucky", "very_lucky"):
        factors.append(f"Opponent {away_team if is_home else home_team} running {opp_pythag.get('luck_score',0):.1f} wins LUCKY — regression due")

    # Signal 4: Umpire
    umpire = signals.get("umpire", {})
    ump_lean = umpire.get("lean", "NEUTRAL")
    ump_rate = umpire.get("over_rate", 0.5)
    if ump_lean == "UNDER" and ump_rate <= 0.47:
        factors.append(f"Umpire {umpire.get('name','')} — large zone, pitcher-friendly (under lean {ump_rate:.0%})")
    elif ump_lean == "OVER" and ump_rate >= 0.54:
        factors.append(f"Umpire {umpire.get('name','')} — tight zone, hitter-friendly (over lean {ump_rate:.0%})")

    # Signal 6: Travel fatigue
    travel = signals.get("travel", {})
    opp_travel = travel.get("away" if is_home else "home", {})
    if opp_travel.get("travel_flag"):
        factors.append(f"Travel fade: {opp_travel.get('note','')}")

    # Signal 9: Opener
    opener = signals.get("opener", {})
    opp_opener_key = "away_opener" if is_home else "home_opener"
    if opener.get(opp_opener_key):
        factors.append(f"Opposing team using opener — bulk pitcher risk, total elevated +0.35")

    return factors
