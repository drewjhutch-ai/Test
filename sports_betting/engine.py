"""
Core analysis engine.
Orchestrates all collectors and analysis modules into a single pipeline:

  1. Fetch today's MLB schedule + probable pitchers
  2. Pull live odds from DraftKings + all books
  3. Get weather for every outdoor game
  4. Build team/pitcher streak profiles
  5. Run ML predictions
  6. Find value bets, arbitrage, and sharp money signals
  7. Update dashboard and reports
  8. Auto-retrain model on completed games
"""
import logging
from datetime import datetime

from .database import init_db, upsert_game, get_db
from .config import MLB_STADIUMS

from .collectors.mlb_collector import (
    get_todays_games, get_team_standings, get_team_id_map,
    get_last_n_games, compute_team_streak, get_pitcher_stats,
    get_head_to_head, get_ballpark_factors,
)
from .collectors.odds_collector import get_live_odds, parse_and_store_odds
from .collectors.weather_collector import get_game_weather, get_stored_weather, classify_weather_impact
from .collectors.sharp_money import detect_wiseguy_moves, get_consensus_sharp_side

from .analysis.streak_analyzer import (
    analyze_team_streak, analyze_pitcher_streak, compute_combined_team_advantage
)
from .analysis.arbitrage_detector import find_arbitrage, find_middle_opportunities
from .analysis.value_analyzer import find_value_bets
from .analysis.line_movement import get_line_movement_report

from .models.prediction_model import MLBPredictor
from .models.trainer import maybe_retrain, update_bet_results

logger = logging.getLogger(__name__)


class BettingEngine:
    def __init__(self):
        init_db()
        self.predictor = MLBPredictor()
        self.team_id_map = {}
        self.park_factors = get_ballpark_factors()

    # ------------------------------------------------------------------ #
    #  Full pipeline run                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """Execute the complete analysis pipeline and return all results."""
        logger.info("=== Engine run started at %s ===", datetime.now().isoformat())

        # Step 1 — schedule
        games = self._load_schedule()
        if not games:
            logger.warning("No games found for today.")
            return {"games": [], "value_bets": [], "arb": [], "sharp": [], "predictions": []}

        # Step 2 — odds
        parsed_games = self._load_odds(games)

        # Step 3 — weather
        self._load_weather(games)

        # Step 4 — MLB stats + streaks
        game_features_map = self._build_game_features(games)

        # Step 5 — predictions
        predictions = self._run_predictions(games, game_features_map)

        # Step 6 — value bets
        value_bets = self._find_value(parsed_games, predictions, game_features_map)

        # Step 7 — arbitrage + middles
        arb_opps = find_arbitrage(parsed_games)
        middles = find_middle_opportunities(parsed_games)

        # Step 8 — sharp money
        sharp_plays = detect_wiseguy_moves(parsed_games)

        # Step 9 — grade completed bets + maybe retrain
        update_bet_results(self.predictor)
        maybe_retrain(self.predictor)

        logger.info(
            "Run complete: %d games | %d value bets | %d arb | %d sharp signals",
            len(games), len(value_bets), len(arb_opps), len(sharp_plays)
        )

        return {
            "games": games,
            "parsed_games": parsed_games,
            "predictions": predictions,
            "value_bets": value_bets,
            "arb": arb_opps,
            "middles": middles,
            "sharp": sharp_plays,
            "run_at": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------ #
    #  Private pipeline steps                                              #
    # ------------------------------------------------------------------ #

    def _load_schedule(self) -> list[dict]:
        games = get_todays_games()
        for g in games:
            upsert_game(g)
        return games

    def _load_odds(self, games: list[dict]) -> list[dict]:
        raw_odds = get_live_odds()
        parsed = parse_and_store_odds(raw_odds)

        # Merge odds game IDs with schedule games by team name
        odds_by_teams = {
            (p["home_team"], p["away_team"]): p
            for p in parsed
        }
        for g in games:
            key = (g["home_team"], g["away_team"])
            odds_match = odds_by_teams.get(key)
            if odds_match:
                g["odds_game_id"] = odds_match["game_id"]
                g["odds_by_book"] = odds_match.get("odds_by_book", {})

        return parsed if parsed else games

    def _load_weather(self, games: list[dict]):
        for g in games:
            home = g["home_team"]
            game_id = g["game_id"]
            weather = get_game_weather(home, game_id)
            g["weather"] = weather

    def _build_game_features(self, games: list[dict]) -> dict:
        """Build the full feature vector for each game."""
        if not self.team_id_map:
            self.team_id_map = get_team_id_map()

        standings = get_team_standings()
        standings_by_name = {s["team_name"]: s for s in standings}

        features_map = {}
        for g in games:
            game_id = g["game_id"]
            home = g["home_team"]
            away = g["away_team"]

            home_stand = standings_by_name.get(home, {})
            away_stand = standings_by_name.get(away, {})

            # Win percentages
            home_w = home_stand.get("wins", 0)
            home_l = home_stand.get("losses", 1)
            away_w = away_stand.get("wins", 0)
            away_l = away_stand.get("losses", 1)
            home_wpct = home_w / max(1, home_w + home_l)
            away_wpct = away_w / max(1, away_w + away_l)

            # Streaks
            home_team_id = self.team_id_map.get(home)
            away_team_id = self.team_id_map.get(away)

            home_streak_raw = {}
            away_streak_raw = {}
            if home_team_id:
                home_streak_raw = compute_team_streak(home_team_id, home)
            if away_team_id:
                away_streak_raw = compute_team_streak(away_team_id, away)

            home_streak = analyze_team_streak(home_streak_raw)
            away_streak = analyze_team_streak(away_streak_raw)

            # Probable pitcher stats
            home_sp_name = g.get("home_probable_pitcher", "")
            away_sp_name = g.get("away_probable_pitcher", "")
            home_pitcher_analysis = {}
            away_pitcher_analysis = {}

            home_sp_stats = self._get_pitcher_by_name(home_sp_name, home)
            away_sp_stats = self._get_pitcher_by_name(away_sp_name, away)

            if home_sp_stats:
                home_pitcher_analysis = analyze_pitcher_streak(home_sp_stats)
            if away_sp_stats:
                away_pitcher_analysis = analyze_pitcher_streak(away_sp_stats)

            # Park factors
            park = self.park_factors.get(home, {"runs": 100, "hr": 100})

            # Weather
            weather = g.get("weather", {})
            weather_impact = weather.get("weather_impact_score", 0)
            is_dome = weather.get("roof_closed", False)
            temp = weather.get("temperature", 72)
            wind = weather.get("wind_speed", 5)

            # Head to head
            h2h = {}
            if home_team_id and away_team_id:
                h2h = get_head_to_head(home_team_id, away_team_id)
            h2h_total = max(1, h2h.get("total_games", 1))
            h2h_home_wpct = h2h.get("home_wins", 0) / h2h_total

            # Combined advantage
            advantage = compute_combined_team_advantage(
                home_streak, away_streak, home_pitcher_analysis, away_pitcher_analysis
            )

            features = {
                # Win probability inputs
                "home_win_pct": home_wpct,
                "away_win_pct": away_wpct,
                "home_run_diff_per_game": _safe_run_diff(home_stand),
                "away_run_diff_per_game": _safe_run_diff(away_stand),
                # Team pitching (approximate from standings ERA not always available)
                "home_team_era": 4.20,
                "away_team_era": 4.20,
                "home_bullpen_era": 4.00,
                "away_bullpen_era": 4.00,
                # SP stats
                "home_sp_era": _sp_stat(home_pitcher_analysis, "season_era", 4.50),
                "away_sp_era": _sp_stat(away_pitcher_analysis, "season_era", 4.50),
                "home_sp_whip": _sp_stat(home_pitcher_analysis, "season_whip", 1.30),
                "away_sp_whip": _sp_stat(away_pitcher_analysis, "season_whip", 1.30),
                "home_sp_k9": 8.5,
                "away_sp_k9": 8.5,
                "home_sp_recent_era": _sp_stat(home_pitcher_analysis, "last_5_era", 4.50),
                "away_sp_recent_era": _sp_stat(away_pitcher_analysis, "last_5_era", 4.50),
                # Streaks
                "home_momentum_score": home_streak.get("momentum_score", 0.5),
                "away_momentum_score": away_streak.get("momentum_score", 0.5),
                "home_streak_count": home_streak.get("streak_count", 0)
                    * (1 if home_streak.get("streak_type") == "W" else -1),
                "away_streak_count": away_streak.get("streak_count", 0)
                    * (1 if away_streak.get("streak_type") == "W" else -1),
                # Park
                "park_runs_factor": park.get("runs", 100),
                "park_hr_factor": park.get("hr", 100),
                # Weather
                "weather_impact_score": weather_impact,
                "temperature": temp,
                "wind_speed": wind,
                "is_home_dome": is_dome,
                # H2H
                "h2h_home_win_pct": h2h_home_wpct,
                "home_last10_wins": _last10_wins(home_streak_raw),
                "away_last10_wins": _last10_wins(away_streak_raw),
                # OPS placeholders (available if statsapi returns batting)
                "home_ops": 0.720,
                "away_ops": 0.720,
                # Meta
                "_home_streak": home_streak,
                "_away_streak": away_streak,
                "_home_pitcher": home_pitcher_analysis,
                "_away_pitcher": away_pitcher_analysis,
                "_advantage": advantage,
                "_weather_label": classify_weather_impact(weather_impact),
                "_park_note": park.get("note", ""),
            }

            features_map[game_id] = features

        return features_map

    def _run_predictions(self, games: list[dict], features_map: dict) -> list[dict]:
        predictions = []
        for g in games:
            game_id = g["game_id"]
            features = features_map.get(game_id, {})
            pred = self.predictor.predict(features)

            # Attach game context to prediction for display
            pred["game_id"] = game_id
            pred["game"] = f"{g['away_team']} @ {g['home_team']}"
            pred["home_team"] = g["home_team"]
            pred["away_team"] = g["away_team"]
            pred["game_date"] = g.get("game_date")
            pred["game_time"] = g.get("game_time")
            pred["home_probable_pitcher"] = g.get("home_probable_pitcher", "TBD")
            pred["away_probable_pitcher"] = g.get("away_probable_pitcher", "TBD")
            pred["weather"] = g.get("weather", {})
            pred["advantage"] = features.get("_advantage", {})
            pred["streak_notes"] = (
                features.get("_home_streak", {}).get("notes", []) +
                features.get("_away_streak", {}).get("notes", []) +
                features.get("_home_pitcher", {}).get("notes", []) +
                features.get("_away_pitcher", {}).get("notes", [])
            )

            # Save to DB
            import json
            with get_db() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_predictions
                    (game_id, model_version, home_win_prob, away_win_prob,
                     predicted_total, confidence, features)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_id,
                    pred.get("model_version", "heuristic"),
                    pred.get("home_win_prob"),
                    pred.get("away_win_prob"),
                    pred.get("predicted_total"),
                    pred.get("confidence"),
                    json.dumps(pred.get("factors", {})),
                ))

            predictions.append(pred)

        return predictions

    def _find_value(
        self, parsed_games: list[dict], predictions: list[dict], features_map: dict
    ) -> list[dict]:
        all_value_bets = []
        pred_by_game = {p["game_id"]: p for p in predictions}

        for game in parsed_games:
            game_id = game.get("game_id")
            pred = pred_by_game.get(game_id)

            # Also try matching via odds game ID in schedule
            if not pred:
                for p in predictions:
                    if p.get("game") and (
                        game.get("home_team") in p["game"] and
                        game.get("away_team") in p["game"]
                    ):
                        pred = p
                        break

            if not pred:
                continue

            vbets = find_value_bets(game, pred)
            all_value_bets.extend(vbets)

        return all_value_bets

    def _get_pitcher_by_name(self, name: str, team: str) -> dict | None:
        """Look up a pitcher's stats by name (searches DB or API)."""
        if not name:
            return None
        with get_db() as conn:
            row = conn.execute("""
                SELECT player_id FROM pitcher_stats
                WHERE player_name LIKE ? AND team=?
                ORDER BY stat_date DESC LIMIT 1
            """, (f"%{name}%", team)).fetchone()

        if row:
            return get_pitcher_stats(int(row["player_id"]))
        return None


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _safe_run_diff(standing: dict) -> float:
    rd = standing.get("run_diff", 0) or 0
    games = max(1, (standing.get("wins", 0) or 0) + (standing.get("losses", 0) or 0))
    return round(rd / games, 3)


def _sp_stat(pitcher_analysis: dict, key: str, default: float) -> float:
    val = pitcher_analysis.get(key)
    return float(val) if val is not None else default


def _last10_wins(streak_raw: dict) -> int:
    last_10 = streak_raw.get("last_10", "5-5")
    try:
        return int(str(last_10).split("-")[0])
    except (ValueError, AttributeError):
        return 5
