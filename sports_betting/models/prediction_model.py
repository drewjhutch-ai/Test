"""
ML prediction model for MLB game outcomes.
Uses XGBoost + feature engineering from stats, weather, streaks, and park factors.
Model continuously retrains as new results come in.
"""
import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import joblib

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from ..config import DATA_DIR
from ..database import get_db

logger = logging.getLogger(__name__)

MODEL_PATH = DATA_DIR / "models"
MODEL_PATH.mkdir(exist_ok=True)


class MLBPredictor:
    """
    Evolving prediction model that improves with each game result.
    Uses ensemble of gradient boosting models for win probability
    and run total prediction.
    """

    VERSION = "v2"

    def __init__(self):
        self.win_model = None
        self.total_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Load existing model or initialize a fresh one."""
        win_path = MODEL_PATH / f"win_model_{self.VERSION}.pkl"
        total_path = MODEL_PATH / f"total_model_{self.VERSION}.pkl"
        scaler_path = MODEL_PATH / f"scaler_{self.VERSION}.pkl"
        features_path = MODEL_PATH / f"features_{self.VERSION}.json"

        if win_path.exists() and total_path.exists():
            try:
                self.win_model = joblib.load(win_path)
                self.total_model = joblib.load(total_path)
                self.scaler = joblib.load(scaler_path)
                with open(features_path) as f:
                    self.feature_names = json.load(f)
                self.is_trained = True
                logger.info("Loaded existing model %s", self.VERSION)
            except Exception as e:
                logger.warning("Failed to load model: %s. Will retrain.", e)
                self._initialize_models()
        else:
            self._initialize_models()

    def _initialize_models(self):
        if XGB_AVAILABLE:
            self.win_model = CalibratedClassifierCV(
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss",
                ),
                cv=3,
                method="sigmoid",
            )
            self.total_model = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            self.win_model = CalibratedClassifierCV(
                GradientBoostingClassifier(n_estimators=100, random_state=42),
                cv=3,
            )
            self.total_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        logger.info("Initialized fresh models")

    def predict(self, game_features: dict) -> dict:
        """Predict win probability and run total for a game."""
        if not self.is_trained:
            return self._heuristic_prediction(game_features)

        try:
            features = self._extract_features(game_features)
            X = pd.DataFrame([features])[self.feature_names]
            X_scaled = self.scaler.transform(X)

            home_win_prob = self.win_model.predict_proba(X_scaled)[0][1]
            predicted_total = float(self.total_model.predict(X_scaled)[0])

            return {
                "home_win_prob": round(home_win_prob, 4),
                "away_win_prob": round(1 - home_win_prob, 4),
                "predicted_total": round(predicted_total, 2),
                "confidence": self._compute_confidence(game_features),
                "model_version": self.VERSION,
                "method": "ml_model",
                "factors": self._explain_top_factors(features),
            }
        except Exception as e:
            logger.warning("ML prediction failed: %s. Using heuristic.", e)
            return self._heuristic_prediction(game_features)

    def _heuristic_prediction(self, features: dict) -> dict:
        """
        Heuristic prediction when model isn't trained yet.
        Uses runs differential, ERA, park factors, and streaks.
        """
        home_score = 0.0
        notes = {}

        # Run differential advantage
        home_rd = features.get("home_run_diff_per_game", 0)
        away_rd = features.get("away_run_diff_per_game", 0)
        rd_factor = (home_rd - away_rd) / 10
        home_score += rd_factor
        notes["run_diff"] = rd_factor

        # ERA matchup
        home_era = features.get("home_sp_era", 4.50)
        away_era = features.get("away_sp_era", 4.50)
        era_factor = (away_era - home_era) / 8
        home_score += era_factor
        notes["era_matchup"] = era_factor

        # Win percentage
        home_wpct = features.get("home_win_pct", 0.500)
        away_wpct = features.get("away_win_pct", 0.500)
        wpct_factor = (home_wpct - away_wpct) * 0.5
        home_score += wpct_factor
        notes["win_pct"] = wpct_factor

        # Home field advantage
        home_score += 0.03
        notes["home_field"] = 0.03

        # Streak momentum
        home_momentum = features.get("home_momentum_score", 0.5)
        away_momentum = features.get("away_momentum_score", 0.5)
        streak_factor = (home_momentum - away_momentum) * 0.2
        home_score += streak_factor
        notes["streaks"] = streak_factor

        # Weather
        weather_impact = features.get("weather_impact_score", 0)
        notes["weather"] = weather_impact

        # Convert score to probability via sigmoid
        import math
        home_win_prob = 1 / (1 + math.exp(-home_score * 3))
        home_win_prob = max(0.25, min(0.75, home_win_prob))

        # Predicted total
        home_era_factor = 4.50 - home_era
        away_era_factor = 4.50 - away_era
        park_factor = features.get("park_runs_factor", 100) / 100
        predicted_total = (9.0 + home_era_factor * 0.4 + away_era_factor * 0.4) * park_factor
        predicted_total += weather_impact * 2
        predicted_total = max(5.0, min(15.0, predicted_total))

        return {
            "home_win_prob": round(home_win_prob, 4),
            "away_win_prob": round(1 - home_win_prob, 4),
            "predicted_total": round(predicted_total, 2),
            "confidence": 0.45,
            "model_version": "heuristic",
            "method": "heuristic",
            "factors": notes,
        }

    def _extract_features(self, game_features: dict) -> dict:
        """Build feature vector from raw game data."""
        return {
            "home_win_pct": game_features.get("home_win_pct", 0.500),
            "away_win_pct": game_features.get("away_win_pct", 0.500),
            "home_run_diff_per_game": game_features.get("home_run_diff_per_game", 0),
            "away_run_diff_per_game": game_features.get("away_run_diff_per_game", 0),
            "home_ops": game_features.get("home_ops", 0.720),
            "away_ops": game_features.get("away_ops", 0.720),
            "home_team_era": game_features.get("home_team_era", 4.20),
            "away_team_era": game_features.get("away_team_era", 4.20),
            "home_bullpen_era": game_features.get("home_bullpen_era", 4.00),
            "away_bullpen_era": game_features.get("away_bullpen_era", 4.00),
            "home_sp_era": game_features.get("home_sp_era", 4.50),
            "away_sp_era": game_features.get("away_sp_era", 4.50),
            "home_sp_whip": game_features.get("home_sp_whip", 1.30),
            "away_sp_whip": game_features.get("away_sp_whip", 1.30),
            "home_sp_k9": game_features.get("home_sp_k9", 8.5),
            "away_sp_k9": game_features.get("away_sp_k9", 8.5),
            "home_sp_recent_era": game_features.get("home_sp_recent_era", 4.50),
            "away_sp_recent_era": game_features.get("away_sp_recent_era", 4.50),
            "home_momentum_score": game_features.get("home_momentum_score", 0.5),
            "away_momentum_score": game_features.get("away_momentum_score", 0.5),
            "home_streak_count": game_features.get("home_streak_count", 0),
            "away_streak_count": game_features.get("away_streak_count", 0),
            "park_runs_factor": game_features.get("park_runs_factor", 100),
            "park_hr_factor": game_features.get("park_hr_factor", 100),
            "weather_impact_score": game_features.get("weather_impact_score", 0),
            "temperature": game_features.get("temperature", 72),
            "wind_speed": game_features.get("wind_speed", 5),
            "h2h_home_win_pct": game_features.get("h2h_home_win_pct", 0.5),
            "home_last10_wins": game_features.get("home_last10_wins", 5),
            "away_last10_wins": game_features.get("away_last10_wins", 5),
            "is_home_dome": int(game_features.get("is_home_dome", False)),
        }

    def _compute_confidence(self, features: dict) -> float:
        """Estimate prediction confidence based on data completeness."""
        score = 0.5
        if features.get("home_sp_era"):
            score += 0.05
        if features.get("home_sp_recent_era"):
            score += 0.05
        if features.get("weather_impact_score") is not None:
            score += 0.03
        if features.get("h2h_home_win_pct"):
            score += 0.03
        if features.get("home_momentum_score"):
            score += 0.04
        return min(0.85, score)

    def _explain_top_factors(self, features: dict) -> dict:
        """Return simplified feature importance explanation."""
        return {
            "win_pct_edge": round(
                features.get("home_win_pct", 0.5) - features.get("away_win_pct", 0.5), 3
            ),
            "sp_era_edge": round(
                features.get("away_sp_era", 4.5) - features.get("home_sp_era", 4.5), 3
            ),
            "streak_edge": round(
                features.get("home_momentum_score", 0.5) - features.get("away_momentum_score", 0.5), 3
            ),
            "park_factor": features.get("park_runs_factor", 100),
            "weather": features.get("weather_impact_score", 0),
        }

    def train(self, training_data: list[dict]) -> dict:
        """
        Train/retrain the model on historical game data.
        Called automatically when enough new results accumulate.
        """
        if len(training_data) < 50:
            logger.warning("Not enough training data (%d games). Need 50+.", len(training_data))
            return {"status": "insufficient_data", "games": len(training_data)}

        df = pd.DataFrame(training_data)
        feature_cols = [c for c in df.columns if c not in
                        ["game_id", "home_won", "actual_total", "date"]]
        self.feature_names = feature_cols

        X = df[feature_cols].fillna(df[feature_cols].median())
        y_win = df["home_won"].astype(int)
        y_total = df["actual_total"]

        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

        self.win_model.fit(X_scaled_df, y_win)
        self.total_model.fit(X_scaled_df, y_total)
        self.is_trained = True

        # Evaluate
        win_probs = self.win_model.predict_proba(X_scaled_df)[:, 1]
        brier = brier_score_loss(y_win, win_probs)
        ll = log_loss(y_win, win_probs)
        accuracy = (win_probs.round() == y_win).mean()

        # Save models
        joblib.dump(self.win_model, MODEL_PATH / f"win_model_{self.VERSION}.pkl")
        joblib.dump(self.total_model, MODEL_PATH / f"total_model_{self.VERSION}.pkl")
        joblib.dump(self.scaler, MODEL_PATH / f"scaler_{self.VERSION}.pkl")
        with open(MODEL_PATH / f"features_{self.VERSION}.json", "w") as f:
            json.dump(feature_cols, f)

        metrics = {
            "status": "trained",
            "games": len(training_data),
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
            "model_version": self.VERSION,
        }

        # Log to DB
        with get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_performance
                (model_version, prediction_date, total_predictions, accuracy, brier_score, log_loss)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (self.VERSION, datetime.now().date().isoformat(),
                  len(training_data), accuracy, brier, ll))

        logger.info("Model retrained: accuracy=%.3f brier=%.4f on %d games",
                    accuracy, brier, len(training_data))
        return metrics

    def collect_training_data(self) -> list[dict]:
        """Pull completed games from DB for model retraining."""
        with get_db() as conn:
            rows = conn.execute("""
                SELECT
                    g.game_id,
                    CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_won,
                    (g.home_score + g.away_score) as actual_total,
                    g.game_date as date,
                    ts_h.wins * 1.0 / (ts_h.wins + ts_h.losses) as home_win_pct,
                    ts_a.wins * 1.0 / (ts_a.wins + ts_a.losses) as away_win_pct,
                    ts_h.era as home_team_era,
                    ts_a.era as away_team_era,
                    w.weather_impact_score,
                    w.temperature,
                    w.wind_speed
                FROM games g
                LEFT JOIN team_stats ts_h ON g.home_team = ts_h.team_name
                    AND ts_h.stat_date = g.game_date
                LEFT JOIN team_stats ts_a ON g.away_team = ts_a.team_name
                    AND ts_a.stat_date = g.game_date
                LEFT JOIN weather_data w ON g.game_id = w.game_id
                WHERE g.status = 'Final'
                AND g.home_score IS NOT NULL
                AND (ts_h.wins + ts_h.losses) > 0
                ORDER BY g.game_date DESC
                LIMIT 2000
            """).fetchall()

        return [dict(r) for r in rows]
