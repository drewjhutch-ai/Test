"""
Signal performance tracker — feedback loop for continuous improvement.
Tracks which of the 12-layer signals actually predict wins over time.
Adjusts signal weights in the model based on real hit rates.
Also provides a CLI to record bets you actually placed.
"""
import json
import logging
from datetime import datetime, timedelta
from ..database import get_db

logger = logging.getLogger(__name__)

# Default signal weights — adjusted automatically over time
DEFAULT_WEIGHTS = {
    "permanent_backs_list":     1.20,
    "permanent_fade_list":      1.20,
    "era_fraud_confirmed":      1.15,
    "cps_gap_strong":           1.10,
    "cps_gap_medium":           1.05,
    "hot_streak_3plus":         1.08,
    "cold_streak_3plus":        0.92,
    "pitcher_hot_last5":        1.10,
    "pitcher_cold_last5":       0.90,
    "weather_wind_in":          1.05,
    "weather_cold_under55":     1.05,
    "dome_game":                1.03,
    "sharp_money_confirm":      1.12,
    "steam_move":               1.15,
    "pinnacle_discrepancy":     1.10,
    "home_field":               1.03,
    "winning_record_520":       1.08,
    "positive_run_diff":        1.06,
    "park_factor_favorable":    1.04,
    "series_momentum":          1.05,
}

WEIGHTS_CACHE_KEY = "signal_weights"


def get_signal_weights() -> dict[str, float]:
    """Load current signal weights from DB, or return defaults."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT value FROM model_performance
            WHERE model_version = 'signal_weights'
            ORDER BY updated_at DESC LIMIT 1
        """).fetchone()

    if row:
        try:
            return json.loads(row["value"] if "value" in row.keys() else str(row[0]))
        except Exception:
            pass
    return DEFAULT_WEIGHTS.copy()


def save_signal_weights(weights: dict):
    """Persist updated signal weights to DB."""
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO model_performance
            (model_version, prediction_date, total_predictions, updated_at)
            VALUES ('signal_weights', ?, 0, datetime('now'))
        """, (datetime.now().date().isoformat(),))
    logger.info("Signal weights saved.")


def record_placed_bet(
    game_id: str,
    book: str,
    market: str,
    side: str,
    price: int,
    units: float,
    signals_present: list[str],
):
    """
    Record a bet you actually placed on DraftKings.
    This is how you tell the model what you did so it can learn from it.
    """
    unit_dollars = 5.0
    stake = units * unit_dollars

    with get_db() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO games (game_id, home_team, away_team, game_date, status)
            VALUES (?, 'Unknown', 'Unknown', date('now'), 'scheduled')
        """, (game_id,))

        conn.execute("""
            INSERT INTO value_bets
            (game_id, book, market, side, book_price, model_probability,
             implied_probability, edge, kelly_fraction, recommended_bet,
             confidence, factors)
            VALUES (?, ?, ?, ?, ?, 0, 0, 0, 0, ?, 'PLACED', ?)
        """, (
            game_id, book, market, side, price,
            stake,
            json.dumps(signals_present),
        ))

    logger.info(
        "Recorded placed bet: %s %s %s @ %+d (%su/$%.0f)",
        game_id, side, market, price, units, stake
    )


def auto_record_model_picks(picks: list, parlays: list, date_str: str) -> int:
    """
    Automatically record all model picks and parlay selections so the model
    can learn from its own output — not just bets you manually log.
    Confidence = 'MODEL_PICK' for singles, 'MODEL_PARLAY' for parlay legs.
    Uses INSERT OR IGNORE on (game_id, confidence) to avoid duplicates on
    re-runs.
    """
    recorded = 0
    today = date_str or datetime.now().strftime("%Y-%m-%d")

    with get_db() as conn:
        # ── Single picks (POTD + all active picks) ──────────────────────
        for pick in picks:
            if getattr(pick, "tier", "SKIP") == "SKIP":
                continue
            game_id = f"model_{today}_{getattr(pick, 'game_id', 'unknown')}"
            mkt = getattr(pick, "recommended_market", "") or getattr(pick, "proposed_market", "full_game_ml")
            backing = getattr(pick, "backing_team", "")
            tier = getattr(pick, "tier", "LEAN")
            units = {"STRONG": 3, "MEDIUM": 2, "LEAN": 1}.get(tier, 1)
            factors = getattr(pick, "factors", [])
            ev = getattr(pick, "ev_pct", 0.0)
            prob = 1.0 - getattr(pick, "losing_pct", 0.5)

            # Deduplicate: skip if already recorded for today
            existing = conn.execute(
                "SELECT id FROM value_bets WHERE game_id = ? AND confidence = 'MODEL_PICK' LIMIT 1",
                (game_id,)
            ).fetchone()
            if existing:
                continue

            conn.execute("""
                INSERT INTO games (game_id, home_team, away_team, game_date, status)
                VALUES (?, ?, ?, ?, 'scheduled')
                ON CONFLICT(game_id) DO NOTHING
            """, (game_id, getattr(pick, "home_team", ""), getattr(pick, "away_team", ""), today))

            conn.execute("""
                INSERT INTO value_bets
                (game_id, book, market, side, book_price, model_probability,
                 implied_probability, edge, kelly_fraction, recommended_bet,
                 confidence, factors)
                VALUES (?, 'draftkings', ?, ?, 0, ?, 0, ?, 0, ?, 'MODEL_PICK', ?)
            """, (
                game_id, mkt, backing.lower(),
                round(prob, 4), round(ev, 4),
                float(units * 5),
                json.dumps(factors[:8]),
            ))
            recorded += 1

        # ── Parlay picks ─────────────────────────────────────────────────
        parlay_labels = ["P1_Anchor", "P2_Core", "P3_Science", "P4_Push", "P5_Moonshot"]
        for pi, parlay in enumerate(parlays):
            plabel = parlay_labels[pi] if pi < len(parlay_labels) else f"P{pi+1}"
            parlay_id = f"model_{today}_{plabel}"
            legs = getattr(parlay, "legs", [])
            total_legs = len(legs)

            for j, leg in enumerate(legs, 1):
                game_id = f"{parlay_id}_leg{j}"
                existing = conn.execute(
                    "SELECT id FROM value_bets WHERE game_id = ? AND confidence = 'MODEL_PARLAY' LIMIT 1",
                    (game_id,)
                ).fetchone()
                if existing:
                    continue

                pick_ref = getattr(leg, "pick", None)
                gid_raw  = getattr(pick_ref, "game_id", "unknown") if pick_ref else "unknown"
                backing  = getattr(pick_ref, "backing_team", "") if pick_ref else ""
                home     = getattr(pick_ref, "home_team", "") if pick_ref else ""
                away     = getattr(pick_ref, "away_team", "") if pick_ref else ""

                conn.execute("""
                    INSERT INTO games (game_id, home_team, away_team, game_date, status)
                    VALUES (?, ?, ?, ?, 'scheduled')
                    ON CONFLICT(game_id) DO NOTHING
                """, (game_id, home, away, today))

                conn.execute("""
                    INSERT INTO value_bets
                    (game_id, book, market, side, book_price, model_probability,
                     implied_probability, edge, kelly_fraction, recommended_bet,
                     confidence, factors)
                    VALUES (?, 'draftkings', ?, ?, ?, ?, 0, 0, 0, 0, 'MODEL_PARLAY', ?)
                """, (
                    game_id,
                    getattr(leg, "market", "ml"),
                    backing.lower(),
                    getattr(leg, "price", 0),
                    round(getattr(leg, "true_prob", 0.5), 4),
                    json.dumps([f"parlay:{parlay_id}", f"leg:{j}of{total_legs}",
                                f"label:{plabel}", f"game:{gid_raw}"]),
                ))
                recorded += 1

    logger.info("Auto-recorded %d model picks/parlay legs for %s", recorded, today)
    return recorded


def grade_pending_picks() -> int:
    """
    Query the MLB Stats API for games that have finished and update the
    result column (WIN / LOSS) + profit_loss for all pending model picks.

    Only grades MODEL_PICK and MODEL_PARLAY rows from previous days.
    PLACED bets use a separate UI flow because their game_id doesn't
    contain a parseable MLB numeric ID.
    """
    import statsapi
    from datetime import date as _date

    today = _date.today().isoformat()
    graded = 0

    with get_db() as conn:
        pending = conn.execute("""
            SELECT id, game_id, market, side, book_price, recommended_bet,
                   confidence, factors
            FROM value_bets
            WHERE result IS NULL
              AND confidence IN ('MODEL_PICK', 'MODEL_PARLAY')
              AND date(detected_at) < ?
        """, (today,)).fetchall()

        for row in pending:
            mlb_id = _extract_mlb_id(row["game_id"], row["factors"])
            if not mlb_id:
                continue

            try:
                games = statsapi.schedule(game_id=mlb_id)
                if not games:
                    continue
                g = games[0]
                if g.get("status") != "Final":
                    continue

                home_name = g.get("home_name", "").lower()
                away_name = g.get("away_name", "").lower()
                home_score = int(g.get("home_score", 0) or 0)
                away_score = int(g.get("away_score", 0) or 0)

                won = _team_won(row["side"], home_name, away_name, home_score, away_score)
                if won is None:
                    continue

                result = "WIN" if won else "LOSS"
                price  = float(row["book_price"] or 0)
                stake  = float(row["recommended_bet"] or 5.0)
                if won:
                    pl = stake * price / 100 if price > 0 else stake * 100 / abs(price)
                else:
                    pl = -stake

                conn.execute(
                    "UPDATE value_bets SET result = ?, profit_loss = ? WHERE id = ?",
                    (result, round(pl, 2), row["id"]),
                )
                graded += 1

            except Exception as e:
                logger.debug("Could not grade pick id=%s game=%s: %s", row["id"], row["game_id"], e)

    if graded:
        logger.info("Graded %d pending picks.", graded)
    return graded


def _extract_mlb_id(game_id: str, factors_json: str | None) -> int | None:
    """Parse the raw MLB numeric game ID from a composite game_id string."""
    # model_YYYY-MM-DD_mlb_NNNNNN  (single picks)
    if "_mlb_" in game_id:
        try:
            return int(game_id.split("_mlb_")[1].split("_")[0])
        except (ValueError, IndexError):
            pass
    # mlb_NNNNNN  (direct)
    if game_id.startswith("mlb_"):
        try:
            return int(game_id[4:].split("_")[0])
        except ValueError:
            pass
    # Parlay legs store game_id in factors JSON as "game:mlb_NNNNNN"
    try:
        for f in json.loads(factors_json or "[]"):
            s = str(f)
            if s.startswith("game:mlb_"):
                return int(s[9:].split("_")[0])
            if s.startswith("game:") and s[5:].isdigit():
                return int(s[5:])
    except Exception:
        pass
    return None


def _team_won(side: str, home: str, away: str,
              home_score: int, away_score: int) -> bool | None:
    """Return True if `side` won, False if lost, None if can't match."""
    if not side:
        return None
    s = side.lower().strip()
    # Try each word in the team name (e.g. "cubs" matches "chicago cubs")
    for token in s.split():
        if len(token) < 3:
            continue
        if token in home:
            return home_score > away_score
        if token in away:
            return away_score > home_score
    return None


def update_signal_performance():
    """
    After games complete, calculate per-signal win rates and
    adjust weights. Called automatically after each retrain cycle.
    """
    with get_db() as conn:
        graded = conn.execute("""
            SELECT factors, result FROM value_bets
            WHERE result IS NOT NULL
            AND factors IS NOT NULL
            AND detected_at >= datetime('now', '-90 days')
        """).fetchall()

    if len(graded) < 20:
        logger.info("Not enough graded bets yet (%d) to update signal weights.", len(graded))
        return

    signal_results: dict[str, list[int]] = {}

    for row in graded:
        won = 1 if row["result"] == "WIN" else 0
        try:
            factors = json.loads(row["factors"]) if isinstance(row["factors"], str) else row["factors"]
            if isinstance(factors, list):
                factor_keys = factors
            elif isinstance(factors, dict):
                factor_keys = list(factors.keys())
            else:
                continue
        except Exception:
            continue

        for signal in _map_factors_to_signals(factor_keys):
            if signal not in signal_results:
                signal_results[signal] = []
            signal_results[signal].append(won)

    weights = DEFAULT_WEIGHTS.copy()
    updated = []

    for signal, results in signal_results.items():
        if len(results) < 10:
            continue
        hit_rate = sum(results) / len(results)
        # Adjust weight: signals hitting >55% get boosted, <45% get penalized
        if hit_rate >= 0.60:
            new_weight = min(1.35, DEFAULT_WEIGHTS.get(signal, 1.0) * 1.10)
        elif hit_rate >= 0.55:
            new_weight = min(1.25, DEFAULT_WEIGHTS.get(signal, 1.0) * 1.05)
        elif hit_rate <= 0.40:
            new_weight = max(0.75, DEFAULT_WEIGHTS.get(signal, 1.0) * 0.90)
        elif hit_rate <= 0.45:
            new_weight = max(0.85, DEFAULT_WEIGHTS.get(signal, 1.0) * 0.95)
        else:
            new_weight = DEFAULT_WEIGHTS.get(signal, 1.0)

        weights[signal] = round(new_weight, 3)
        updated.append(f"{signal}: {hit_rate:.0%} hit rate ({len(results)} bets) → weight {new_weight:.3f}")

    save_signal_weights(weights)
    for u in updated:
        logger.info("Signal weight updated: %s", u)

    return weights


def get_signal_performance_report() -> list[dict]:
    """Return a summary of each signal's historical performance."""
    with get_db() as conn:
        graded = conn.execute("""
            SELECT factors, result FROM value_bets
            WHERE result IS NOT NULL AND factors IS NOT NULL
        """).fetchall()

    signal_results: dict[str, list] = {}
    for row in graded:
        won = 1 if row["result"] == "WIN" else 0
        try:
            factors = json.loads(row["factors"]) if isinstance(row["factors"], str) else []
            if isinstance(factors, dict):
                factors = list(factors.keys())
        except Exception:
            continue
        for signal in _map_factors_to_signals(factors):
            signal_results.setdefault(signal, []).append(won)

    report = []
    for signal, results in sorted(signal_results.items(), key=lambda x: -len(x[1])):
        if len(results) < 5:
            continue
        hit_rate = sum(results) / len(results)
        report.append({
            "signal": signal,
            "bets": len(results),
            "wins": sum(results),
            "hit_rate": round(hit_rate, 4),
            "current_weight": get_signal_weights().get(signal, 1.0),
        })

    return sorted(report, key=lambda x: -x["hit_rate"])


def _map_factors_to_signals(factors: list[str]) -> list[str]:
    """Map human-readable factor strings to signal keys."""
    signals = []
    joined = " ".join(str(f).lower() for f in factors)

    mappings = {
        "permanent backs":      "permanent_backs_list",
        "permanent fade":       "permanent_fade_list",
        "era fraud":            "era_fraud_confirmed",
        "cps gap":              "cps_gap_strong",
        "hot streak":           "hot_streak_3plus",
        "cold streak":          "cold_streak_3plus",
        "hot":                  "pitcher_hot_last5",
        "cold":                 "pitcher_cold_last5",
        "wind":                 "weather_wind_in",
        "dome":                 "dome_game",
        "sharp":                "sharp_money_confirm",
        "steam":                "steam_move",
        "pinnacle":             "pinnacle_discrepancy",
        "home field":           "home_field",
        "winning record":       "winning_record_520",
        "run differential":     "positive_run_diff",
        "park factor":          "park_factor_favorable",
        "series momentum":      "series_momentum",
    }

    for keyword, signal_key in mappings.items():
        if keyword in joined:
            signals.append(signal_key)

    return list(set(signals))
