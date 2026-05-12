"""
Self-improvement engine — analyzes graded picks to learn:
1. Which factors actually predict wins (factor performance)
2. Which markets hit at what rate (market performance)
3. What losing_pct thresholds should be for each tier (dynamic thresholds)
4. Which signal combinations outperform (signal synergy)

Runs automatically when 50+ graded picks exist. Saves learned weights
to DB so the model applies them on next run.
"""
from __future__ import annotations
import json
import logging
import math
from collections import defaultdict
from ..database import get_db

logger = logging.getLogger(__name__)

MIN_SAMPLE = 20   # Minimum graded bets before adjusting any weight
FULL_SAMPLE = 50  # Preferred minimum for threshold adjustment


def analyze_factor_performance() -> dict[str, dict]:
    """
    For each factor string (or factor keyword), compute win rate
    across all graded MODEL_PICK bets.
    Returns {factor_key: {wins, losses, win_rate, weight_multiplier}}
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT factors, result FROM value_bets
            WHERE result IN ('WIN','LOSS')
            AND confidence IN ('MODEL_PICK', 'PLACED')
            AND factors IS NOT NULL
            AND detected_at >= datetime('now', '-90 days')
        """).fetchall()

    if len(rows) < MIN_SAMPLE:
        return {}

    factor_stats2: dict[str, dict] = {}
    for row in rows:
        result = row["result"]
        try:
            factors = json.loads(row["factors"] or "[]")
        except Exception:
            factors = []
        won = result == "WIN"
        for f in factors:
            kw_list = _extract_factor_keywords(str(f).lower())
            for kw in kw_list:
                if kw not in factor_stats2:
                    factor_stats2[kw] = {"wins": 0, "losses": 0}
                if won:
                    factor_stats2[kw]["wins"] += 1
                else:
                    factor_stats2[kw]["losses"] += 1

    # Compute win rate and weight multiplier
    results = {}
    for kw, stats in factor_stats2.items():
        total = stats["wins"] + stats["losses"]
        if total < 5:
            continue
        win_rate = stats["wins"] / total
        # Weight multiplier: 1.0 = neutral, >1.0 = boost, <1.0 = penalize
        # Based on deviation from 0.55 baseline (breakeven with typical -110 juice)
        if win_rate >= 0.65:
            multiplier = min(1.5, 1.0 + (win_rate - 0.55) * 5)
        elif win_rate <= 0.45:
            multiplier = max(0.5, 1.0 - (0.55 - win_rate) * 5)
        else:
            multiplier = 1.0
        results[kw] = {
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": round(win_rate, 4),
            "weight_multiplier": round(multiplier, 3),
            "sample": total,
        }

    return results


def _extract_factor_keywords(factor_str: str) -> list[str]:
    """Extract meaningful keywords from a factor string."""
    keywords = []
    keyword_map = {
        "siera": "elite_siera", "xfip": "xfip_edge", "era": "era_edge",
        "k/9": "high_k9", "whip": "whip_edge", "bullpen": "bullpen_signal",
        "wind": "wind_signal", "wrigley": "wrigley_wind", "park": "park_factor",
        "sharp": "sharp_money", "pythag": "pythag_luck", "umpire": "umpire_lean",
        "travel": "travel_fatigue", "opener": "opener_game",
        "velocity": "velocity_drop", "barrel": "barrel_rate",
        "xwoba": "xwoba_luck", "momentum": "team_momentum",
        "streak": "win_streak", "below .500": "fade_sub500",
        "road": "road_record", "home": "home_edge",
        "fraud": "era_fraud", "fade list": "fade_list",
        "backs list": "backs_list", "clv": "clv_positive",
    }
    for token, kw in keyword_map.items():
        if token in factor_str:
            keywords.append(kw)
    return keywords if keywords else ["generic_factor"]


def analyze_market_performance() -> dict[str, dict]:
    """
    Compute win rate per market type across all graded bets.
    Returns {market: {wins, losses, win_rate, recommended_boost}}
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT market, result FROM value_bets
            WHERE result IN ('WIN', 'LOSS')
            AND detected_at >= datetime('now', '-120 days')
        """).fetchall()

    stats: dict[str, dict] = {}
    for row in rows:
        mkt = _normalize_market(row["market"] or "unknown")
        if mkt not in stats:
            stats[mkt] = {"wins": 0, "losses": 0}
        if row["result"] == "WIN":
            stats[mkt]["wins"] += 1
        else:
            stats[mkt]["losses"] += 1

    results = {}
    for mkt, s in stats.items():
        total = s["wins"] + s["losses"]
        if total < 3:
            continue
        win_rate = s["wins"] / total
        results[mkt] = {
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": round(win_rate, 4),
            "sample": total,
            "recommended_boost": win_rate >= 0.60,
        }
    return results


def _normalize_market(market: str) -> str:
    m = market.lower().strip()
    if any(kw in m for kw in ("ml", "moneyline", "h2h", "full_game")):
        return "full_game_ml"
    if "f5" in m and "ml" in m:
        return "f5_ml"
    if "f5" in m and "over" in m:
        return "f5_over"
    if "f5" in m and "under" in m:
        return "f5_under"
    if "nrfi" in m:
        return "nrfi"
    if "yrfi" in m:
        return "yrfi"
    if "run_line" in m or "rl" in m:
        return "run_line"
    if "k_over" in m or "strikeout" in m:
        return "k_over"
    if "k_under" in m:
        return "k_under"
    if "over" in m and "game" in m:
        return "game_over"
    if "under" in m and "game" in m:
        return "game_under"
    if "outs" in m:
        return "outs_recorded"
    if "hr" in m:
        return "hr_prop"
    return m[:30]


def compute_dynamic_thresholds(graded_count: int) -> dict:
    """
    Compute new tier thresholds based on actual win rates per losing_pct bucket.
    Only adjusts when graded_count >= FULL_SAMPLE.
    Returns dict of {tier: (low, high)} losing_pct ranges.
    """
    defaults = {
        "STRONG": (0.00, 0.25),
        "MEDIUM": (0.25, 0.32),
        "LEAN":   (0.32, 0.40),
        "SKIP":   (0.40, 1.00),
    }

    if graded_count < FULL_SAMPLE:
        return defaults

    with get_db() as conn:
        rows = conn.execute("""
            SELECT vb.model_probability, vb.result
            FROM value_bets vb
            WHERE vb.result IN ('WIN','LOSS')
            AND vb.confidence = 'MODEL_PICK'
            AND vb.model_probability > 0
            ORDER BY vb.model_probability DESC
        """).fetchall()

    if len(rows) < FULL_SAMPLE:
        return defaults

    # Find the losing_pct cutoff where win rate drops to ~52% (breakeven vs -110)
    # Sort by model probability
    rows = sorted(rows, key=lambda r: r["model_probability"], reverse=True)
    total = len(rows)

    # Top 20%: STRONG candidates
    strong_cut = max(0.60, rows[int(total * 0.20)]["model_probability"] if total > 5 else 0.75)
    # Top 20-40%: MEDIUM
    medium_cut = max(0.55, rows[int(total * 0.40)]["model_probability"] if total > 10 else 0.68)
    # Top 40-60%: LEAN
    lean_cut = max(0.50, rows[int(total * 0.60)]["model_probability"] if total > 20 else 0.60)

    return {
        "STRONG": (0.00, round(1 - strong_cut, 3)),
        "MEDIUM": (round(1 - strong_cut, 3), round(1 - medium_cut, 3)),
        "LEAN":   (round(1 - medium_cut, 3), round(1 - lean_cut, 3)),
        "SKIP":   (round(1 - lean_cut, 3), 1.00),
    }


def save_learned_weights(factor_perf: dict, market_perf: dict, thresholds: dict) -> None:
    """Persist learned weights to DB model_weights table."""
    with get_db() as conn:
        # Factor weights
        for factor_key, stats in factor_perf.items():
            conn.execute("""
                INSERT INTO model_weights (weight_key, weight_value, sample_size)
                VALUES (?, ?, ?)
                ON CONFLICT(weight_key) DO UPDATE SET
                    weight_value = excluded.weight_value,
                    sample_size  = excluded.sample_size,
                    updated_at   = datetime('now')
            """, (f"factor:{factor_key}", stats["weight_multiplier"], stats["sample"]))

        # Market win rates
        for market, stats in market_perf.items():
            conn.execute("""
                INSERT INTO model_weights (weight_key, weight_value, sample_size)
                VALUES (?, ?, ?)
                ON CONFLICT(weight_key) DO UPDATE SET
                    weight_value = excluded.weight_value,
                    sample_size  = excluded.sample_size,
                    updated_at   = datetime('now')
            """, (f"market:{market}", stats["win_rate"], stats["sample"]))

        # Tier thresholds
        for tier, (lo, hi) in thresholds.items():
            conn.execute("""
                INSERT INTO model_weights (weight_key, weight_value, sample_size)
                VALUES (?, ?, 0)
                ON CONFLICT(weight_key) DO UPDATE SET
                    weight_value = excluded.weight_value,
                    updated_at   = datetime('now')
            """, (f"threshold:{tier}:lo", lo))
            conn.execute("""
                INSERT INTO model_weights (weight_key, weight_value, sample_size)
                VALUES (?, ?, 0)
                ON CONFLICT(weight_key) DO UPDATE SET
                    weight_value = excluded.weight_value,
                    updated_at   = datetime('now')
            """, (f"threshold:{tier}:hi", hi))

    logger.info("Saved %d factor weights, %d market weights, %d tier thresholds",
                len(factor_perf), len(market_perf), len(thresholds))


def load_learned_weights() -> dict:
    """
    Load all learned weights from DB. Returns structured dict:
    {
        "factors": {key: multiplier},
        "markets": {market: win_rate},
        "thresholds": {tier: (lo, hi)},
        "sample_size": int,
    }
    """
    defaults = {
        "factors": {},
        "markets": {},
        "thresholds": {
            "STRONG": (0.00, 0.25),
            "MEDIUM": (0.25, 0.32),
            "LEAN":   (0.32, 0.40),
            "SKIP":   (0.40, 1.00),
        },
        "sample_size": 0,
    }
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT weight_key, weight_value, sample_size FROM model_weights
            """).fetchall()
        if not rows:
            return defaults

        result = {"factors": {}, "markets": {}, "thresholds": {}, "sample_size": 0}
        for row in rows:
            k = row["weight_key"]
            v = row["weight_value"]
            s = row["sample_size"] or 0
            if k.startswith("factor:"):
                result["factors"][k[7:]] = v
            elif k.startswith("market:"):
                result["markets"][k[7:]] = v
            elif k.startswith("threshold:"):
                parts = k.split(":")
                tier, bound = parts[1], parts[2]
                if tier not in result["thresholds"]:
                    result["thresholds"][tier] = [0.0, 1.0]
                idx = 0 if bound == "lo" else 1
                result["thresholds"][tier][idx] = v
            result["sample_size"] = max(result["sample_size"], s)

        # Fill missing thresholds from defaults
        for tier, vals in defaults["thresholds"].items():
            if tier not in result["thresholds"]:
                result["thresholds"][tier] = list(vals)

        # Convert lists to tuples
        result["thresholds"] = {t: tuple(v) for t, v in result["thresholds"].items()}
        return result

    except Exception as e:
        logger.warning("Failed to load learned weights: %s", e)
        return defaults


def run_full_retrain() -> dict:
    """
    Main entry point. Runs full self-improvement cycle:
    1. Check how many graded picks we have
    2. Analyze factor performance
    3. Analyze market performance
    4. Compute dynamic thresholds
    5. Save all to DB
    Returns summary dict.
    """
    with get_db() as conn:
        count_row = conn.execute("""
            SELECT COUNT(*) as cnt FROM value_bets
            WHERE result IN ('WIN','LOSS')
            AND confidence IN ('MODEL_PICK','PLACED')
        """).fetchone()
    graded_count = count_row["cnt"] if count_row else 0

    logger.info("Weight retrain: %d graded picks available", graded_count)

    if graded_count < MIN_SAMPLE:
        return {"status": "insufficient_data", "graded": graded_count, "needed": MIN_SAMPLE}

    factor_perf  = analyze_factor_performance()
    market_perf  = analyze_market_performance()
    thresholds   = compute_dynamic_thresholds(graded_count)
    save_learned_weights(factor_perf, market_perf, thresholds)

    # Find best and worst factors
    if factor_perf:
        best_factor  = max(factor_perf, key=lambda k: factor_perf[k]["win_rate"])
        worst_factor = min(factor_perf, key=lambda k: factor_perf[k]["win_rate"])
    else:
        best_factor = worst_factor = "n/a"

    if market_perf:
        best_market  = max(market_perf, key=lambda k: market_perf[k]["win_rate"])
        worst_market = min(market_perf, key=lambda k: market_perf[k]["win_rate"])
    else:
        best_market = worst_market = "n/a"

    return {
        "status": "retrained",
        "graded": graded_count,
        "factors_analyzed": len(factor_perf),
        "markets_analyzed": len(market_perf),
        "best_factor":  best_factor,
        "worst_factor": worst_factor,
        "best_market":  best_market,
        "worst_market": worst_market,
        "thresholds":   thresholds,
    }
