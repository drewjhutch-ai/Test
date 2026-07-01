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

# Sample-size gates. Raised sharply per the research: online weight updates on
# small recent samples chase variance ("the past may not be representative"),
# which is exactly what corrupted the tier thresholds and zeroed out picks.
# Betting literature puts the minimum to distinguish edge from luck at ~200 bets
# (300-500 to trust). Until then the model runs on sane fixed defaults.
MIN_SAMPLE = 50    # Minimum graded bets before adjusting any factor weight
FULL_SAMPLE = 200  # Minimum before touching tier thresholds at all

# Shrinkage applied to learned factor multipliers — pulls every multiplier toward
# 1.0 (neutral) so a noisy small sample can't swing the model hard.
FACTOR_SHRINKAGE = 0.5


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
        # Shrink toward neutral so a small/noisy sample can't swing the model.
        multiplier = round(1.0 + (multiplier - 1.0) * FACTOR_SHRINKAGE, 3)
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
        "STRONG": (0.00, 0.28),
        "MEDIUM": (0.28, 0.36),
        "LEAN":   (0.36, 0.44),
        "SKIP":   (0.44, 1.00),
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

    rows = sorted(rows, key=lambda r: r["model_probability"], reverse=True)
    total = len(rows)

    # Percentile cuts for tier boundaries
    raw_strong = rows[int(total * 0.20)]["model_probability"] if total > 5 else 0.72
    raw_medium = rows[int(total * 0.40)]["model_probability"] if total > 10 else 0.64
    raw_lean   = rows[int(total * 0.60)]["model_probability"] if total > 20 else 0.56

    # Enforce floor values so thresholds stay meaningful
    strong_cut = max(0.65, min(raw_strong, 0.85))
    medium_cut = max(0.58, min(raw_medium, strong_cut - 0.05))
    lean_cut   = max(0.52, min(raw_lean,   medium_cut - 0.05))

    s_lo = 0.00
    s_hi = round(1 - strong_cut, 3)
    m_lo = s_hi
    m_hi = round(1 - medium_cut, 3)
    l_lo = m_hi
    l_hi = round(1 - lean_cut, 3)

    # Safety: if any range collapsed (width < 0.04) fall back to defaults
    widths = [s_hi - s_lo, m_hi - m_lo, l_hi - l_lo]
    if any(w < 0.04 for w in widths):
        logger.warning(
            "compute_dynamic_thresholds: percentile collapse detected (widths=%s) — using defaults",
            widths,
        )
        return defaults

    return {
        "STRONG": (s_lo, s_hi),
        "MEDIUM": (m_lo, m_hi),
        "LEAN":   (l_lo, l_hi),
        "SKIP":   (l_hi, 1.00),
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


def analyze_tier_accuracy() -> dict[str, dict]:
    """
    Compute actual win rate per tier (STRONG/MEDIUM/LEAN) vs expected.
    The model assigns tiers based on losing_pct; this checks whether
    STRONG picks actually win more often than LEAN in practice.
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT vb.model_probability, vb.edge, vb.result, vb.factors
            FROM value_bets vb
            WHERE vb.result IN ('WIN','LOSS')
            AND vb.confidence IN ('MODEL_PICK', 'PLACED')
            AND vb.detected_at >= datetime('now', '-180 days')
        """).fetchall()

    tier_stats: dict[str, dict] = {
        "STRONG": {"wins": 0, "losses": 0},
        "MEDIUM": {"wins": 0, "losses": 0},
        "LEAN":   {"wins": 0, "losses": 0},
    }

    for row in rows:
        # Derive tier from model_probability stored at bet time
        prob = row["model_probability"] or 0.5
        lose_pct = 1.0 - prob
        if lose_pct < 0.25:
            tier = "STRONG"
        elif lose_pct < 0.32:
            tier = "MEDIUM"
        elif lose_pct < 0.40:
            tier = "LEAN"
        else:
            continue
        if row["result"] == "WIN":
            tier_stats[tier]["wins"] += 1
        else:
            tier_stats[tier]["losses"] += 1

    results = {}
    for tier, s in tier_stats.items():
        total = s["wins"] + s["losses"]
        if total < 3:
            continue
        win_rate = s["wins"] / total
        expected = {"STRONG": 0.75, "MEDIUM": 0.68, "LEAN": 0.60}.get(tier, 0.60)
        results[tier] = {
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": round(win_rate, 4),
            "expected_win_rate": expected,
            "vs_expected": round(win_rate - expected, 4),
            "sample": total,
        }
    return results


def analyze_context_patterns() -> dict[str, dict]:
    """
    Analyze win rates by situational context extracted from factor strings:
    home/away, weather (wind/dome), series position, bullpen signals, etc.
    Returns {context_label: {win_rate, wins, losses, sample}}.
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT factors, result FROM value_bets
            WHERE result IN ('WIN','LOSS')
            AND factors IS NOT NULL
            AND detected_at >= datetime('now', '-180 days')
        """).fetchall()

    context_stats: dict[str, dict] = {}

    context_checks = [
        ("home_field",       lambda f: "home" in f and "road" not in f),
        ("road_team",        lambda f: "road" in f or "away" in f),
        ("wind_in",          lambda f: "wind" in f and "in" in f),
        ("wind_out",         lambda f: "wind" in f and "out" in f),
        ("dome_game",        lambda f: "dome" in f),
        ("bullpen_edge",     lambda f: "bullpen" in f),
        ("sharp_money",      lambda f: "sharp" in f),
        ("hot_streak",       lambda f: "streak" in f and "cold" not in f),
        ("cold_streak",      lambda f: "cold" in f and "streak" in f),
        ("era_fraud",        lambda f: "fraud" in f),
        ("backs_list",       lambda f: "backs" in f),
        ("fade_list",        lambda f: "fade" in f and "list" in f),
        ("pythag_luck",      lambda f: "pythag" in f),
        ("velocity_drop",    lambda f: "velocity" in f),
        ("park_factor",      lambda f: "park" in f),
    ]

    for row in rows:
        won = row["result"] == "WIN"
        try:
            factors = json.loads(row["factors"] or "[]")
        except Exception:
            continue
        joined = " ".join(str(f).lower() for f in factors)
        for label, check in context_checks:
            if check(joined):
                if label not in context_stats:
                    context_stats[label] = {"wins": 0, "losses": 0}
                if won:
                    context_stats[label]["wins"] += 1
                else:
                    context_stats[label]["losses"] += 1

    results = {}
    for label, s in context_stats.items():
        total = s["wins"] + s["losses"]
        if total < 3:
            continue
        win_rate = s["wins"] / total
        results[label] = {
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": round(win_rate, 4),
            "sample": total,
            "signal_strength": round(abs(win_rate - 0.55), 4),
        }
    return results


def _save_analysis_results(tier_accuracy: dict, context_patterns: dict) -> None:
    """Persist tier accuracy and context pattern results to model_weights table."""
    with get_db() as conn:
        for tier, stats in tier_accuracy.items():
            conn.execute("""
                INSERT INTO model_weights (weight_key, weight_value, sample_size)
                VALUES (?, ?, ?)
                ON CONFLICT(weight_key) DO UPDATE SET
                    weight_value = excluded.weight_value,
                    sample_size  = excluded.sample_size,
                    updated_at   = datetime('now')
            """, (f"tier_accuracy:{tier}", stats["win_rate"], stats["sample"]))

        for label, stats in context_patterns.items():
            conn.execute("""
                INSERT INTO model_weights (weight_key, weight_value, sample_size)
                VALUES (?, ?, ?)
                ON CONFLICT(weight_key) DO UPDATE SET
                    weight_value = excluded.weight_value,
                    sample_size  = excluded.sample_size,
                    updated_at   = datetime('now')
            """, (f"context:{label}", stats["win_rate"], stats["sample"]))


def _purge_corrupt_thresholds() -> None:
    """
    Delete any tier threshold rows where lo >= hi (degenerate/empty ranges).
    Called at the top of run_full_retrain so corrupt values are cleared before
    compute_dynamic_thresholds() writes fresh ones.
    """
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT weight_key, weight_value FROM model_weights WHERE weight_key LIKE 'threshold:%'"
            ).fetchall()

        # Reconstruct threshold pairs from DB rows
        tier_vals: dict[str, dict] = {}
        for row in rows:
            parts = row["weight_key"].split(":")
            if len(parts) != 3:
                continue
            _, tier, bound = parts
            if tier not in tier_vals:
                tier_vals[tier] = {}
            tier_vals[tier][bound] = row["weight_value"]

        # Identify corrupt tiers
        corrupt = [
            tier for tier, bv in tier_vals.items()
            if "lo" in bv and "hi" in bv and bv["lo"] >= bv["hi"]
        ]

        if corrupt:
            with get_db() as conn:
                for tier in corrupt:
                    conn.execute(
                        "DELETE FROM model_weights WHERE weight_key LIKE ?",
                        (f"threshold:{tier}:%",),
                    )
            logger.warning("Purged corrupt tier thresholds from DB: %s", corrupt)
        else:
            logger.info("_purge_corrupt_thresholds: no corrupt thresholds found")

    except Exception as e:
        logger.warning("_purge_corrupt_thresholds failed (non-fatal): %s", e)


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
    _purge_corrupt_thresholds()

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
    tier_accuracy   = analyze_tier_accuracy()
    context_patterns = analyze_context_patterns()
    save_learned_weights(factor_perf, market_perf, thresholds)
    _save_analysis_results(tier_accuracy, context_patterns)

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
        "tier_accuracy": tier_accuracy,
        "context_patterns": len(context_patterns),
        "best_factor":  best_factor,
        "worst_factor": worst_factor,
        "best_market":  best_market,
        "worst_market": worst_market,
        "thresholds":   thresholds,
    }
