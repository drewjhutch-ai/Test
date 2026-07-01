"""
Market anchoring for model v4.

This is the single most important module in the rework. The research is
unambiguous (Woodland & Woodland 1994; Buchdahl's ~3.4% real-world ROI ceiling;
Pinnacle/Unabated CLV literature): the no-vig closing line is a near-unbiased
estimate of true probability, and a public-data model that bets whenever it
merely *disagrees* with the market will bleed out via negative closing-line
value.

So instead of trusting the model's raw number, we:
  1. Remove the vig from both sides of the market to get the fair (no-vig)
     probability.
  2. Treat that fair probability as a strong Bayesian prior and blend the
     model's estimate *toward* it in log-odds space, giving the market most of
     the weight (default 75%).
  3. Only bet when the blended fair probability beats the market's fair
     probability by a meaningful cushion AND the price is genuinely +EV.
  4. Size with fractional (quarter) Kelly.

Everything here is pure math with no I/O so it is trivially unit-testable.
"""
from __future__ import annotations
import math

# ── Tunable constants ────────────────────────────────────────────────
# How much weight the model's own estimate gets when blended with the
# market. The market gets (1 - MODEL_WEIGHT). Retail models rarely justify
# more than ~30% of their own weight; 0.25 is a deliberately humble default.
DEFAULT_MODEL_WEIGHT = 0.25

# Minimum edge, in no-vig probability terms, of the blended estimate over the
# market's fair probability before we will bet. ~3% is the "cushion beyond the
# overround" that flipped a real public MLB model to profit in backtests.
EDGE_CUSHION = 0.03

# Fraction of full Kelly to stake. Full Kelly has a ~1/3 chance of halving the
# bankroll before doubling it; quarter-Kelly captures most of the growth with a
# fraction of the variance and buffers against our probability error.
KELLY_FRACTION = 0.25

# Probability clamp for blended output — avoids absurd extremes from bad data.
_MIN_PROB = 0.02
_MAX_PROB = 0.98


# ── Odds conversions ─────────────────────────────────────────────────

def american_to_decimal(price: int | float) -> float:
    """American odds → decimal odds."""
    price = float(price)
    if price > 0:
        return price / 100.0 + 1.0
    return 100.0 / abs(price) + 1.0


def decimal_to_american(dec: float) -> int:
    """Decimal odds → American odds (rounded to the nearest 5)."""
    if dec <= 1.0:
        return 0
    raw = (dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)
    return int(round(raw / 5.0) * 5)


def american_to_implied(price: int | float) -> float:
    """American odds → implied probability (WITH vig baked in)."""
    price = float(price)
    if price > 0:
        return 100.0 / (price + 100.0)
    return abs(price) / (abs(price) + 100.0)


def implied_to_american(prob: float) -> int:
    """Fair probability → American odds (rounded to the nearest 5)."""
    prob = min(max(prob, 1e-6), 1 - 1e-6)
    return decimal_to_american(1.0 / prob)


# ── De-vig (remove the bookmaker margin) ─────────────────────────────

def devig_two_way(home_price: int | float, away_price: int | float) -> dict:
    """
    Remove the vig from a two-way market using the standard multiplicative
    (proportional / normalization) method.

    Returns a dict with each side's no-vig fair probability plus the overround
    (hold). Example: -130 / +110  →  home ~0.543, away ~0.457, overround ~4.1%.
    """
    h_imp = american_to_implied(home_price)
    a_imp = american_to_implied(away_price)
    overround = h_imp + a_imp
    if overround <= 0:
        return {"home": 0.5, "away": 0.5, "overround": 0.0, "valid": False}
    return {
        "home": h_imp / overround,
        "away": a_imp / overround,
        "overround": overround - 1.0,
        "valid": True,
    }


# ── Log-odds blending (model shrunk toward the market prior) ──────────

def _logit(p: float) -> float:
    p = min(max(p, _MIN_PROB), _MAX_PROB)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def blend_with_market(
    p_model: float,
    p_market_novig: float,
    model_weight: float = DEFAULT_MODEL_WEIGHT,
) -> float:
    """
    Blend the model probability with the no-vig market probability in log-odds
    space (the standard ensemble-forecasting approach — log-odds are additive):

        logit(p_blend) = w * logit(p_model) + (1 - w) * logit(p_market)

    With w = 0.25 the blended estimate only moves 25% of the way from the market
    toward the model, which automatically suppresses the model's wild
    disagreements (usually errors) while preserving its genuine information.
    """
    w = min(max(model_weight, 0.0), 1.0)
    blended = _sigmoid(w * _logit(p_model) + (1.0 - w) * _logit(p_market_novig))
    return round(min(max(blended, _MIN_PROB), _MAX_PROB), 4)


# ── Edge & staking ───────────────────────────────────────────────────

def edge_vs_market(p_blend: float, p_market_novig: float) -> float:
    """Blended fair prob minus the market's fair prob (the 'cushion')."""
    return round(p_blend - p_market_novig, 4)


def edge_vs_price(p_blend: float, price: int | float) -> float:
    """
    Blended fair prob minus the price's implied prob (with vig).
    Positive → the bet is +EV at the offered price.
    """
    return round(p_blend - american_to_implied(price), 4)


def expected_value(p_blend: float, price: int | float) -> float:
    """EV per 1 unit staked at the given price and blended probability."""
    dec = american_to_decimal(price)
    return round(p_blend * (dec - 1.0) - (1.0 - p_blend), 4)


def kelly_stake(p_blend: float, price: int | float, fraction: float = KELLY_FRACTION) -> float:
    """
    Fractional-Kelly stake as a fraction of bankroll.

        f* = (b*p - q) / b     with b = decimal - 1, q = 1 - p

    Returns 0.0 for non-positive-edge bets. Never negative.
    """
    b = american_to_decimal(price) - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p_blend
    f_star = (b * p_blend - q) / b
    return round(max(0.0, f_star * fraction), 4)


def assess_bet(
    p_model: float,
    backing_price: int | float,
    home_price: int | float,
    away_price: int | float,
    backing_is_home: bool,
    model_weight: float = DEFAULT_MODEL_WEIGHT,
    edge_cushion: float = EDGE_CUSHION,
) -> dict:
    """
    One-stop market assessment for a single side.

    Returns everything the caller needs to gate and size the bet:
      - p_market_novig : the market's fair prob for the backing side
      - p_blend        : model blended toward the market prior
      - edge_market    : p_blend - p_market_novig  (the cushion)
      - edge_price     : p_blend - implied(price)  (+ve ⇒ +EV at the price)
      - ev             : expected value per unit at the price
      - kelly          : quarter-Kelly stake fraction
      - qualifies      : True iff edge_market ≥ cushion AND edge_price > 0
      - overround      : the market hold (data-quality sanity check)
    """
    devig = devig_two_way(home_price, away_price)
    p_market = devig["home"] if backing_is_home else devig["away"]
    p_blend = blend_with_market(p_model, p_market, model_weight)

    e_market = edge_vs_market(p_blend, p_market)
    e_price = edge_vs_price(p_blend, backing_price)
    qualifies = bool(devig["valid"] and e_market >= edge_cushion and e_price > 0)

    return {
        "p_market_novig": round(p_market, 4),
        "p_blend": p_blend,
        "edge_market": e_market,
        "edge_price": e_price,
        "ev": expected_value(p_blend, backing_price),
        "kelly": kelly_stake(p_blend, backing_price),
        "qualifies": qualifies,
        "overround": round(devig["overround"], 4),
    }
