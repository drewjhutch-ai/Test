"""
Home Run Prop 5-Factor Framework.
All 5 factors must be checked before recommending a HR prop.
Max realistic single-game HR probability: ~45%.
"""
from __future__ import annotations
from dataclasses import dataclass
from .park_database import get_hr_factor


@dataclass
class HRPropInput:
    batter_name: str
    batter_hand: str           # "L" or "R"
    team: str
    pitcher_name: str
    pitcher_hand: str          # "L" or "R"
    home_team: str             # For park factor lookup
    hr_rate: float             # HRs / games played this season
    pitcher_hr9: float         # HR per 9 innings allowed
    pitcher_barrel_rate: float # 0.0 – 1.0
    pitcher_hard_hit_rate: float
    pitcher_hr_fb_rate: float
    temperature: float = 72.0
    wind_speed: float = 5.0
    wind_direction: str = "calm"
    batter_woba_vs_hand: float = 0.320  # wOBA vs this pitcher handedness
    price: int = -115


def evaluate_hr_prop(inp: HRPropInput) -> dict:
    """
    Score all 5 HR factors and compute composite probability and EV.
    """
    factor_scores: dict[str, dict] = {}
    total_score = 0.0

    # ---- Factor 1: Batter HR rate ----
    if inp.hr_rate >= 0.20:
        f1 = 0.25
        f1_note = f"Elite ({inp.hr_rate:.0%}/game)"
    elif inp.hr_rate >= 0.15:
        f1 = 0.15
        f1_note = f"Good ({inp.hr_rate:.0%}/game)"
    elif inp.hr_rate >= 0.10:
        f1 = 0.05
        f1_note = f"Below minimum ({inp.hr_rate:.0%}/game)"
    else:
        f1 = -0.05
        f1_note = f"Poor ({inp.hr_rate:.0%}/game) — consider skipping"
    factor_scores["F1_batter_hr_rate"] = {"score": f1, "note": f1_note}
    total_score += f1

    # ---- Factor 2: Pitcher HR vulnerability ----
    f2 = 0.0
    f2_notes = []
    if inp.pitcher_hr9 > 1.50:
        f2 += 0.10
        f2_notes.append(f"HR/9={inp.pitcher_hr9:.2f} (very high)")
    elif inp.pitcher_hr9 > 1.20:
        f2 += 0.05
        f2_notes.append(f"HR/9={inp.pitcher_hr9:.2f} (high)")
    if inp.pitcher_barrel_rate > 0.10:
        f2 += 0.05
        f2_notes.append(f"Barrel%={inp.pitcher_barrel_rate:.0%} (high)")
    if inp.pitcher_hr_fb_rate > 0.15:
        f2 += 0.05
        f2_notes.append(f"HR/FB={inp.pitcher_hr_fb_rate:.0%} (high)")
    factor_scores["F2_pitcher_vulnerability"] = {"score": f2, "note": " | ".join(f2_notes) or "Average"}
    total_score += f2

    # ---- Factor 3: Park HR factor ----
    park_factor = get_hr_factor(inp.home_team, inp.batter_hand)
    if park_factor >= 1.20:
        f3 = 0.08
        f3_note = f"Park HR factor {park_factor:.2f} — elite hitter park"
    elif park_factor >= 1.08:
        f3 = 0.04
        f3_note = f"Park HR factor {park_factor:.2f} — hitter friendly"
    elif park_factor >= 0.95:
        f3 = 0.0
        f3_note = f"Park HR factor {park_factor:.2f} — neutral"
    else:
        f3 = -0.05
        f3_note = f"Park HR factor {park_factor:.2f} — pitcher-friendly (avoid)"
    factor_scores["F3_park_factor"] = {"score": f3, "note": f3_note}
    total_score += f3

    # ---- Factor 4: Weather ----
    f4 = 0.0
    f4_notes = []
    if inp.wind_speed >= 12 and inp.wind_direction == "out":
        f4 += 0.07
        f4_notes.append(f"Wind {inp.wind_speed:.0f}mph out — strong HR boost")
    elif inp.wind_speed >= 8 and inp.wind_direction == "out":
        f4 += 0.04
        f4_notes.append(f"Wind {inp.wind_speed:.0f}mph out — moderate HR boost")
    elif inp.wind_direction == "in":
        f4 -= 0.04
        f4_notes.append("Wind in — HR suppressed")
    # Wind blowing to pull side of batter
    if (inp.batter_hand == "L" and inp.wind_direction == "L-R") or \
       (inp.batter_hand == "R" and inp.wind_direction == "R-L"):
        f4 += 0.03
        f4_notes.append("Wind toward pull side — directional HR boost")
    if inp.temperature >= 80:
        f4 += 0.03
        f4_notes.append(f"Hot ({inp.temperature:.0f}°F) — ball carries")
    elif inp.temperature < 55:
        f4 -= 0.03
        f4_notes.append(f"Cold ({inp.temperature:.0f}°F) — ball suppressed")
    factor_scores["F4_weather"] = {"score": f4, "note": " | ".join(f4_notes) or "Neutral"}
    total_score += f4

    # ---- Factor 5: Handedness wOBA split ----
    if inp.batter_woba_vs_hand >= 0.370:
        f5 = 0.05
        f5_note = f"wOBA vs {inp.pitcher_hand}HP={inp.batter_woba_vs_hand:.3f} (elite split)"
    elif inp.batter_woba_vs_hand >= 0.320:
        f5 = 0.02
        f5_note = f"wOBA vs {inp.pitcher_hand}HP={inp.batter_woba_vs_hand:.3f} (good)"
    elif inp.batter_woba_vs_hand >= 0.280:
        f5 = 0.0
        f5_note = f"wOBA vs {inp.pitcher_hand}HP={inp.batter_woba_vs_hand:.3f} (neutral)"
    else:
        f5 = -0.03
        f5_note = f"wOBA vs {inp.pitcher_hand}HP={inp.batter_woba_vs_hand:.3f} (weak split — avoid)"
    factor_scores["F5_handedness"] = {"score": f5, "note": f5_note}
    total_score += f5

    # Composite probability (base 10% for any ML HR prop, add factor scores)
    base_prob = 0.10
    composite_prob = min(0.45, max(0.04, base_prob + total_score))

    # EV calculation
    if inp.price > 0:
        decimal = inp.price / 100 + 1
    else:
        decimal = 100 / abs(inp.price) + 1
    implied_prob = 1 / decimal
    ev_pct = (composite_prob * (decimal - 1)) - (1 - composite_prob)

    factors_passed = sum(1 for f in factor_scores.values() if f["score"] > 0)
    recommendation = "SKIP"
    if factors_passed >= 4 and composite_prob >= 0.18:
        recommendation = "PLAY (2u standalone)"
    elif factors_passed >= 3 and composite_prob >= 0.15:
        recommendation = "LEAN (1u standalone)"

    return {
        "batter": inp.batter_name,
        "pitcher": inp.pitcher_name,
        "home_team": inp.home_team,
        "factors": factor_scores,
        "factors_passed": factors_passed,
        "composite_probability": round(composite_prob, 4),
        "implied_probability": round(implied_prob, 4),
        "ev_pct": round(ev_pct, 4),
        "price": inp.price,
        "recommendation": recommendation,
        "parlay_note": "HR props can leg into parlays but max $10 stake. Lottery ticket by nature.",
        "reality_check": "Max realistic single-game HR prob is ~45%. Accept negative EV individually.",
    }
