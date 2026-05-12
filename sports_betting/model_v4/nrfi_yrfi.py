"""
NRFI/YRFI first-inning framework.
Calculates first-inning run probability for both directions.
Confirmed high-probability contexts are hard-coded from 24-day data.
"""
from __future__ import annotations
from dataclasses import dataclass
from .park_database import get_park, is_dome, TEAM_TO_PARK


@dataclass
class NrfiProfile:
    game_id: str
    home_team: str
    away_team: str
    home_pitcher_name: str
    away_pitcher_name: str
    home_pitcher_era: float = 4.50
    away_pitcher_era: float = 4.50
    home_pitcher_whip: float = 1.30
    away_pitcher_whip: float = 1.30
    home_lineup_fi_rate: float = 0.50   # First-inning scoring rate (team R/game fi)
    away_lineup_fi_rate: float = 0.50
    temperature: float = 72.0
    wind_speed: float = 5.0
    wind_direction: str = "calm"
    rain_pct: float = 0.0


# Confirmed NRFI contexts from 24 days of data
CONFIRMED_NRFI_CONTEXTS = [
    ("Tampa Bay Rays", "McClanahan", "86% career NRFI at Tropicana"),
    ("San Diego Padres", "", "SD 0.11 first-inning R/game historically"),
    ("Seattle Mariners", "", "T-Mobile lowest park factor (0.816)"),
    ("Miami Marlins", "Sanchez", "83.9% NRFI rate at loanDepot"),
]

CONFIRMED_YRFI_CONTEXTS = [
    ("Chicago Cubs", "", "Wrigley wind out >10mph → strong YRFI"),
    ("Cincinnati Reds", "", "GABP #1 LHB HR park → YRFI lean"),
    ("Colorado Rockies", "", "Coors field →  automatic YRFI lean (check wind dir)"),
]


def compute_nrfi_probability(profile: NrfiProfile) -> dict:
    """
    Calculate NRFI probability using:
    - Both pitchers' ERA/WHIP
    - Both lineups' first-inning scoring rates
    - Park factor
    - Weather
    - Dome bonus
    - Confirmed contexts
    """
    # Base probability from pitcher quality
    avg_era = (profile.home_pitcher_era + profile.away_pitcher_era) / 2
    avg_whip = (profile.home_pitcher_whip + profile.away_pitcher_whip) / 2

    # ERA-to-NRFI probability mapping (rough heuristic)
    if avg_era <= 2.50:
        base_nrfi = 0.72
    elif avg_era <= 3.00:
        base_nrfi = 0.68
    elif avg_era <= 3.50:
        base_nrfi = 0.65
    elif avg_era <= 4.00:
        base_nrfi = 0.62
    elif avg_era <= 4.50:
        base_nrfi = 0.58
    else:
        base_nrfi = 0.52

    # WHIP adjustment
    if avg_whip < 1.00:
        base_nrfi += 0.04
    elif avg_whip > 1.40:
        base_nrfi -= 0.05

    # Lineup first-inning rate adjustment
    avg_fi_rate = (profile.home_lineup_fi_rate + profile.away_lineup_fi_rate) / 2
    if avg_fi_rate < 0.40:
        base_nrfi += 0.04
    elif avg_fi_rate > 0.65:
        base_nrfi -= 0.04

    # Park factor
    park = get_park(profile.home_team)
    run_factor = park.get("run_factor", 1.00)
    if run_factor >= 1.10:
        base_nrfi -= 0.04
    elif run_factor <= 0.92:
        base_nrfi += 0.04

    # Dome bonus (most reliable NRFI setting)
    if is_dome(profile.home_team) or park.get("type") == "dome":
        base_nrfi += 0.05

    # Weather adjustments (outdoor only)
    if not is_dome(profile.home_team):
        if profile.temperature < 55:
            base_nrfi += 0.04
        if profile.wind_speed >= 12 and profile.wind_direction == "out":
            base_nrfi -= 0.06  # More likely to score
        if profile.wind_speed >= 12 and profile.wind_direction == "in":
            base_nrfi += 0.05
        if profile.rain_pct >= 30:
            base_nrfi = 0.0  # Exclude game

    # Confirmed McClanahan / Tropicana context
    if "rays" in profile.home_team.lower() and "mcclanahan" in profile.home_pitcher_name.lower():
        base_nrfi = max(base_nrfi, 0.82)

    base_nrfi = round(min(0.92, max(0.0, base_nrfi)), 4)

    # Parlay eligibility tier
    if base_nrfi >= 0.80:
        tier = "ANCHOR QUALITY (5-leg eligible)"
    elif base_nrfi >= 0.73:
        tier = "MEDIUM (5-6 leg eligible)"
    elif base_nrfi >= 0.70:
        tier = "LEAN (6th leg max)"
    else:
        tier = "STANDALONE ONLY (< 70%)"

    parlay_eligible = base_nrfi >= 0.70

    return {
        "nrfi_probability": base_nrfi,
        "yrfi_probability": round(1 - base_nrfi, 4),
        "lean": "NRFI" if base_nrfi >= 0.55 else "YRFI",
        "tier": tier,
        "parlay_eligible": parlay_eligible,
        "avg_pitcher_era": round(avg_era, 2),
        "avg_pitcher_whip": round(avg_whip, 2),
        "park_type": park.get("type", "outdoor"),
        "weather_summary": (
            f"{profile.temperature:.0f}°F | Wind {profile.wind_speed:.0f}mph "
            f"{profile.wind_direction} | Rain {profile.rain_pct:.0f}%"
        ),
    }


def rank_games_for_nrfi_parlay(profiles: list[NrfiProfile]) -> list[dict]:
    """
    Rank all today's games by NRFI probability.
    Returns sorted list, parlay-eligible games first.
    """
    results = []
    for p in profiles:
        analysis = compute_nrfi_probability(p)
        analysis["game"] = f"{p.away_team} @ {p.home_team}"
        analysis["game_id"] = p.game_id
        analysis["home_pitcher"] = p.home_pitcher_name
        analysis["away_pitcher"] = p.away_pitcher_name
        results.append(analysis)

    results.sort(key=lambda x: x["nrfi_probability"], reverse=True)
    return results


def build_nrfi_parlay(ranked_games: list[dict], legs: int = 5) -> dict | None:
    """
    Build an NRFI-only parlay from the highest-confidence games.
    Only uses games with parlay_eligible=True.
    Assumes NRFI pays roughly -130 on average (~1.769 decimal).
    """
    eligible = [g for g in ranked_games if g.get("parlay_eligible") and g.get("rain_pct_ok", True)]

    if len(eligible) < legs:
        return None

    chosen = eligible[:legs]
    combined_prob = 1.0
    combined_decimal = 1.0
    NRFI_TYPICAL_DECIMAL = 1.769  # Roughly -130

    for g in chosen:
        combined_prob *= g["nrfi_probability"]
        combined_decimal *= NRFI_TYPICAL_DECIMAL

    ev_pct = (combined_prob * combined_decimal) - 1

    american = int((combined_decimal - 1) * 100) if combined_decimal >= 2.0 else int(-100 / (combined_decimal - 1))
    stake = 20 if legs == 5 else 10

    return {
        "type": f"{legs}-Leg NRFI Parlay",
        "legs": chosen,
        "combined_probability": round(combined_prob, 4),
        "combined_decimal": round(combined_decimal, 2),
        "american_odds": f"+{american}",
        "ev_pct": round(ev_pct, 3),
        "recommended_stake": f"${stake}",
        "potential_win": round((combined_decimal - 1) * stake, 2),
        "note": "Dome games preferred. Combine with elite starters.",
    }


def classify_yrfi_lean(profile: NrfiProfile) -> dict:
    """Classify YRFI lean conditions for a game."""
    analysis = compute_nrfi_probability(profile)
    yrfi_prob = analysis["yrfi_probability"]

    lean = "NEUTRAL"
    reasons = []

    if profile.home_pitcher_era > 4.50:
        reasons.append(f"Home SP ERA {profile.home_pitcher_era:.2f} > 4.50")
    if profile.away_pitcher_era > 4.50:
        reasons.append(f"Away SP ERA {profile.away_pitcher_era:.2f} > 4.50")
    if profile.wind_speed >= 10 and profile.wind_direction == "out":
        reasons.append(f"Wind {profile.wind_speed:.0f}mph blowing out")
    if profile.temperature >= 80:
        reasons.append(f"Hot ({profile.temperature:.0f}°F) at open park")
    park = get_park(profile.home_team)
    if park.get("run_factor", 1.0) >= 1.10:
        reasons.append(f"Hitter-friendly park (run factor {park.get('run_factor'):.2f})")

    if yrfi_prob >= 0.55 and len(reasons) >= 2:
        lean = "YRFI LEAN"
    if yrfi_prob >= 0.60 and len(reasons) >= 3:
        lean = "STRONG YRFI"

    return {
        "lean": lean,
        "yrfi_probability": yrfi_prob,
        "reasons": reasons,
        "standalone_only": True,  # YRFI never automatically goes in NRFI parlays
    }
