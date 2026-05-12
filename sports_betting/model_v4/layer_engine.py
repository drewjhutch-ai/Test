"""
12-Layer analysis framework — the core decision engine of model v4.0.
Each layer is a discrete, auditable step. All 12 run in sequence.
No pick is named until Layer 1 identity check passes.
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from .pitcher_lists import is_on_fade_list, is_on_backs_list, is_era_fraud
from .park_database import get_park, is_dome, get_run_factor, wrigley_wind_signal, TEAM_TO_PARK

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data structures                                                     #
# ------------------------------------------------------------------ #

@dataclass
class PitcherProfile:
    name: str
    team: str
    era: float = 4.50
    xera: float = 4.50
    fip: float = 4.50
    xfip: float = 4.50
    siera: float = 4.50
    whip: float = 1.30
    k9: float = 8.0
    bb9: float = 3.0
    kbb_ratio: float = 0.0
    barrel_rate: float = 0.08
    swstr_pct: float = 0.10
    last_5_era: float | None = None
    is_debut: bool = False
    is_il_return: bool = False
    is_bullpen_game: bool = False
    confirmed_sources: int = 0


@dataclass
class TeamProfile:
    name: str
    wins: int = 0
    losses: int = 0
    run_diff: int = 0
    runs_per_game_last_5: float = 4.0
    bullpen_era: float = 4.00
    bullpen_era_14d: float = 4.00
    road_wins: int = 0
    road_losses: int = 0
    wrc_plus_14d: int = 100
    series_wins: int = 0   # Wins in current series (negative = losses)
    momentum: int = 0      # +N win streak, -N loss streak


@dataclass
class WeatherProfile:
    temperature: float = 72.0
    wind_speed: float = 5.0
    wind_direction: str = "calm"   # in / out / L-R / R-L / calm
    rain_pct: float = 0.0
    humidity: float = 50.0
    is_dome: bool = False


@dataclass
class LayerOutput:
    layer: int
    name: str
    passed: bool
    notes: list[str] = field(default_factory=list)
    redirects: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)


@dataclass
class PickCandidate:
    game_id: str
    home_team: str
    away_team: str
    home_pitcher: PitcherProfile
    away_pitcher: PitcherProfile
    home_team_profile: TeamProfile
    away_team_profile: TeamProfile
    weather: WeatherProfile
    backing_team: str
    proposed_market: str
    layer_outputs: list[LayerOutput] = field(default_factory=list)
    tier: str = "SKIP"
    factors: list[str] = field(default_factory=list)
    losing_scenario: str = ""
    losing_pct: float = 1.0
    recommended_market: str = ""
    ev_pct: float = 0.0
    cps_home: float = 0.0
    cps_away: float = 0.0
    cps_gap: float = 0.0
    factor_count: int = 0
    skip_reason: str = ""

    @property
    def backing_pitcher(self) -> PitcherProfile:
        return self.home_pitcher if self.backing_team == self.home_team else self.away_pitcher

    @property
    def opposing_pitcher(self) -> PitcherProfile:
        return self.away_pitcher if self.backing_team == self.home_team else self.home_pitcher

    @property
    def backing_team_profile(self) -> TeamProfile:
        return self.home_team_profile if self.backing_team == self.home_team else self.away_team_profile

    @property
    def win_pct(self) -> float:
        tp = self.backing_team_profile
        total = max(1, tp.wins + tp.losses)
        return tp.wins / total


# ------------------------------------------------------------------ #
#  Layer implementations                                               #
# ------------------------------------------------------------------ #

def layer_1_identity(pick: PickCandidate) -> LayerOutput:
    """Verify both starters from 2+ independent sources."""
    out = LayerOutput(1, "Pitcher Identity & Confirmation", passed=False)

    for pitcher, side in [(pick.home_pitcher, "Home"), (pick.away_pitcher, "Away")]:
        if pitcher.is_bullpen_game:
            out.notes.append(f"{side}: Bullpen game — no CPS analysis, use pen ERA.")
            continue
        if pitcher.confirmed_sources < 2:
            out.notes.append(
                f"WARN: {pitcher.name} ({side}) confirmed by only "
                f"{pitcher.confirmed_sources} source(s) — need 2."
            )
        if pitcher.is_debut:
            out.notes.append(f"FLAG: {pitcher.name} is making debut/first start — never parlay K props.")
        if pitcher.is_il_return:
            out.notes.append(f"FLAG: {pitcher.name} returning from IL — extra uncertainty.")
        if pitcher.name in ("TBD", ""):
            out.notes.append(f"{side} pitcher is TBD — monitor for game-time confirmation.")
            out.data["tbd"] = True
            return out

    out.passed = True
    return out


def layer_2_cps(pick: PickCandidate) -> LayerOutput:
    """
    Calculate Composite Pitching Score for both pitchers.
    CPS = SIERA*0.40 + xFIP*0.25 + FIP*0.20 + xERA*0.15 ± adjustments
    """
    out = LayerOutput(2, "Composite Pitching Score (CPS)", passed=False)

    def compute_cps(p: PitcherProfile) -> float:
        base = (p.siera * 0.40 + p.xfip * 0.25 + p.fip * 0.20 + p.xera * 0.15)
        adj = 0.0
        kbb = p.kbb_ratio or (p.k9 / max(0.1, p.bb9))
        if kbb > 2.0:
            adj -= 0.30
        elif kbb < 1.0:
            adj += 0.30
        if p.whip > 1.40:
            adj += 0.50
        elif p.whip < 1.00:
            adj -= 0.30
        if p.barrel_rate > 0.12:
            adj += 0.25
        elif p.barrel_rate < 0.05:
            adj -= 0.25
        if p.swstr_pct > 0.15:
            adj -= 0.30
        return round(base + adj, 3)

    cps_home = compute_cps(pick.home_pitcher)
    cps_away = compute_cps(pick.away_pitcher)
    gap = abs(cps_home - cps_away)
    advantage = "home" if cps_home < cps_away else "away"

    pick.cps_home = cps_home
    pick.cps_away = cps_away
    pick.cps_gap = gap

    out.data = {
        "cps_home": cps_home,
        "cps_away": cps_away,
        "gap": gap,
        "advantage": advantage,
    }

    if gap < 0.75:
        out.notes.append(f"CPS gap {gap:.2f} < 0.75 — insufficient edge. Skip pitcher-based picks.")
        return out

    confidence = "MINIMUM" if gap < 1.50 else "MEDIUM" if gap < 2.50 else "STRONG"
    out.notes.append(
        f"CPS: Home {cps_home:.2f} | Away {cps_away:.2f} | Gap {gap:.2f} ({confidence})"
    )
    out.notes.append(f"Pitching edge: {advantage.upper()} pitcher ({gap:.2f} CPS gap)")

    # ERA fraud check on both pitchers
    for pitcher, label in [(pick.home_pitcher, "Home"), (pick.away_pitcher, "Away")]:
        fraud = is_era_fraud(pitcher.era, pitcher.xera, pitcher.fip, pitcher.siera)
        if fraud["is_fraud"]:
            out.notes.append(
                f"ERA FRAUD DETECTED on {label} ({pitcher.name}): "
                + " | ".join(fraud["confirmations"])
            )
            out.data[f"{label.lower()}_era_fraud"] = True

    out.passed = True
    return out


def layer_3_bullpen(pick: PickCandidate) -> LayerOutput:
    """Evaluate bullpen quality for full-game picks. Skip for F5."""
    out = LayerOutput(3, "Bullpen Composite Score (CBS)", passed=True)

    if "f5" in pick.proposed_market.lower():
        out.notes.append("F5 market — Layer 3 skipped (starter quality only matters for 5 innings).")
        return out

    home_pen = pick.home_team_profile.bullpen_era
    away_pen = pick.away_team_profile.bullpen_era
    gap = away_pen - home_pen  # Positive = home bullpen advantage
    pen_advantage = "home" if gap > 0 else "away" if gap < 0 else "neutral"

    # Known bad bullpens from confirmed data
    KNOWN_BAD_PENS = {"HOU": 6.31, "TB": 5.90, "LAA": 5.75}
    home_abbr = pick.home_team[:3].upper()
    away_abbr = pick.away_team[:3].upper()
    if home_abbr in KNOWN_BAD_PENS:
        out.notes.append(f"WARN: {pick.home_team} bullpen confirmed bad ({KNOWN_BAD_PENS[home_abbr]} ERA).")
    if away_abbr in KNOWN_BAD_PENS:
        out.notes.append(f"WARN: {pick.away_team} bullpen confirmed bad ({KNOWN_BAD_PENS[away_abbr]} ERA).")

    out.data = {
        "home_pen_era": home_pen,
        "away_pen_era": away_pen,
        "gap": round(gap, 2),
        "pen_advantage": pen_advantage,
    }

    if abs(gap) >= 0.50:
        out.notes.append(
            f"Bullpen edge: {pen_advantage.upper()} ({abs(gap):.2f} ERA gap)."
        )
    else:
        out.notes.append("Bullpen: neutral (gap < 0.50).")

    return out


def layer_4_oqs(pick: PickCandidate) -> LayerOutput:
    """Offensive Quality Score — confirm backing team can actually score."""
    out = LayerOutput(4, "Offensive Quality Score (OQS)", passed=False)
    tp = pick.backing_team_profile
    total = max(1, tp.wins + tp.losses)
    win_pct = tp.wins / total
    road_total = max(1, tp.road_wins + tp.road_losses)
    road_pct = tp.road_wins / road_total
    r5 = tp.runs_per_game_last_5

    out.data = {
        "win_pct": round(win_pct, 3),
        "road_pct": round(road_pct, 3),
        "run_diff": tp.run_diff,
        "r5": r5,
    }

    is_away = pick.backing_team == pick.away_team
    market = pick.proposed_market.lower()

    # F5 and K props have relaxed OQS
    if "f5" in market or "k_over" in market or "er_under" in market:
        out.passed = True
        out.notes.append("Relaxed OQS for F5/K prop — positive run differential sufficient.")
        if tp.run_diff < 0:
            out.notes.append(f"WARN: Negative run differential ({tp.run_diff}).")
        return out

    # ERA fraud fade: check backing team scored 4+ R/game last 5
    if "fade" in market or pick.proposed_market == "era_fraud_fade":
        if r5 < 4.0:
            out.notes.append(
                f"OQS FAIL: ERA fraud fade requires backing team to score 4+ R/game last 5. "
                f"Current: {r5:.1f}. DOWNGRADE to lean."
            )
            out.redirects.append("Downgrade to lean. Need additional confirming factor.")
        else:
            out.passed = True
            out.notes.append(f"OQS PASS: Backing team scoring {r5:.1f} R/game last 5.")
        return out

    # Full-game ML: need .520+ record OR positive run diff
    if win_pct >= 0.520:
        out.passed = True
        out.notes.append(f"OQS PASS: Win% {win_pct:.3f} ≥ .520.")
    elif tp.run_diff > 0:
        out.passed = True
        out.notes.append(f"OQS PASS: Positive run differential ({tp.run_diff:+d}).")
    else:
        out.notes.append(
            f"OQS FAIL: Win% {win_pct:.3f} < .500 and run diff {tp.run_diff:+d}. "
            "Redirect to F5 market."
        )
        out.redirects.append("Team below .500 with negative run diff — use F5 ML or K prop.")
        return out

    if is_away and road_pct < 0.450:
        out.notes.append(
            f"WARN: Away team road record {tp.road_wins}-{tp.road_losses} ({road_pct:.3f}) below .450."
        )
        out.redirects.append("Road record below .450 — apply lean downgrade.")

    return out


def layer_5_ev(pick: PickCandidate, book_price: int, true_probability: float) -> LayerOutput:
    """Calculate Expected Value at current market price."""
    out = LayerOutput(5, "Expected Value (EV)", passed=False)

    if book_price > 0:
        decimal = book_price / 100 + 1
    else:
        decimal = 100 / abs(book_price) + 1

    implied_prob = 1 / decimal
    ev_pct = (true_probability * (decimal - 1)) - (1 - true_probability)
    edge = true_probability - implied_prob

    out.data = {
        "book_price": book_price,
        "decimal": round(decimal, 4),
        "implied_prob": round(implied_prob, 4),
        "true_probability": round(true_probability, 4),
        "edge": round(edge, 4),
        "ev_pct": round(ev_pct, 4),
    }

    pick.ev_pct = ev_pct

    if ev_pct >= 0.08:
        tier, label = "STRONG", f"+{ev_pct:.1%} EV (STRONG ≥ +8%)"
    elif ev_pct >= 0.05:
        tier, label = "MEDIUM", f"+{ev_pct:.1%} EV (MEDIUM ≥ +5%)"
    elif ev_pct >= 0.03:
        tier, label = "LEAN", f"+{ev_pct:.1%} EV (LEAN ≥ +3%)"
    else:
        tier, label = "SKIP", f"{ev_pct:.1%} EV (below +3% threshold)"

    out.notes.append(
        f"EV: True prob {true_probability:.1%} vs implied {implied_prob:.1%} "
        f"→ Edge {edge:+.1%} → {label}"
    )
    out.data["tier_from_ev"] = tier
    out.passed = ev_pct >= 0.03
    return out


def layer_6_context(pick: PickCandidate) -> LayerOutput:
    """Series context, momentum, and 7-day rolling cap check."""
    out = LayerOutput(6, "Series Context & Situational Flags", passed=True)
    tp = pick.backing_team_profile

    series_wins = tp.series_wins  # Positive = wins, negative = losses in series
    momentum = tp.momentum

    if series_wins == 2 and pick.proposed_market in ("h2h", "full_game_ml", "f5_ml"):
        out.notes.append(
            "FLAG: Backing team won G1 and G2 — attempting sweep. "
            "PARLAY BLOCKED per hard rules. Standalone 1u lean only."
        )
        out.redirects.append("Sweep attempt G3 — remove from parlay, keep as 1u standalone max.")

    if series_wins <= -2:
        out.passed = False
        out.notes.append(
            "STOP: Backing team lost 2+ consecutive games in this series. "
            "BLOCKED — backing a team with 2-game series losing streak."
        )
        return out

    if series_wins == 1:
        out.notes.append("POSITIVE: Backing team won G1 — series momentum in favor.")
        out.data["series_momentum"] = "+1"
    elif series_wins == -1:
        out.notes.append("NEUTRAL: Backing team lost G1 — rubber match, no series edge.")

    if momentum >= 3:
        out.notes.append(f"HOT STREAK: {pick.backing_team} on {momentum}-game winning streak.")
    elif momentum <= -3:
        out.notes.append(f"COLD STREAK: {pick.backing_team} on {abs(momentum)}-game losing streak.")
        out.redirects.append("Consider fading this team or using F5 to isolate pitcher.")

    return out


def layer_7_weather(pick: PickCandidate) -> LayerOutput:
    """Apply weather data to adjust pick direction and confidence."""
    out = LayerOutput(7, "Weather & Park Environment", passed=True)

    if pick.weather.is_dome or is_dome(pick.home_team):
        out.notes.append(f"DOME/ROOF — Layer 7 weather adjustments skipped.")
        out.data["dome"] = True
        return out

    w = pick.weather
    adjustments: list[str] = []

    # Rain
    if w.rain_pct >= 50:
        out.passed = False
        out.notes.append(f"RAIN {w.rain_pct:.0f}% — EXCLUDE ENTIRELY. Do not bet.")
        return out
    elif w.rain_pct >= 30:
        adjustments.append("PARLAY EXCLUDED — rain 30-50%")
    elif w.rain_pct >= 20:
        adjustments.append(f"Rain {w.rain_pct:.0f}% — monitor. Exclude if rising.")

    # Temperature
    if w.temperature < 55:
        adjustments.append(f"Temp {w.temperature:.0f}°F < 55 → NRFI/UNDER lean, K props safe")
    elif w.temperature < 70:
        pass  # Neutral
    elif w.temperature >= 80:
        adjustments.append(f"Temp {w.temperature:.0f}°F ≥ 80 → HR/YRFI/OVER boost")
    elif w.temperature >= 70:
        adjustments.append(f"Temp {w.temperature:.0f}°F → slight hitter lean at open parks")

    # Wind — Wrigley special case
    if "cubs" in pick.home_team.lower() or "wrigley" in pick.home_team.lower():
        signal = wrigley_wind_signal(w.wind_speed, w.wind_direction)
        adjustments.append(f"WRIGLEY WIND: {signal}")
    else:
        if w.wind_speed >= 15:
            if w.wind_direction == "out":
                adjustments.append(f"Wind {w.wind_speed:.0f}mph OUT — STRONG YRFI/OVER/HR boost")
            elif w.wind_direction == "in":
                adjustments.append(f"Wind {w.wind_speed:.0f}mph IN — STRONG NRFI/UNDER/HR suppress")
        elif w.wind_speed >= 11:
            if w.wind_direction == "out":
                adjustments.append(f"Wind {w.wind_speed:.0f}mph out — YRFI/OVER lean, HR upgrade")
            elif w.wind_direction == "in":
                adjustments.append(f"Wind {w.wind_speed:.0f}mph in — NRFI/UNDER lean")
            elif w.wind_direction == "L-R":
                adjustments.append("Wind L-R — LHB pull power boost")
            elif w.wind_direction == "R-L":
                adjustments.append("Wind R-L — RHB pull power boost")

    out.notes.extend(adjustments)
    out.data["adjustments"] = adjustments
    out.data["weather_summary"] = (
        f"{w.temperature:.0f}°F | Wind {w.wind_speed:.0f}mph {w.wind_direction} | Rain {w.rain_pct:.0f}%"
    )
    return out


def layer_8_bet_type(pick: PickCandidate) -> LayerOutput:
    """Select the optimal market based on all prior layers."""
    out = LayerOutput(8, "Bet Type Selection", passed=True)
    tp = pick.backing_team_profile
    total = max(1, tp.wins + tp.losses)
    win_pct = tp.wins / total
    r5 = tp.runs_per_game_last_5
    k9 = pick.backing_pitcher.k9
    home_sp = pick.home_pitcher
    away_sp = pick.away_pitcher

    recommended = pick.proposed_market

    # On backs list — use its preferred market
    on_backs, backs_data = is_on_backs_list(pick.backing_pitcher.name)
    if on_backs and backs_data.get("preferred_market"):
        recommended = backs_data["preferred_market"]
        out.notes.append(f"Permanent backs: {pick.backing_pitcher.name} → {recommended}")

    # Elite pitcher on winning team
    elif win_pct >= 0.520 and r5 >= 4.0 and pick.cps_gap >= 1.50:
        recommended = "Full-game ML"
        out.notes.append(f"Elite pitcher + winning team → Full-game ML")

    # Elite pitcher on losing team
    elif win_pct < 0.500 and pick.cps_gap >= 1.50:
        recommended = "F5 ML"
        out.notes.append(f"Elite pitcher + sub-.500 team → F5 ML (isolates pitcher quality)")

    # High K rate
    elif k9 >= 9.5:
        expected_ks = k9 / 9 * 6  # Assuming 6 innings
        out.notes.append(
            f"K/9 = {k9:.1f} → K Over prop. Expected ~{expected_ks:.1f} Ks in 6 IP."
        )
        recommended = "K Over prop"

    # ERA fraud fade
    fraud = is_era_fraud(
        pick.opposing_pitcher.era,
        pick.opposing_pitcher.xera,
        pick.opposing_pitcher.fip,
        pick.opposing_pitcher.siera,
    )
    if fraud["is_fraud"] and recommended not in ("K Over prop",):
        recommended = "Opponent ML or F5 ML (ERA fraud fade)"
        out.notes.append("ERA fraud confirmed on opposing pitcher → back the opponent.")

    pick.recommended_market = recommended
    out.data["recommended_market"] = recommended
    return out


def layer_9_sharp_money(
    pick: PickCandidate,
    handle_pct: float | None = None,
    ticket_pct: float | None = None,
    line_moved_toward_backing: bool | None = None,
) -> LayerOutput:
    """Check for sharp money signals supporting or contradicting the pick."""
    out = LayerOutput(9, "Sharp Money Check", passed=True)

    if handle_pct is None and ticket_pct is None:
        out.notes.append("Sharp data unavailable — proceeding with standard analysis.")
        return out

    # Reverse line movement check
    if line_moved_toward_backing is False:
        out.notes.append(
            "WARN: Line moved AGAINST the backing side. Potential sharp money on other side. "
            "Reconsider pick — note explicitly."
        )
        out.data["sharp_against"] = True
    elif line_moved_toward_backing is True:
        out.notes.append("POSITIVE: Line moved toward backing side — sharp confirmation.")
        out.data["sharp_confirm"] = True

    # Handle vs ticket gap (RLM signal)
    if handle_pct is not None and ticket_pct is not None:
        gap = abs(handle_pct - ticket_pct)
        if gap >= 15:
            sharp_side = "backing" if handle_pct > ticket_pct else "opposing"
            out.notes.append(
                f"Handle {handle_pct:.0f}% vs tickets {ticket_pct:.0f}% "
                f"— {gap:.0f}pt gap. Sharp money on {sharp_side} side."
            )
            if sharp_side == "opposing":
                out.notes.append("WARN: Sharps are on the OTHER side.")
                out.data["sharp_against"] = True
            else:
                out.data["sharp_confirm"] = True

    return out


def layer_10_losing_scenario(
    pick: PickCandidate,
    losing_scenario_text: str,
    estimated_lose_pct: float,
) -> LayerOutput:
    """Explicitly define how this pick loses and gate on probability."""
    out = LayerOutput(10, "Losing Scenario Assessment", passed=False)

    pick.losing_scenario = losing_scenario_text
    pick.losing_pct = estimated_lose_pct

    tier_map = {
        "STRONG":  (0.00, 0.25),
        "MEDIUM":  (0.25, 0.32),
        "LEAN":    (0.32, 0.40),
        "SKIP":    (0.40, 1.00),
    }

    assigned_tier = "SKIP"
    for tier, (lo, hi) in tier_map.items():
        if lo <= estimated_lose_pct < hi:
            assigned_tier = tier
            break

    out.data["tier"] = assigned_tier
    out.data["lose_pct"] = estimated_lose_pct
    out.notes.append(
        f"Losing scenario ({estimated_lose_pct:.0%}): {losing_scenario_text}"
    )
    out.notes.append(f"Tier from lose%: {assigned_tier}")

    if assigned_tier != "SKIP":
        pick.tier = assigned_tier
        out.passed = True
    else:
        out.notes.append("SKIP: Lose probability > 40%. Do not bet.")

    return out


def layer_11_factor_count(pick: PickCandidate, confirmed_factors: list[str]) -> LayerOutput:
    """Count independent confirming factors and validate the tier assignment."""
    out = LayerOutput(11, "Independent Factor Count", passed=False)

    INVALID = {
        "ERA alone", "Win-loss record alone", "Like the team",
        "Parlay fill", "gut feeling",
    }

    valid = [f for f in confirmed_factors if not any(inv.lower() in f.lower() for inv in INVALID)]
    count = len(valid)
    pick.factor_count = count
    pick.factors = valid

    minimums = {"STRONG": 7, "MEDIUM": 5, "LEAN": 3}
    required = minimums.get(pick.tier, 3)

    out.data = {"count": count, "required": required, "factors": valid}
    out.notes.append(f"Factors: {count} confirmed (need {required} for {pick.tier})")
    for i, f in enumerate(valid, 1):
        out.notes.append(f"  {i}. {f}")

    if count >= required:
        out.passed = True
    else:
        gap = required - count
        if count >= minimums.get("LEAN", 3):
            pick.tier = "LEAN"
            out.notes.append(f"Downgraded to LEAN (only {count} factors vs {required} required for {pick.tier}).")
            out.passed = True
        else:
            pick.tier = "SKIP"
            pick.skip_reason = f"Only {count} confirming factors (need ≥3 for LEAN)"
            out.notes.append(f"SKIP: Insufficient factors ({count}/{required}).")

    return out


def layer_12_limits(card: list[PickCandidate]) -> LayerOutput:
    """Apply final card limits — max 8 picks, max 3 STRONG, max 20u total exposure."""
    out = LayerOutput(12, "Card Limits & Output Control", passed=True)

    LIMITS = {
        "max_picks": 8,
        "max_strong": 3,
        "max_units": 20,
        "unit_by_tier": {"STRONG": 3, "MEDIUM": 2, "LEAN": 1},
    }

    active = [p for p in card if p.tier != "SKIP"]
    # Sort by strength
    active.sort(key=lambda p: {"STRONG": 0, "MEDIUM": 1, "LEAN": 2}.get(p.tier, 3))

    total_units = 0
    strong_count = 0
    final = []

    for pick in active:
        units = LIMITS["unit_by_tier"].get(pick.tier, 1)
        if pick.tier == "STRONG" and strong_count >= LIMITS["max_strong"]:
            pick.tier = "MEDIUM"
            pick.skip_reason = "Downgraded: already at max 3 STRONG picks"
            units = 2
        if total_units + units > LIMITS["max_units"]:
            pick.tier = "SKIP"
            pick.skip_reason = "Card exposure limit reached (20u max)"
            continue
        if len(final) >= LIMITS["max_picks"]:
            pick.tier = "SKIP"
            pick.skip_reason = "Card pick limit reached (8 max)"
            continue

        total_units += units
        if pick.tier == "STRONG":
            strong_count += 1
        final.append(pick)

    out.data = {
        "total_picks": len(final),
        "total_units": total_units,
        "strong_count": strong_count,
        "skipped": len(card) - len(final),
    }
    out.notes.append(
        f"Card: {len(final)} picks | {total_units}u total | {strong_count} STRONG"
    )

    return out


# ------------------------------------------------------------------ #
#  Full pipeline                                                       #
# ------------------------------------------------------------------ #

def run_all_layers(
    pick: PickCandidate,
    book_price: int,
    true_probability: float,
    losing_scenario: str,
    losing_pct: float,
    confirmed_factors: list[str],
    handle_pct: float | None = None,
    ticket_pct: float | None = None,
    line_moved_toward_backing: bool | None = None,
) -> PickCandidate:
    """
    Execute all 12 layers in sequence for one pick candidate.
    Returns the pick with tier, market, factors, and all layer outputs attached.
    """
    layers_fn = [
        lambda: layer_1_identity(pick),
        lambda: layer_2_cps(pick),
        lambda: layer_3_bullpen(pick),
        lambda: layer_4_oqs(pick),
        lambda: layer_5_ev(pick, book_price, true_probability),
        lambda: layer_6_context(pick),
        lambda: layer_7_weather(pick),
        lambda: layer_8_bet_type(pick),
        lambda: layer_9_sharp_money(pick, handle_pct, ticket_pct, line_moved_toward_backing),
        lambda: layer_10_losing_scenario(pick, losing_scenario, losing_pct),
        lambda: layer_11_factor_count(pick, confirmed_factors),
    ]

    for fn in layers_fn:
        result = fn()
        pick.layer_outputs.append(result)
        # Layer 1 failure = hard stop
        if result.layer == 1 and not result.passed and result.data.get("tbd"):
            pick.tier = "SKIP"
            pick.skip_reason = "Pitcher TBD — cannot analyze"
            break

    return pick
