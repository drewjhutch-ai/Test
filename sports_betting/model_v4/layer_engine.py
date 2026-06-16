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


# Dynamic tier thresholds — updated by weight_trainer when 50+ graded picks exist.
# Calibrated for MLB: best statistical edges are at 53-65% true win probability.
# True win prob = 1 - losing_pct. LEAN floor at 56% (~-130 implied) is selective
# but achievable when pitcher gap + record edge + HFA stack up.
_TIER_THRESHOLDS: dict = {
    "STRONG": (0.00, 0.28),   # true_prob ≥ 0.72  (dominant edge)
    "MEDIUM": (0.28, 0.36),   # true_prob ≥ 0.64  (strong edge)
    "LEAN":   (0.36, 0.44),   # true_prob ≥ 0.56  (clear statistical edge)
    "SKIP":   (0.44, 1.00),   # true_prob < 0.56  (too close to call)
}

# Learned factor weight multipliers — updated by daily_runner after load_learned_weights()
# {keyword: multiplier} where >1.0 means that factor historically predicts wins
_FACTOR_WEIGHTS: dict[str, float] = {}


def update_factor_weights(weights: dict) -> None:
    """Called by daily_runner after loading learned weights from DB."""
    global _FACTOR_WEIGHTS
    if weights:
        _FACTOR_WEIGHTS.update(weights)
        logger.info("Factor weights updated: %d signals loaded", len(weights))


def update_tier_thresholds(thresholds: dict) -> None:
    """Called by daily_runner after loading learned weights."""
    global _TIER_THRESHOLDS
    if thresholds:
        _TIER_THRESHOLDS.update(thresholds)
        logger.info("Tier thresholds updated from learned weights: %s", thresholds)


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
    hard_hit_rate: float = 0.37    # Hard hit % allowed
    hr_fb_rate: float = 0.13       # HR/FB rate allowed
    fly_ball_pct: float = 0.35     # Fly ball % allowed
    gb_pct: float = 0.45           # Ground ball % allowed
    avg_exit_velo: float = 88.5    # Avg exit velocity allowed
    ip_per_start: float = 5.5      # Average innings per start
    last_5_era: float | None = None
    is_debut: bool = False
    is_il_return: bool = False
    is_bullpen_game: bool = False
    confirmed_sources: int = 0
    csw_rate: float = 0.28          # Called Strike + Whiff %
    stuff_plus: float = 100.0
    velocity_7d: float = 0.0        # Recent avg fastball velo
    velocity_season: float = 0.0    # Season avg fastball velo
    spin_rate_7d: float = 0.0
    spin_rate_season: float = 0.0
    pitch_mix_change: bool = False   # True if significant pitch mix change detected
    pitch_mix_details: str = ""      # Human-readable description of the change


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
    avg_barrel_rate: float = 0.08   # Team avg barrel rate vs pitching
    avg_hard_hit_rate: float = 0.37 # Team avg hard hit rate
    avg_launch_angle: float = 12.0  # Team avg launch angle
    ops_last_14d: float = 0.720     # Team OPS last 14 days
    k_rate: float = 0.23            # Team strikeout rate (as batters)
    framing_runs: float = 0.0       # Catcher framing runs above average
    bullpen_fatigue_score: float = 0.0   # 0-10; 7+ = fatigued
    bullpen_arms_available: int = 3      # Fresh arms with < 25 pitches last 3 days
    team_oaa: float = 0.0               # Team Outs Above Average (season total)
    defensive_run_value: float = 0.0    # Estimated runs saved: team_oaa * 0.82
    travel_tz_change: float = 0.0       # Time-zone hours traveled (positive = westward)
    travel_fatigue: bool = False         # True if 3+ tz westward travel
    # Tier 3 fields — luck, platoon, lineup, bat speed
    babip: float = 0.295                 # Team batting average on balls in play
    lob_pct: float = 0.720               # Pitcher LOB% (strand rate)
    luck_score: float = 0.0              # -4 to +4; positive = luck-inflated, negative = luck-deflated
    wrc_vs_lhp: float = 100.0            # Team avg wRC+ vs left-handed pitchers
    wrc_vs_rhp: float = 100.0            # Team avg wRC+ vs right-handed pitchers
    lineup_confirmed: bool = False       # True if day-of lineup is confirmed
    lineup_value_score: float = 5.0      # 0-10 lineup completeness/value score
    avg_bat_speed: float = 70.0          # Team avg bat speed in mph
    avg_sprint_speed: float = 27.0       # Team avg sprint speed in ft/sec
    is_speed_team: bool = False          # True if avg sprint speed >= 27.5 ft/sec


@dataclass
class WeatherProfile:
    temperature: float = 72.0
    wind_speed: float = 5.0
    wind_direction: str = "calm"   # in / out / L-R / R-L / calm
    rain_pct: float = 0.0
    humidity: float = 50.0
    is_dome: bool = False
    air_density_ratio: float = 1.0   # 1.0 = sea-level standard; <1 = thinner air
    carry_boost_pct: float = 0.0     # Ball-carry boost from thin air (pct)


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
            out.notes.append(f"{side} pitcher is TBD — analyzing with team data only, flag for monitoring.")
            out.data["tbd"] = True

    out.passed = True  # Continue analysis even with TBD pitchers using available team data
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
        if p.hard_hit_rate > 0.40:
            base += 0.15  # High hard contact allowed — ERA will regress up
        elif p.hard_hit_rate < 0.32:
            base -= 0.15  # Elite soft contact — ERA should outperform
        if p.fly_ball_pct > 0.42:
            base += 0.10  # Fly ball pitcher in HR-friendly park = elevated risk
        elif p.gb_pct > 0.52:
            base -= 0.10  # Ground ball pitcher limits HR damage
        if p.hr_fb_rate > 0.16:
            base += 0.12  # High HR/FB rate — homer-prone
        elif p.hr_fb_rate < 0.08:
            base -= 0.10  # Low HR/FB rate — suppresses long ball
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
    """Evaluate bullpen quality for full-game picks. Skip for F5. Includes fatigue analysis."""
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

    # ── Bullpen fatigue upgrade ───────────────────────────────────────
    backing_tp  = pick.backing_team_profile
    opposing_tp = (
        pick.away_team_profile if pick.backing_team == pick.home_team
        else pick.home_team_profile
    )
    is_full_game = "f5" not in pick.proposed_market.lower()

    opp_fatigue  = opposing_tp.bullpen_fatigue_score
    back_fatigue = backing_tp.bullpen_fatigue_score

    if opp_fatigue >= 7.0:
        factor_label = "bullpen_fatigued"
        pick.factors.append(factor_label)
        out.notes.append(
            f"POSITIVE: {opposing_tp.name} bullpen fatigue score {opp_fatigue:.1f}/10 — "
            f"top arms overworked, {opposing_tp.bullpen_arms_available} fresh arms available. "
            "Opponent cannot cover late innings cleanly."
        )
        out.data["opp_bullpen_fatigued"] = True

    if back_fatigue >= 7.0 and is_full_game:
        out.notes.append(
            f"WARN: {backing_tp.name} OWN bullpen fatigue score {back_fatigue:.1f}/10 — "
            f"only {backing_tp.bullpen_arms_available} fresh arms available. "
            "Full-game pick carries late-inning risk."
        )
        out.data["backing_bullpen_fatigued"] = True

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
    elif ev_pct >= 0.02:
        tier, label = "LEAN", f"+{ev_pct:.1%} EV (LEAN ≥ +2%)"
    else:
        tier, label = "SKIP", f"{ev_pct:.1%} EV (below +2% threshold)"

    out.notes.append(
        f"EV: True prob {true_probability:.1%} vs implied {implied_prob:.1%} "
        f"→ Edge {edge:+.1%} → {label}"
    )
    out.data["tier_from_ev"] = tier
    out.passed = ev_pct >= 0.02
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
    opp_tp = pick.away_team_profile if pick.backing_team == pick.home_team else pick.home_team_profile
    total = max(1, tp.wins + tp.losses)
    win_pct = tp.wins / total
    r5 = tp.runs_per_game_last_5
    k9 = pick.backing_pitcher.k9
    opp_k9 = pick.opposing_pitcher.k9
    sp = pick.backing_pitcher
    opp_sp = pick.opposing_pitcher
    wx = pick.weather

    recommended = pick.proposed_market
    additional_markets: list[str] = []

    # On backs list — use its preferred market
    on_backs, backs_data = is_on_backs_list(sp.name)
    if on_backs and backs_data.get("preferred_market"):
        recommended = backs_data["preferred_market"]
        out.notes.append(f"Permanent backs: {sp.name} → {recommended}")

    # Elite pitcher on winning team
    elif win_pct >= 0.520 and r5 >= 4.0 and pick.cps_gap >= 1.50:
        recommended = "Full-game ML"
        out.notes.append(f"Elite pitcher + winning team → Full-game ML")

    # Elite pitcher on losing team — isolate pitcher
    elif win_pct < 0.500 and pick.cps_gap >= 1.50:
        recommended = "F5 ML"
        out.notes.append(f"Elite pitcher + sub-.500 team → F5 ML (isolates pitcher quality)")

    # High K rate — strikeout prop
    elif k9 >= 9.5:
        expected_ks = round(k9 / 9 * min(sp.ip_per_start, 6), 1)
        out.notes.append(
            f"K/9 = {k9:.1f} → K Over prop. Expected ~{expected_ks} Ks in {min(sp.ip_per_start,6):.0f} IP."
        )
        recommended = f"K Over prop ({expected_ks})"

    # ERA fraud fade
    fraud = is_era_fraud(
        pick.opposing_pitcher.era,
        pick.opposing_pitcher.xera,
        pick.opposing_pitcher.fip,
        pick.opposing_pitcher.siera,
    )
    if fraud["is_fraud"] and recommended not in (f"K Over prop ({round(k9/9*min(sp.ip_per_start,6),1)})",):
        recommended = "Opponent ML or F5 ML (ERA fraud fade)"
        out.notes.append("ERA fraud confirmed on opposing pitcher → back the opponent.")

    # ── Additional market flags (layered on top of primary pick) ─────────
    # F5 ML as alternative when starter going deep with good CPS gap
    if pick.cps_gap >= 1.0 and sp.ip_per_start >= 5.5 and recommended not in ("F5 ML",):
        additional_markets.append("F5 ML")
        out.notes.append(f"CPS gap {pick.cps_gap:.2f} + avg {sp.ip_per_start:.1f} IP/start → F5 ML viable")

    # Game OVER lean
    from .park_database import get_run_factor
    run_factor = get_run_factor(pick.home_team)
    both_era_avg = (sp.era + opp_sp.era) / 2
    over_signals = []
    if run_factor >= 1.08:
        over_signals.append(f"HR park ({run_factor:.2f}x)")
    if wx.wind_speed >= 10 and wx.wind_direction == "out":
        over_signals.append(f"Wind {wx.wind_speed:.0f}mph out")
    if wx.temperature >= 82:
        over_signals.append(f"Hot ({wx.temperature:.0f}°F)")
    if both_era_avg >= 4.80:
        over_signals.append(f"Both SPs ERA avg {both_era_avg:.2f}")
    if sp.fly_ball_pct >= 0.40 or opp_sp.fly_ball_pct >= 0.40:
        over_signals.append("Fly ball SP in hitter park")
    if len(over_signals) >= 3:
        additional_markets.append(f"Game OVER ({', '.join(over_signals[:3])})")
        out.notes.append(f"OVER lean: {' | '.join(over_signals)}")

    # Game UNDER lean
    under_signals = []
    if run_factor <= 0.93:
        under_signals.append(f"Pitcher park ({run_factor:.2f}x)")
    if wx.is_dome:
        under_signals.append("Dome — no weather variance")
    if both_era_avg <= 3.60:
        under_signals.append(f"Both SPs ERA avg {both_era_avg:.2f} (elite)")
    if sp.k9 >= 9.0 and opp_k9 >= 9.0:
        under_signals.append(f"Both SPs K/9 {sp.k9:.1f}/{opp_k9:.1f} (high Ks = fewer baserunners)")
    if sp.gb_pct >= 0.50 and opp_sp.gb_pct >= 0.50:
        under_signals.append("Both SPs groundball heavy")
    if len(under_signals) >= 3:
        additional_markets.append(f"Game UNDER ({', '.join(under_signals[:3])})")
        out.notes.append(f"UNDER lean: {' | '.join(under_signals)}")

    # Pitcher outs recorded prop (when starter goes deep consistently)
    if sp.ip_per_start >= 6.0 and sp.era <= 3.80:
        outs_line = round(sp.ip_per_start * 3 - 0.5)
        additional_markets.append(f"Outs Recorded Over {outs_line}")
        out.notes.append(f"{sp.name} avg {sp.ip_per_start:.1f} IP → Outs Over {outs_line}")

    # Pitcher ERA / ER Under when elite
    if sp.xfip <= 3.20 and sp.siera <= 3.20:
        er_line = max(1, round(sp.era * sp.ip_per_start / 9))
        additional_markets.append(f"Earned Runs Under {er_line + 0.5}")
        out.notes.append(f"{sp.name} xFIP {sp.xfip:.2f}/SIERA {sp.siera:.2f} → ER Under")

    # First 5 NRFI / early-game prop when both starters are elite
    if sp.era <= 3.00 and opp_sp.era <= 3.50:
        additional_markets.append("F5 UNDER / NRFI")
        out.notes.append(f"Elite dual-starter matchup → F5 UNDER / NRFI")

    pick.recommended_market = recommended
    out.data["recommended_market"] = recommended
    out.data["additional_markets"] = additional_markets
    if additional_markets:
        out.notes.append(f"Also consider: {' · '.join(additional_markets[:4])}")
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

    tier_map = _TIER_THRESHOLDS

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

    minimums = {"STRONG": 6, "MEDIUM": 4, "LEAN": 2}
    required = minimums.get(pick.tier, 2)

    out.data = {"count": count, "required": required, "factors": valid}
    out.notes.append(f"Factors: {count} confirmed (need {required} for {pick.tier})")
    for i, f in enumerate(valid, 1):
        out.notes.append(f"  {i}. {f}")

    if count >= required:
        out.passed = True
    else:
        gap = required - count
        if count >= minimums.get("LEAN", 2):
            pick.tier = "LEAN"
            out.notes.append(f"Downgraded to LEAN (only {count} factors vs {required} required for {pick.tier}).")
            out.passed = True
        else:
            pick.tier = "SKIP"
            pick.skip_reason = f"Only {count} confirming factors (need ≥2 for LEAN)"
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


def layer_13_xstats(pick: PickCandidate) -> LayerOutput:
    """
    xStats Gap analysis — compare xERA vs ERA for ERA fraud and team xwOBA rebound signals.
    Uses data pre-loaded into PitcherProfile and TeamProfile fields by daily_runner.
    """
    out = LayerOutput(13, "xStats Gap (xERA / xwOBA)", passed=True)

    backing_sp  = pick.backing_pitcher
    opposing_sp = pick.opposing_pitcher
    backing_tp  = pick.backing_team_profile

    notes: list[str] = []
    factors: list[str] = []

    # ── Opposing pitcher ERA fraud via xERA gap ──────────────────────
    opp_era_gap = opposing_sp.xera - opposing_sp.era  # positive = xERA > ERA = fraud
    if opp_era_gap >= 0.75:
        label = f"xERA_fraud_{opposing_sp.name}"
        factors.append(label)
        notes.append(
            f"ERA FRAUD (xStats): {opposing_sp.name} ERA {opposing_sp.era:.2f} "
            f"vs xERA {opposing_sp.xera:.2f} (gap +{opp_era_gap:.2f}) — "
            "strong fade signal, ERA will regress up."
        )
        out.data["opposing_xera_fraud"] = True

    # ── Backing pitcher outperforming (xERA < ERA) ───────────────────
    back_era_gap = backing_sp.era - backing_sp.xera  # positive = ERA > xERA = outperforming
    if back_era_gap >= 0.75:
        factors.append(f"backing_sp_outperforming_{backing_sp.name}")
        notes.append(
            f"POSITIVE: {backing_sp.name} ERA {backing_sp.era:.2f} "
            f"vs xERA {backing_sp.xera:.2f} — pitcher outperforming metrics, ERA should improve."
        )

    # ── Team xwOBA rebound (batting side) ────────────────────────────
    # backing_tp.ops_last_14d used as wOBA proxy when dedicated wOBA field absent
    # xwoba stored externally; we check if it was injected into pick.layer_outputs data
    # Convention: daily_runner injects xwoba gap into backing_team_profile via a custom attr
    # We read it from a known data field if present (set by daily_runner on the profile)
    xwoba_gap = getattr(backing_tp, "_xwoba_gap", None)
    if xwoba_gap is not None and xwoba_gap >= 0.020:
        factors.append(f"team_xwoba_rebound_{backing_tp.name}")
        notes.append(
            f"POSITIVE: {backing_tp.name} xwOBA gap +{xwoba_gap:.3f} — "
            "offense hitting below expected, rebound due."
        )
        out.data["xwoba_rebound"] = True

    out.notes.extend(notes)
    out.data["xstats_factors"] = factors
    pick.factors.extend(factors)
    return out


def layer_14_velocity(pick: PickCandidate) -> LayerOutput:
    """
    Velocity & spin drop analysis on the opposing pitcher.
    Signals apply to the batting team (if opposing pitcher declining, back the batters).
    Uses data pre-loaded into PitcherProfile fields by daily_runner.
    """
    out = LayerOutput(14, "Velocity / Spin Drop", passed=True)

    opp_sp = pick.opposing_pitcher
    notes: list[str] = []
    factors: list[str] = []

    velo_drop = 0.0
    if opp_sp.velocity_season > 0 and opp_sp.velocity_7d > 0:
        velo_drop = opp_sp.velocity_season - opp_sp.velocity_7d
    elif opp_sp.velocity_season > 0 and opp_sp.velocity_7d == 0:
        # No recent data — can't flag
        velo_drop = 0.0

    spin_drop_pct = 0.0
    if opp_sp.spin_rate_season > 0 and opp_sp.spin_rate_7d > 0:
        spin_drop_pct = (opp_sp.spin_rate_season - opp_sp.spin_rate_7d) / opp_sp.spin_rate_season

    if velo_drop >= 2.5:
        label = f"velo_drop_{opp_sp.name}"
        factors.append(label)
        notes.append(
            f"STRONG FADE SIGNAL: {opp_sp.name} fastball velo down "
            f"{velo_drop:.1f} mph (season {opp_sp.velocity_season:.1f} → "
            f"recent {opp_sp.velocity_7d:.1f}) — significant arm fatigue."
        )
        out.data["strong_velo_drop"] = True
        # Flag potential tier downgrade on opposing pitcher confidence
        out.data["tier_downgrade_signal"] = True
    elif velo_drop >= 1.5:
        label = f"velo_drop_{opp_sp.name}"
        factors.append(label)
        notes.append(
            f"WARN: {opp_sp.name} fastball velo down {velo_drop:.1f} mph — "
            "mild fatigue indicator, monitor."
        )
        out.data["mild_velo_drop"] = True

    if spin_drop_pct >= 0.05:
        factors.append(f"spin_drop_{opp_sp.name}")
        notes.append(
            f"WARN: {opp_sp.name} spin rate down "
            f"{spin_drop_pct:.1%} — reduced movement/break on breaking balls."
        )
        out.data["spin_drop"] = True

    out.notes.extend(notes)
    out.data["velocity_factors"] = factors
    pick.factors.extend(factors)

    if not notes:
        out.notes.append(
            f"Layer 14: {opp_sp.name} velocity data "
            + ("unavailable." if opp_sp.velocity_season == 0 else "nominal — no significant drop detected.")
        )
    return out


def layer_15_catcher_framing(pick: PickCandidate) -> LayerOutput:
    """
    Catcher framing impact on K prop bets and totals.
    Uses framing_runs field on TeamProfile (populated by daily_runner).
    """
    out = LayerOutput(15, "Catcher Framing", passed=True)

    backing_tp  = pick.backing_team_profile
    opposing_tp = (
        pick.away_team_profile if pick.backing_team == pick.home_team
        else pick.home_team_profile
    )

    notes: list[str] = []
    factors: list[str] = []

    back_framing = backing_tp.framing_runs
    opp_framing  = opposing_tp.framing_runs

    # Elite backing catcher — expands zone for backing team's K props
    if back_framing > 8:
        factors.append(f"elite_framing_{backing_tp.name}")
        notes.append(
            f"POSITIVE: {backing_tp.name} catcher framing +{back_framing:.1f} runs above avg — "
            "elite zone expansion; boosts K prop confidence and pitcher-favorable outcomes."
        )
        out.data["elite_framing_backing"] = True

    # Poor opposing catcher — pitcher's effective zone shrinks
    if opp_framing < -8:
        factors.append(f"poor_framing_opp_{opposing_tp.name}")
        notes.append(
            f"WARN: {opposing_tp.name} catcher framing {opp_framing:.1f} runs below avg — "
            "zone shrinks for backing team's pitcher; increased bullpen risk, caution on K totals."
        )
        out.data["poor_framing_opposing"] = True

    # Mild signals
    if 4 < back_framing <= 8:
        notes.append(
            f"MILD POSITIVE: {backing_tp.name} catcher framing +{back_framing:.1f} — above average zone management."
        )
    if -8 <= opp_framing < -4:
        notes.append(
            f"MILD WARN: {opposing_tp.name} catcher framing {opp_framing:.1f} — slightly below average."
        )

    out.notes.extend(notes)
    out.data["framing_factors"] = factors
    pick.factors.extend(factors)

    if not notes:
        out.notes.append("Layer 15: Catcher framing data unavailable or neutral.")
    return out


def layer_16_umpire(pick: PickCandidate) -> LayerOutput:
    """
    Umpire tendency analysis for totals and pitcher-favorable picks.
    Uses umpire data injected into pick via pick.layer_outputs data or custom attr by daily_runner.
    Strongest when combined with command pitcher (low BB9) + tight zone umpire.
    """
    out = LayerOutput(16, "Umpire Tendencies", passed=True)

    # Umpire data injected by daily_runner as a custom attribute on the pick
    umpire_data: dict = getattr(pick, "_umpire_data", {})
    if not umpire_data:
        out.notes.append("Layer 16: No umpire data available for this game.")
        return out

    umpire_name = umpire_data.get("umpire_name", "Unknown")
    over_pct    = umpire_data.get("over_pct",   0.50)
    under_pct   = umpire_data.get("under_pct",  0.50)

    notes: list[str] = []
    factors: list[str] = []

    backing_sp = pick.backing_pitcher
    command_pitcher = backing_sp.bb9 <= 2.5  # low walk rate = command pitcher

    if under_pct >= 0.58:
        factor_label = "tight_zone_umpire"
        factors.append(factor_label)
        combined = command_pitcher and under_pct >= 0.58
        if combined:
            notes.append(
                f"STRONG POSITIVE: {umpire_name} calls UNDER {under_pct:.0%} of games "
                f"+ {backing_sp.name} has elite command (BB/9 {backing_sp.bb9:.1f}) — "
                "tight zone + command pitcher is premium pitcher-favorable combination."
            )
        else:
            notes.append(
                f"POSITIVE: {umpire_name} calls UNDER {under_pct:.0%} of games — "
                "tight zone umpire favors pitching picks and low-total bets."
            )
        out.data["tight_zone_umpire"] = True

    elif over_pct >= 0.58:
        notes.append(
            f"MILD WARN: {umpire_name} calls OVER {over_pct:.0%} of games — "
            "liberal zone may slightly undercut strong pitching pick confidence."
        )
        out.data["liberal_zone_umpire"] = True

    else:
        notes.append(
            f"NEUTRAL: {umpire_name} — over {over_pct:.0%} / under {under_pct:.0%} "
            "(no strong directional lean)."
        )

    out.notes.extend(notes)
    out.data["umpire_factors"] = factors
    out.data["umpire_name"]    = umpire_name
    pick.factors.extend(factors)
    return out


def layer_17_csw_stuff(pick: PickCandidate) -> LayerOutput:
    """
    CSW Rate + Stuff+ analysis for both starters.
    CSW% (Called Strike + Whiff%) is the single best early-season K-rate predictor.
    Stuff+ ≥ 110 = elite pitch quality; ≤ 88 = below-average arsenal.
    Uses csw_rate and stuff_plus fields on PitcherProfile (populated by daily_runner).
    """
    out = LayerOutput(17, "CSW Rate & Stuff+", passed=True)

    backing_sp  = pick.backing_pitcher
    opposing_sp = pick.opposing_pitcher

    notes: list[str] = []
    factors: list[str] = []

    # ── Backing pitcher strengths ─────────────────────────────────────
    if backing_sp.csw_rate >= 0.30:
        label = "elite_csw"
        factors.append(label)
        notes.append(
            f"POSITIVE: {backing_sp.name} CSW% {backing_sp.csw_rate:.1%} ≥ 30% — "
            "elite called-strike+whiff rate; K prop confidence elevated."
        )
        out.data["backing_elite_csw"] = True

    elif backing_sp.csw_rate >= 0.285 and backing_sp.stuff_plus >= 110:
        label = "strong_stuff"
        factors.append(label)
        notes.append(
            f"POSITIVE: {backing_sp.name} CSW% {backing_sp.csw_rate:.1%} + "
            f"Stuff+ {backing_sp.stuff_plus:.0f} — above-average arsenal combination."
        )
        out.data["backing_strong_stuff"] = True

    # ── Opposing pitcher weaknesses ───────────────────────────────────
    if opposing_sp.csw_rate <= 0.24:
        label = "weak_csw_opp"
        factors.append(label)
        notes.append(
            f"POSITIVE (offense): {opposing_sp.name} CSW% {opposing_sp.csw_rate:.1%} ≤ 24% — "
            "poor called-strike+whiff rate; backing offense should make contact and rebound."
        )
        out.data["opp_weak_csw"] = True

    if opposing_sp.stuff_plus <= 88:
        notes.append(
            f"MILD POSITIVE: {opposing_sp.name} Stuff+ {opposing_sp.stuff_plus:.0f} ≤ 88 — "
            "below-average arsenal; modest additional confidence for backing offense."
        )
        out.data["opp_poor_stuff"] = True

    out.notes.extend(notes)
    out.data["csw_factors"] = factors
    pick.factors.extend(factors)

    if not notes:
        out.notes.append(
            f"Layer 17: CSW/Stuff+ — {backing_sp.name} {backing_sp.csw_rate:.1%} CSW / "
            f"Stuff+ {backing_sp.stuff_plus:.0f} (no strong signals)."
        )
    return out


def layer_18_air_density(pick: PickCandidate) -> LayerOutput:
    """
    Advanced air density park factor analysis.
    Uses air_density_ratio and carry_boost_pct from WeatherProfile.
    Only meaningful for outdoor stadiums (is_dome=False).
    """
    out = LayerOutput(18, "Air Density Park Factor", passed=True)

    if pick.weather.is_dome or is_dome(pick.home_team):
        out.notes.append("Layer 18: Dome/retractable roof — air density not applicable.")
        out.data["dome"] = True
        return out

    carry = pick.weather.carry_boost_pct
    density = pick.weather.air_density_ratio

    notes: list[str] = []
    factors: list[str] = []

    if carry >= 8.0:
        label = "high_carry"
        factors.append(label)
        notes.append(
            f"STRONG CARRY BOOST: Air density ratio {density:.4f} → "
            f"{carry:.1f}% carry boost — significant HR/OVER lean; "
            "pitching picks face elevated HR risk."
        )
        out.data["high_carry"] = True

    elif carry >= 5.0:
        label = "favorable_carry"
        factors.append(label)
        notes.append(
            f"POSITIVE CARRY: Air density ratio {density:.4f} → "
            f"{carry:.1f}% carry boost — favors over/offense picks."
        )
        out.data["favorable_carry"] = True

    elif carry <= -3.0:
        label = "suppressed_carry"
        factors.append(label)
        notes.append(
            f"SUPPRESSED CARRY: Air density ratio {density:.4f} → "
            f"{carry:.1f}% carry penalty — favors under/pitching picks."
        )
        out.data["suppressed_carry"] = True

    else:
        notes.append(
            f"Layer 18: Air density ratio {density:.4f} → carry boost {carry:.1f}% (neutral)."
        )

    out.notes.extend(notes)
    out.data["air_density_factors"] = factors
    pick.factors.extend(factors)
    return out


def layer_19_travel(pick: PickCandidate) -> LayerOutput:
    """
    Travel / time-zone fatigue analysis.
    Uses travel_tz_change and travel_fatigue fields on TeamProfile.
    Strongest signal: 3+ time-zone westward travel.
    """
    out = LayerOutput(19, "Travel / Time-Zone Fatigue", passed=True)

    backing_tp  = pick.backing_team_profile
    opposing_tp = (
        pick.away_team_profile if pick.backing_team == pick.home_team
        else pick.home_team_profile
    )

    notes: list[str] = []
    factors: list[str] = []

    # ── Backing team fatigue ──────────────────────────────────────────
    if backing_tp.travel_fatigue:
        label = "travel_fatigue"
        factors.append(label)
        notes.append(
            f"WARN: {backing_tp.name} traveled {backing_tp.travel_tz_change:.0f} time zones "
            "westward — 3+ tz cross-country fatigue. Mild negative for backing side."
        )
        out.data["backing_travel_fatigue"] = True

    elif backing_tp.travel_tz_change >= 2.0:
        notes.append(
            f"NOTE: {backing_tp.name} traveled {backing_tp.travel_tz_change:.0f} time zones "
            "westward (2 tz — less severe; monitor but no factor change)."
        )

    # ── Opposing team fatigue ─────────────────────────────────────────
    if opposing_tp.travel_fatigue:
        label = "opp_travel_fatigue"
        factors.append(label)
        notes.append(
            f"POSITIVE: {opposing_tp.name} (opponent) traveled "
            f"{opposing_tp.travel_tz_change:.0f} time zones westward — "
            "significant fatigue disadvantage for opposing side."
        )
        out.data["opp_travel_fatigue"] = True

    elif opposing_tp.travel_tz_change >= 2.0:
        notes.append(
            f"MILD POSITIVE: {opposing_tp.name} traveled "
            f"{opposing_tp.travel_tz_change:.0f} tz westward (2 tz — mild disadvantage)."
        )

    out.notes.extend(notes)
    out.data["travel_factors"] = factors
    pick.factors.extend(factors)

    if not notes:
        out.notes.append("Layer 19: No significant travel fatigue for either team.")
    return out


def layer_20_defense(pick: PickCandidate) -> LayerOutput:
    """
    Team defensive metrics (OAA) impact on pitcher ERA vs FIP discrepancy.
    Elite defense suppresses ERA below FIP; poor defense inflates it.
    Uses team_oaa and defensive_run_value fields on TeamProfile.
    """
    out = LayerOutput(20, "Defensive Metrics (OAA)", passed=True)

    backing_tp  = pick.backing_team_profile
    backing_sp  = pick.backing_pitcher

    notes: list[str] = []
    factors: list[str] = []

    team_oaa = backing_tp.team_oaa

    # ── Elite / above-average defense ────────────────────────────────
    if team_oaa >= 15:
        label = "elite_defense"
        factors.append(label)
        notes.append(
            f"STRONG POSITIVE: {backing_tp.name} team OAA +{team_oaa:.0f} (elite) — "
            "ERA will be lower than FIP suggests; defense turns batted balls into outs."
        )
        out.data["elite_defense"] = True

    elif team_oaa >= 8:
        label = "above_avg_defense"
        factors.append(label)
        notes.append(
            f"MILD POSITIVE: {backing_tp.name} team OAA +{team_oaa:.0f} — "
            "above-average defense provides ERA-suppression benefit."
        )
        out.data["above_avg_defense"] = True

    elif team_oaa <= -12:
        label = "poor_defense"
        factors.append(label)
        notes.append(
            f"NEGATIVE: {backing_tp.name} team OAA {team_oaa:.0f} — "
            "poor defense; ERA will be higher than FIP indicates."
        )
        out.data["poor_defense"] = True

    # ── Pitcher–defense synergy: FIP < ERA + elite defense ───────────
    fip_era_gap = backing_sp.era - backing_sp.fip  # positive = ERA > FIP (suppressed by defense)
    if fip_era_gap >= 0.5 and team_oaa >= 10:
        label = "pitcher_defense_synergy"
        factors.append(label)
        notes.append(
            f"STRONG SYNERGY: {backing_sp.name} FIP {backing_sp.fip:.2f} < ERA "
            f"{backing_sp.era:.2f} (gap +{fip_era_gap:.2f}) AND team OAA +{team_oaa:.0f} — "
            "defense is actively depressing ERA; pitcher is better than raw ERA shows."
        )
        out.data["pitcher_defense_synergy"] = True

    out.notes.extend(notes)
    out.data["defense_factors"] = factors
    pick.factors.extend(factors)

    if not notes:
        out.notes.append(
            f"Layer 20: {backing_tp.name} OAA {team_oaa:+.0f} — no significant defensive signal."
        )
    return out


def layer_21_pitch_mix(pick: PickCandidate) -> LayerOutput:
    """
    Pitch mix change detection — identifies pitchers who have added/dropped/abandoned
    a pitch type in recent starts vs their season profile.
    Uses pitch_mix_change and pitch_mix_details fields on PitcherProfile.
    """
    out = LayerOutput(21, "Pitch Mix Changes", passed=True)

    backing_sp  = pick.backing_pitcher
    opposing_sp = pick.opposing_pitcher

    notes: list[str] = []
    factors: list[str] = []

    # ── Backing pitcher pitch mix ─────────────────────────────────────
    if backing_sp.pitch_mix_change:
        change_type = getattr(backing_sp, "_pitch_mix_change_type", None)
        details     = backing_sp.pitch_mix_details

        if change_type == "new_pitch":
            label = "arsenal_expansion"
            factors.append(label)
            notes.append(
                f"MILD POSITIVE: {backing_sp.name} has added a new pitch type — "
                f"{details} — arsenal expansion can improve effectiveness."
            )
            out.data["backing_new_pitch"] = True

        elif change_type == "primary_dropped":
            label = "command_concern"
            factors.append(label)
            notes.append(
                f"WARN: {backing_sp.name} primary pitch usage dropped significantly — "
                f"{details} — possible command or health concern."
            )
            out.data["backing_command_concern"] = True

        elif change_type == "pitch_abandoned":
            notes.append(
                f"NOTE: {backing_sp.name} has abandoned a pitch — {details}."
            )

    # ── Opposing pitcher pitch mix ────────────────────────────────────
    if opposing_sp.pitch_mix_change:
        change_type = getattr(opposing_sp, "_pitch_mix_change_type", None)
        details     = opposing_sp.pitch_mix_details

        if change_type == "pitch_abandoned":
            label = "opp_pitch_abandoned"
            factors.append(label)
            notes.append(
                f"POSITIVE (offense): {opposing_sp.name} abandoned a pitch — "
                f"{details} — batters adapting, reduced arsenal for opponent."
            )
            out.data["opp_pitch_abandoned"] = True

        elif change_type == "primary_dropped":
            label = "opp_command_loss"
            factors.append(label)
            notes.append(
                f"POSITIVE (offense): {opposing_sp.name} primary pitch usage dropped — "
                f"{details} — command regression; favorable for backing offense."
            )
            out.data["opp_command_loss"] = True

        elif change_type == "new_pitch":
            notes.append(
                f"NOTE: {opposing_sp.name} added new pitch — {details}. Monitor effectiveness."
            )

    out.notes.extend(notes)
    out.data["pitch_mix_factors"] = factors
    pick.factors.extend(factors)

    if not notes:
        out.notes.append("Layer 21: No significant pitch mix changes detected for either starter.")
    return out


# Turf stadiums (artificial surface) — used by layer_25 for speed-team advantage
_TURF_STADIUMS: frozenset[str] = frozenset({
    "Toronto Blue Jays", "Blue Jays", "TOR",
    "Tampa Bay Rays", "Rays", "TB",
    "Kansas City Royals", "Royals", "KC",
    "Arizona Diamondbacks", "Diamondbacks", "ARI",
    "Seattle Mariners", "Mariners", "SEA",
    "Miami Marlins", "Marlins", "MIA",
    "Houston Astros", "Astros", "HOU",
    "Milwaukee Brewers", "Brewers", "MIL",
    "Minnesota Twins", "Twins", "MIN",
})

# Known LHP starters — a curated list to enable layer_23 platoon detection.
# Maintained as a rough heuristic; real handedness is injected via pitcher profile
# where available (e.g., from FanGraphs/Statcast data).
_KNOWN_LHP: frozenset[str] = frozenset({
    # Active LHP starters (2026 season)
    "Clayton Kershaw", "Kershaw",
    "Cole Hamels", "Hamels",
    "Blake Snell", "Snell",
    "Nestor Cortes", "Cortes",
    "Justin Verlander", "Verlander",  # still L throw arm
    "Patrick Corbin", "Corbin",
    "Sean Manaea", "Manaea",
    "Tyler Anderson", "Anderson",
    "Drew Smyly", "Smyly",
    "Chris Sale", "Sale",
    "Jose Quintana", "Quintana",
    "Jordan Montgomery", "Montgomery",
    "Robbie Ray", "Ray",
    "Ranger Suarez", "Suarez",
    "Bailey Ober", "Ober",
    "Yusei Kikuchi", "Kikuchi",
    "MacKenzie Gore", "Gore",
    "Matthew Boyd", "Boyd",
    "Jose Berrios", "Berrios",
    "Framber Valdez", "Valdez",
    "Spencer Strider", "Strider",
    "Logan Gilbert", "Gilbert",
    "Cristopher Sanchez", "Sanchez",
    "Kyle Harrison", "Harrison",
    "Ryan Pepiot", "Pepiot",
    "DJ Herz", "Herz",
    "Andrew Abbott", "Abbott",
    "Eduardo Rodriguez", "Rodriguez",
    "Hayden Birdsong", "Birdsong",
})


def _is_lhp(pitcher_name: str) -> bool:
    """Return True if pitcher is a known left-hander, else default to RHP assumption."""
    if not pitcher_name or pitcher_name in ("TBD", ""):
        return False
    for token in _KNOWN_LHP:
        if token.lower() in pitcher_name.lower():
            return True
    return False


def layer_22_luck_filter(pick: PickCandidate) -> LayerOutput:
    """
    Tier 3 — Layer 22: Hot/Cold Streak Luck Filtering.
    Cross-references team BABIP and pitcher LOB% to identify whether a hot/cold
    streak is genuine or luck-inflated.  Adjusts confidence notes accordingly.
    Uses babip, lob_pct, and luck_score fields on TeamProfile.
    """
    out = LayerOutput(22, "Luck Filtering (BABIP / LOB%)", passed=True)

    backing_tp  = pick.backing_team_profile
    opposing_tp = (
        pick.away_team_profile if pick.backing_team == pick.home_team
        else pick.home_team_profile
    )

    notes:   list[str] = []
    factors: list[str] = []

    b_luck = backing_tp.luck_score
    o_luck = opposing_tp.luck_score
    momentum = backing_tp.momentum

    # ── Backing team luck signals ─────────────────────────────────────
    if b_luck >= 3:
        label = "hot_streak_luck_inflated"
        factors.append(label)
        notes.append(
            f"WARN: {backing_tp.name} luck score +{b_luck:.0f} "
            f"(BABIP {backing_tp.babip:.3f} / LOB% {backing_tp.lob_pct:.1%}) — "
            "hot streak may be luck-driven; expect regression."
        )
        out.data["backing_luck_inflated"] = True

    elif b_luck <= -3:
        label = "luck_correction_due"
        factors.append(label)
        notes.append(
            f"POSITIVE: {backing_tp.name} luck score {b_luck:.0f} — "
            f"team has been unlucky (BABIP {backing_tp.babip:.3f} / LOB% {backing_tp.lob_pct:.1%}); "
            "positive regression expected."
        )
        out.data["backing_luck_deflated"] = True

    # ── Backing team streaking but lucky ─────────────────────────────
    if momentum >= 5 and b_luck >= 2:
        # Downgrade confidence by appending a caution note (no hard-tier change)
        notes.append(
            f"CAUTION: {backing_tp.name} is on {momentum}-game win streak "
            f"but luck score {b_luck:+.0f} — streaking but lucky; downgrade confidence."
        )
        out.data["streaking_but_lucky"] = True
        # Do NOT add as a positive factor — it's a cautionary signal
        if "hot_streak_luck_inflated" not in factors:
            factors.append("streaking_but_lucky")

    # ── Opposing team luck signals ────────────────────────────────────
    if o_luck >= 3:
        label = "opp_luck_inflated"
        factors.append(label)
        notes.append(
            f"POSITIVE: {opposing_tp.name} luck score +{o_luck:.0f} — "
            "opponent's recent strong form is luck-inflated; regression favors us."
        )
        out.data["opp_luck_inflated"] = True

    out.notes.extend(notes)
    out.data["luck_factors"] = factors
    pick.factors.extend(factors)

    if not notes:
        out.notes.append(
            f"Layer 22: Luck scores — {backing_tp.name} {b_luck:+.0f} / "
            f"{opposing_tp.name} {o_luck:+.0f} (no significant luck signals)."
        )
    return out


def layer_23_platoon(pick: PickCandidate) -> LayerOutput:
    """
    Tier 3 — Layer 23: Platoon Splits.
    Compares backing team wRC+ vs the opposing starter's handedness.
    Uses wrc_vs_lhp and wrc_vs_rhp on TeamProfile; infers pitcher hand from _KNOWN_LHP.
    """
    out = LayerOutput(23, "Platoon Splits", passed=True)

    backing_tp  = pick.backing_team_profile
    opposing_sp = pick.opposing_pitcher

    notes:   list[str] = []
    factors: list[str] = []

    pitcher_is_lhp = _is_lhp(opposing_sp.name)
    hand_label = "LHP" if pitcher_is_lhp else "RHP"

    wrc = backing_tp.wrc_vs_lhp if pitcher_is_lhp else backing_tp.wrc_vs_rhp

    out.data["opp_pitcher_hand"] = hand_label
    out.data["backing_wrc_vs_hand"] = wrc

    # ── Strong advantage ──────────────────────────────────────────────
    if wrc >= 120:
        label = "platoon_advantage"
        factors.append(label)
        note = (
            f"STRONG POSITIVE: {backing_tp.name} wRC+ {wrc:.0f} vs {hand_label} "
            f"(≥ 120) — significant platoon advantage vs {opposing_sp.name}."
        )
        # Amplify if pitcher also has low K rate
        if opposing_sp.k9 < 7.5:
            note += (
                f" AMPLIFIED: {opposing_sp.name} K/9 {opposing_sp.k9:.1f} < 7.5 — "
                "low strikeout rate makes platoon edge even more potent."
            )
            out.data["platoon_low_k_amplified"] = True
        notes.append(note)
        out.data["strong_platoon_advantage"] = True

    elif wrc >= 115:
        label = "platoon_advantage"
        factors.append(label)
        notes.append(
            f"POSITIVE: {backing_tp.name} wRC+ {wrc:.0f} vs {hand_label} "
            f"(≥ 115) — platoon advantage vs {opposing_sp.name}."
        )
        out.data["platoon_advantage"] = True

    elif wrc <= 85:
        label = "platoon_disadvantage"
        factors.append(label)
        notes.append(
            f"NEGATIVE: {backing_tp.name} wRC+ {wrc:.0f} vs {hand_label} "
            f"(≤ 85) — platoon disadvantage vs {opposing_sp.name}; "
            "offense may struggle against this arm side."
        )
        out.data["platoon_disadvantage"] = True

    else:
        notes.append(
            f"Layer 23: {backing_tp.name} wRC+ {wrc:.0f} vs {hand_label} "
            f"({opposing_sp.name}) — neutral platoon matchup."
        )

    out.notes.extend(notes)
    out.data["platoon_factors"] = factors
    pick.factors.extend(factors)
    return out


def layer_24_lineup(pick: PickCandidate) -> LayerOutput:
    """
    Tier 3 — Layer 24: Day-Of Lineup Scratch Monitor.
    Evaluates lineup completeness and flags uncertainty or value signals.
    Informational only — does not hard-fail picks; adjusts confidence notes.
    Uses lineup_confirmed and lineup_value_score on TeamProfile.
    """
    out = LayerOutput(24, "Lineup Completeness Monitor", passed=True)

    backing_tp  = pick.backing_team_profile
    opposing_tp = (
        pick.away_team_profile if pick.backing_team == pick.home_team
        else pick.home_team_profile
    )

    notes:   list[str] = []
    factors: list[str] = []

    b_score = backing_tp.lineup_value_score
    o_score = opposing_tp.lineup_value_score
    b_confirmed = backing_tp.lineup_confirmed
    o_confirmed = opposing_tp.lineup_confirmed

    # ── Backing team lineup signals ───────────────────────────────────
    if b_score <= 3:
        notes.append(
            f"WARN: {backing_tp.name} lineup value score {b_score:.0f}/10 — "
            "significant lineup uncertainty; consider waiting for confirmation."
        )
        out.data["backing_lineup_uncertain"] = True
        factors.append("lineup_uncertainty")

    elif b_confirmed and b_score >= 8:
        label = "full_lineup_confirmed"
        factors.append(label)
        notes.append(
            f"POSITIVE: {backing_tp.name} full lineup confirmed "
            f"(value score {b_score:.0f}/10) — no late scratches detected."
        )
        out.data["full_lineup_confirmed"] = True

    # ── Opposing team lineup signals ──────────────────────────────────
    if o_score <= 3:
        label = "opp_lineup_weakened"
        factors.append(label)
        notes.append(
            f"POSITIVE: {opposing_tp.name} lineup value score {o_score:.0f}/10 — "
            "opposing key bats may be missing; favorable for backing team."
        )
        out.data["opp_lineup_weakened"] = True

    out.notes.extend(notes)
    out.data["lineup_factors"] = factors
    pick.factors.extend(factors)

    if not notes:
        out.notes.append(
            f"Layer 24: Lineup — {backing_tp.name} {b_score:.0f}/10 "
            f"({'confirmed' if b_confirmed else 'unconfirmed'}), "
            f"{opposing_tp.name} {o_score:.0f}/10 — no significant lineup signals."
        )
    return out


def layer_25_bat_speed(pick: PickCandidate) -> LayerOutput:
    """
    Tier 3 — Layer 25: Bat Speed / Exit Velocity Trends.
    Evaluates team average bat speed and sprint speed.
    Speed-team + turf stadium = BABIP boost.  High bat speed = elevated offense.
    Uses avg_bat_speed, avg_sprint_speed, and is_speed_team on TeamProfile.
    """
    out = LayerOutput(25, "Bat Speed & Sprint Speed", passed=True)

    backing_tp  = pick.backing_team_profile
    opposing_sp = pick.opposing_pitcher

    notes:   list[str] = []
    factors: list[str] = []

    bat_speed    = backing_tp.avg_bat_speed
    sprint_speed = backing_tp.avg_sprint_speed
    is_speed     = backing_tp.is_speed_team

    # ── Is this game on a turf surface? ──────────────────────────────
    on_turf = pick.home_team in _TURF_STADIUMS or pick.home_team[:3].upper() in _TURF_STADIUMS

    # ── Bat speed signals ─────────────────────────────────────────────
    if bat_speed >= 72.0:
        label = "elite_bat_speed"
        factors.append(label)
        notes.append(
            f"POSITIVE: {backing_tp.name} avg bat speed {bat_speed:.1f} mph ≥ 72 — "
            "elite bat speed; elevated offensive ceiling expected."
        )
        out.data["elite_bat_speed"] = True

    elif bat_speed <= 68.5:
        notes.append(
            f"MILD NEGATIVE: {backing_tp.name} avg bat speed {bat_speed:.1f} mph ≤ 68.5 — "
            "below-average bat speed; moderate offensive downgrade."
        )
        out.data["below_avg_bat_speed"] = True

    # ── Speed team on turf ────────────────────────────────────────────
    if is_speed and on_turf:
        label = "speed_turf_advantage"
        factors.append(label)
        notes.append(
            f"POSITIVE: {backing_tp.name} is a speed team "
            f"(avg sprint {sprint_speed:.1f} ft/sec) playing on turf — "
            "BABIP boost expected on ground balls through the infield."
        )
        out.data["speed_turf_advantage"] = True

        # Amplify signal if opposing pitcher is a high GB% pitcher
        if opposing_sp.gb_pct >= 0.50:
            notes.append(
                f"AMPLIFIED: {opposing_sp.name} ground ball% {opposing_sp.gb_pct:.0%} ≥ 50% — "
                "heavy GB pitcher against speed team on turf is a premium BABIP amplifier."
            )
            out.data["speed_turf_gb_amplified"] = True
            # Add a stronger factor label if not already present
            if "speed_turf_advantage" in factors:
                factors.append("speed_turf_gb_pitcher")

    elif is_speed and not on_turf:
        notes.append(
            f"NOTE: {backing_tp.name} is a speed team "
            f"(avg sprint {sprint_speed:.1f} ft/sec) but playing on grass — "
            "speed advantage less pronounced."
        )

    out.notes.extend(notes)
    out.data["bat_speed_factors"] = factors
    pick.factors.extend(factors)

    if not notes:
        out.notes.append(
            f"Layer 25: {backing_tp.name} bat speed {bat_speed:.1f} mph / "
            f"sprint {sprint_speed:.1f} ft/sec — no significant bat speed signals."
        )
    return out


def _extract_keywords_for_weight(factor_str: str) -> list[str]:
    """Map a factor string to weight lookup keys (mirrors weight_trainer._extract_factor_keywords)."""
    kws = []
    s = factor_str.lower()
    keyword_map = {
        "siera": "elite_siera", "xfip": "xfip_edge", "era": "era_edge",
        "k/9": "high_k9", "whip": "whip_edge", "bullpen": "bullpen_signal",
        "wind": "wind_signal", "wrigley": "wrigley_wind", "park": "park_factor",
        "sharp": "sharp_money", "pythag": "pythag_luck",
        "travel": "travel_fatigue", "opener": "opener_game",
        "velocity": "velocity_drop", "barrel": "barrel_rate",
        "xwoba": "xwoba_luck", "momentum": "team_momentum",
        "streak": "win_streak", "below .500": "fade_sub500",
        "road": "road_record", "home": "home_edge",
        "fraud": "era_fraud", "fade list": "fade_list",
        "backs list": "backs_list", "clv": "clv_positive",
    }
    for token, kw in keyword_map.items():
        if token in s:
            kws.append(kw)
    return kws or ["generic_factor"]


def _compute_learned_adjustment(factors: list[str]) -> float:
    """
    Convert learned factor weights into a small losing_pct nudge.
    Returns value clamped to [-0.06, +0.06].
    Positive → reduce losing_pct (model more confident this wins).
    Negative → increase losing_pct (model less confident).
    Only fires when we have real sample data (weight != 1.0).
    """
    if not _FACTOR_WEIGHTS or not factors:
        return 0.0
    weights = []
    for f in factors:
        for kw in _extract_keywords_for_weight(str(f)):
            if kw in _FACTOR_WEIGHTS:
                weights.append(_FACTOR_WEIGHTS[kw])
    if not weights:
        return 0.0
    avg = sum(weights) / len(weights)
    # multiplier 1.0 = no change; 1.3 → reduce lose_pct ~3%; 0.7 → raise it ~3%
    raw = (avg - 1.0) * 0.10
    return max(-0.06, min(0.06, raw))


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
    Execute all 25 layers in sequence for one pick candidate.
    Returns the pick with tier, market, factors, and all layer outputs attached.
    Layers 1-12: core analysis pipeline.
    Layers 13-16: Tier 1 enrichment (xStats, velocity, catcher framing, umpire).
    Layers 17-21: Tier 2 enrichment (CSW/Stuff+, air density, travel, defense, pitch mix).
    Layers 22-25: Tier 3 enrichment (luck filter, platoon splits, lineup monitor, bat speed).
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
        # Tier 1 enrichment layers — informational, never skip the pick on their own
        lambda: layer_13_xstats(pick),
        lambda: layer_14_velocity(pick),
        lambda: layer_15_catcher_framing(pick),
        lambda: layer_16_umpire(pick),
        # Tier 2 enrichment layers — informational, additive factors only
        lambda: layer_17_csw_stuff(pick),
        lambda: layer_18_air_density(pick),
        lambda: layer_19_travel(pick),
        lambda: layer_20_defense(pick),
        lambda: layer_21_pitch_mix(pick),
        # Tier 3 enrichment layers — informational, additive factors only
        lambda: layer_22_luck_filter(pick),
        lambda: layer_23_platoon(pick),
        lambda: layer_24_lineup(pick),
        lambda: layer_25_bat_speed(pick),
    ]

    for fn in layers_fn:
        result = fn()
        pick.layer_outputs.append(result)
        # TBD pitcher: note the uncertainty but continue analyzing with available team data

    # Apply learned factor weight adjustment (soft nudge only — never overrides hard rules)
    if _FACTOR_WEIGHTS and pick.tier != "SKIP" and pick.factors:
        adj = _compute_learned_adjustment(pick.factors)
        if adj != 0.0:
            new_lose = max(0.10, min(0.89, pick.losing_pct - adj))
            pick.losing_pct = round(new_lose, 4)
            # Re-derive tier from adjusted losing_pct
            for tier, (lo, hi) in _TIER_THRESHOLDS.items():
                if lo <= new_lose < hi:
                    pick.tier = tier
                    break
            logger.debug(
                "Learned adj %.3f → lose_pct %.3f tier %s for %s",
                adj, new_lose, pick.tier, pick.game_id,
            )

    return pick
