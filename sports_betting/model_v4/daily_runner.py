"""
Daily execution runner for model v4.0.
Implements the DAILY_EXECUTION_TEMPLATE from the system prompt:

  Phase 0 — Debrief prior results
  Phase 1 — Data pull (schedule, pitchers, weather, odds)
  Layers 1-12 — Full analysis per game
  Output — Formatted card + parlays + NRFI section

Bridges the v4 model to the existing data collection infrastructure.
"""
from __future__ import annotations
import logging
from datetime import datetime

from ..collectors.mlb_collector import (
    get_todays_games, get_team_standings, get_team_id_map,
    compute_team_streak, get_pitcher_stats, get_ballpark_factors,
)
from ..collectors.odds_collector import get_live_odds, parse_and_store_odds
from ..collectors.weather_collector import get_game_weather
from ..collectors.sharp_money import detect_wiseguy_moves
from ..collectors.fangraphs_collector import (
    get_pitcher_stats_fangraphs, get_statcast_pitcher_metrics, enrich_pitcher_profile
)
from ..database import init_db, upsert_game
from ..models.trainer import update_bet_results, compute_roi_summary
from ..analysis.signal_tracker import update_signal_performance

from .pitcher_lists import is_on_fade_list, is_on_backs_list, is_era_fraud
from .hard_rules import run_all_hard_rules_for_card, check_seven_day_cap
from .layer_engine import (
    PickCandidate, PitcherProfile, TeamProfile, WeatherProfile, run_all_layers
)
from .parlay_builder import picks_to_legs, build_full_parlay_card
from .nrfi_yrfi import NrfiProfile, rank_games_for_nrfi_parlay, build_nrfi_parlay
from .pick_card import render_pick_card
from ..collectors.hr_props_collector import run_hr_parlay_analysis

logger = logging.getLogger(__name__)


def run_daily_model(date_str: str | None = None, verbose: bool = True) -> dict:
    """
    Full daily model execution. Returns all results as a dict.
    date_str: "YYYY-MM-DD" or None for today.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    logger.info("=== Model v4.0 daily run: %s ===", date_str)
    init_db()

    # ------------------------------------------------------------------ #
    # PHASE 0 — Debrief prior results                                     #
    # ------------------------------------------------------------------ #
    logger.info("Phase 0: Grading prior bets...")
    from ..models.prediction_model import MLBPredictor
    predictor = MLBPredictor()
    update_bet_results(predictor)
    roi = compute_roi_summary()
    if roi.get("total_bets"):
        logger.info(
            "Season: %d-%d | ROI: %.2f%% | Hit Rate: %.1f%%",
            roi["wins"], roi["losses"], roi["roi"], roi["hit_rate"] * 100
        )

    # ------------------------------------------------------------------ #
    # PHASE 1 — Data pull                                                 #
    # ------------------------------------------------------------------ #
    logger.info("Phase 1: Pulling schedule, odds, weather, FanGraphs, Statcast...")
    games = get_todays_games()
    for g in games:
        upsert_game(g)

    raw_odds = get_live_odds()
    parsed_games = parse_and_store_odds(raw_odds)

    team_id_map = get_team_id_map()
    standings = get_team_standings()
    standings_by_name = {s["team_name"]: s for s in standings}

    # Upgrade 1 — FanGraphs live pitcher stats
    logger.info("Fetching FanGraphs pitcher stats...")
    fg_stats = get_pitcher_stats_fangraphs()

    # Upgrade 2 — Baseball Savant Statcast metrics
    logger.info("Fetching Statcast pitcher metrics...")
    sc_stats = get_statcast_pitcher_metrics()

    # ------------------------------------------------------------------ #
    # LAYERS 1-12 — Full analysis per game                               #
    # ------------------------------------------------------------------ #
    logger.info("Running 12-layer analysis on %d games...", len(games))

    pick_candidates: list[PickCandidate] = []
    skipped_games: list[dict] = []
    nrfi_profiles: list[NrfiProfile] = []
    prices_map: dict[str, int] = {}

    # Build odds lookup
    odds_by_teams: dict[tuple, dict] = {
        (pg.get("home_team"), pg.get("away_team")): pg
        for pg in parsed_games
    }

    for game in games:
        home = game["home_team"]
        away = game["away_team"]
        game_id = game["game_id"]

        # Weather
        weather_raw = get_game_weather(home, game_id)
        weather = WeatherProfile(
            temperature=weather_raw.get("temperature", 72),
            wind_speed=weather_raw.get("wind_speed", 5),
            wind_direction=_classify_wind_direction(weather_raw),
            rain_pct=weather_raw.get("precipitation_chance", 0),
            is_dome=weather_raw.get("roof_closed", False),
        )

        # Build pitcher profiles
        home_sp_name = game.get("home_probable_pitcher", "TBD")
        away_sp_name = game.get("away_probable_pitcher", "TBD")
        home_pitcher = _build_pitcher_profile(home_sp_name, home, team_id_map)
        away_pitcher = _build_pitcher_profile(away_sp_name, away, team_id_map)

        # Upgrade 1+2 — enrich with real FanGraphs + Statcast data
        enrich_pitcher_profile(home_pitcher, fg_stats, sc_stats)
        enrich_pitcher_profile(away_pitcher, fg_stats, sc_stats)

        # Build team profiles
        home_stand = standings_by_name.get(home, {})
        away_stand = standings_by_name.get(away, {})
        home_team_id = team_id_map.get(home)
        away_team_id = team_id_map.get(away)

        home_streak_raw = compute_team_streak(home_team_id, home) if home_team_id else {}
        away_streak_raw = compute_team_streak(away_team_id, away) if away_team_id else {}

        home_profile = _build_team_profile(home, home_stand, home_streak_raw)
        away_profile = _build_team_profile(away, away_stand, away_streak_raw)

        # Odds from parsed games
        odds_entry = odds_by_teams.get((home, away), {})
        dk_h2h = odds_entry.get("odds_by_book", {}).get("draftkings", {}).get("h2h", {})
        home_price = dk_h2h.get("home_price", -110)
        away_price = dk_h2h.get("away_price", +100)

        # Determine which side to analyze
        # Priority: backing team is the better pitcher's team
        backing_team = _select_backing_team(home, away, home_pitcher, away_pitcher, home_profile, away_profile)
        backing_price = home_price if backing_team == home else away_price
        prices_map[game_id] = backing_price

        # NRFI profile
        nrfi_profiles.append(NrfiProfile(
            game_id=game_id,
            home_team=home,
            away_team=away,
            home_pitcher_name=home_sp_name,
            away_pitcher_name=away_sp_name,
            home_pitcher_era=home_pitcher.era,
            away_pitcher_era=away_pitcher.era,
            home_pitcher_whip=home_pitcher.whip,
            away_pitcher_whip=away_pitcher.whip,
            temperature=weather.temperature,
            wind_speed=weather.wind_speed,
            wind_direction=weather.wind_direction,
            rain_pct=weather.rain_pct,
        ))

        # Hard rules pre-check
        cap = check_seven_day_cap(backing_team)
        if cap["is_capped"]:
            skipped_games.append({
                "game": f"{away} @ {home}",
                "reason": f"7-day cap reached for {backing_team} (3/3 ML picks used)",
            })
            continue

        # ERA fraud checks
        home_fade, _ = is_on_fade_list(home_sp_name)
        away_fade, _ = is_on_fade_list(away_sp_name)

        if home_fade:
            logger.info("Fade list: %s — reversing backing to away.", home_sp_name)
            backing_team = away
            backing_price = away_price

        if away_fade:
            logger.info("Fade list: %s — confirming home backing.", away_sp_name)
            backing_team = home
            backing_price = home_price

        # Estimate true probability from model
        true_prob = _estimate_true_probability(
            backing_team, home, away,
            home_pitcher, away_pitcher,
            home_profile, away_profile,
            weather,
        )

        # Losing scenario
        losing_pct = 1 - true_prob
        losing_scenario = _write_losing_scenario(
            backing_team, home, away,
            home_pitcher, away_pitcher, weather
        )

        # Factor list
        confirmed_factors = _build_factor_list(
            backing_team, home, away,
            home_pitcher, away_pitcher,
            home_profile, away_profile,
            weather,
        )

        # Create and run the pick
        pick = PickCandidate(
            game_id=game_id,
            home_team=home,
            away_team=away,
            home_pitcher=home_pitcher,
            away_pitcher=away_pitcher,
            home_team_profile=home_profile,
            away_team_profile=away_profile,
            weather=weather,
            backing_team=backing_team,
            proposed_market="full_game_ml",
        )

        # Note TBD games but still analyze with available data
        if home_sp_name in ("TBD", "") or away_sp_name in ("TBD", ""):
            pick.skip_reason = None  # Clear any skip — allow layers to run with warning

        pick = run_all_layers(
            pick=pick,
            book_price=backing_price,
            true_probability=true_prob,
            losing_scenario=losing_scenario,
            losing_pct=losing_pct,
            confirmed_factors=confirmed_factors,
        )

        if pick.tier == "SKIP":
            skipped_games.append({
                "game": f"{away} @ {home}",
                "reason": pick.skip_reason or "Did not survive 12-layer filter",
            })
        else:
            pick_candidates.append(pick)

    # Hard rules pass over the full card (max-1-per-game, etc.)
    pick_dicts = [{"game_id": p.game_id, "home_team": p.home_team,
                   "away_team": p.away_team, "backing_team": p.backing_team,
                   "market": p.proposed_market,
                   "backing_team_win_pct": p.win_pct,
                   "rain_pct": p.weather.rain_pct,
                   "pitcher_is_debut": p.backing_pitcher.is_debut,
                   "series_game_number": p.backing_team_profile.series_wins + 1,
                   "team_series_wins": p.backing_team_profile.series_wins,
                   } for p in pick_candidates]
    pick_dicts = run_all_hard_rules_for_card(pick_dicts)

    # Attach hard rules results back
    for p, pd in zip(pick_candidates, pick_dicts):
        p.hard_rules_result = pd.get("hard_rules_result")
        if p.hard_rules_result and not p.hard_rules_result.standalone_eligible:
            p.tier = "SKIP"
            p.skip_reason = "; ".join(
                v.reason for v in p.hard_rules_result.violations
            )

    # NRFI ranking
    nrfi_ranked = rank_games_for_nrfi_parlay(nrfi_profiles)

    # Build parlays
    legs = picks_to_legs(pick_candidates, prices_map)
    parlays = build_full_parlay_card(legs)

    # NRFI 5-leg parlay
    nrfi_parlay = build_nrfi_parlay(nrfi_ranked, legs=5)
    if nrfi_parlay:
        logger.info("NRFI 5-leg parlay built: %s", nrfi_parlay.get("american_odds"))

    # HR parlay analysis
    weather_by_game = {
        f"{g.get('away_team','?')} @ {g.get('home_team','?')}": g.get("weather_raw", {})
        for g in parsed_games
    }
    hr_results = run_hr_parlay_analysis(parsed_games, weather_by_game)

    # Sharp money
    sharp_plays = detect_wiseguy_moves(parsed_games)

    # Upgrade 3 — update signal weights from completed game feedback
    update_signal_performance()

    # Render the card
    if verbose:
        render_pick_card(
            picks=pick_candidates,
            parlays=parlays,
            nrfi_ranked=nrfi_ranked,
            skipped_games=skipped_games,
            date_str=date_str,
            roi_summary=roi if roi.get("total_bets") else None,
        )

    return {
        "date": date_str,
        "picks": pick_candidates,
        "parlays": parlays,
        "nrfi_parlay": nrfi_parlay,
        "nrfi_ranked": nrfi_ranked,
        "skipped": skipped_games,
        "sharp_plays": sharp_plays,
        "hr_results": hr_results,
        "roi": roi,
    }


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _classify_wind_direction(weather_raw: dict) -> str:
    deg = weather_raw.get("wind_direction", 0)
    if deg is None:
        return "calm"
    # Simplified: 0/360=N, 90=E, 180=S, 270=W
    # Most MLB parks face roughly north or northwest
    if 45 <= deg <= 135:
        return "out"
    elif 225 <= deg <= 315:
        return "in"
    elif 135 < deg < 225:
        return "R-L"
    else:
        return "L-R"


def _build_pitcher_profile(name: str, team: str, team_id_map: dict) -> PitcherProfile:
    if not name or name == "TBD":
        return PitcherProfile(name="TBD", team=team, confirmed_sources=0)

    profile = PitcherProfile(name=name, team=team, confirmed_sources=1)

    # Check permanent lists
    on_backs, backs_data = is_on_backs_list(name)
    if on_backs:
        profile.era = backs_data.get("era", 3.50)
        profile.siera = backs_data.get("siera", profile.era + 0.30)
        profile.xfip = backs_data.get("xfip", profile.era + 0.30)
        profile.k9 = backs_data.get("k9", 9.0)
        profile.whip = backs_data.get("whip", 1.10)
        profile.confirmed_sources = 2
        return profile

    on_fade, fade_data = is_on_fade_list(name)
    if on_fade:
        profile.era = fade_data.get("era", 5.50)
        profile.siera = fade_data.get("siera", profile.era + 0.50)
        profile.xera = fade_data.get("xera", profile.era + 1.50)
        profile.fip = fade_data.get("fip", profile.era + 1.00)
        profile.bb9 = fade_data.get("bb9", 4.0)
        profile.whip = fade_data.get("whip", 1.55)
        profile.confirmed_sources = 2
        return profile

    return profile


def _build_team_profile(name: str, standing: dict, streak_raw: dict) -> TeamProfile:
    w = standing.get("wins", 0) or 0
    l = standing.get("losses", 0) or 0
    streak_count = streak_raw.get("streak_count", 0) or 0
    streak_type = streak_raw.get("streak_type", "")
    momentum = streak_count if streak_type == "W" else -streak_count

    last_10 = streak_raw.get("last_10", "5-5")
    try:
        wins10 = int(str(last_10).split("-")[0])
    except (ValueError, AttributeError):
        wins10 = 5

    return TeamProfile(
        name=name,
        wins=w,
        losses=l,
        run_diff=standing.get("run_diff", 0) or 0,
        runs_per_game_last_5=4.0,
        bullpen_era=4.00,
        bullpen_era_14d=4.00,
        road_wins=0,
        road_losses=0,
        wrc_plus_14d=100,
        series_wins=0,
        momentum=momentum,
    )


def _select_backing_team(
    home: str, away: str,
    home_p: PitcherProfile, away_p: PitcherProfile,
    home_t: TeamProfile, away_t: TeamProfile,
) -> str:
    """Pick the side with the larger combined pitcher + team quality edge."""
    home_score = 0.0
    away_score = 0.0

    home_cps = (home_p.siera + home_p.era) / 2
    away_cps = (away_p.siera + away_p.era) / 2

    home_score += (away_cps - home_cps) * 0.5  # Lower ERA = better
    away_score += (home_cps - away_cps) * 0.5

    home_total = max(1, home_t.wins + home_t.losses)
    away_total = max(1, away_t.wins + away_t.losses)
    home_score += (home_t.wins / home_total) * 0.3
    away_score += (away_t.wins / away_total) * 0.3

    home_score += 0.03  # Home field

    return home if home_score >= away_score else away


def _estimate_true_probability(
    backing: str, home: str, away: str,
    home_p: PitcherProfile, away_p: PitcherProfile,
    home_t: TeamProfile, away_t: TeamProfile,
    weather: WeatherProfile,
) -> float:
    import math
    score = 0.0

    # Pitcher edge
    home_cps = (home_p.siera + home_p.era) / 2
    away_cps = (away_p.siera + away_p.era) / 2
    if backing == home:
        score += (away_cps - home_cps) * 0.18
    else:
        score += (home_cps - away_cps) * 0.18

    # Win% edge
    home_total = max(1, home_t.wins + home_t.losses)
    away_total = max(1, away_t.wins + away_t.losses)
    home_wpct = home_t.wins / home_total
    away_wpct = away_t.wins / away_total
    if backing == home:
        score += (home_wpct - away_wpct) * 0.40
        score += 0.04  # HFA
    else:
        score += (away_wpct - home_wpct) * 0.40

    # Streak
    if backing == home:
        score += home_t.momentum * 0.01
    else:
        score += away_t.momentum * 0.01

    prob = 1 / (1 + math.exp(-score * 3))
    return round(max(0.40, min(0.75, prob)), 4)


def _write_losing_scenario(
    backing: str, home: str, away: str,
    home_p: PitcherProfile, away_p: PitcherProfile,
    weather: WeatherProfile,
) -> str:
    opposing = away_p if backing == home else home_p
    return (
        f"{opposing.name} outperforms metrics and limits {backing} to ≤2 runs, "
        f"OR {backing} offense goes cold. "
        + (f"Rain ({weather.rain_pct:.0f}%) causes delays. " if weather.rain_pct >= 20 else "")
    )


def _build_factor_list(
    backing: str, home: str, away: str,
    home_p: PitcherProfile, away_p: PitcherProfile,
    home_t: TeamProfile, away_t: TeamProfile,
    weather: WeatherProfile,
) -> list[str]:
    factors = []
    sp = home_p if backing == home else away_p
    opp_sp = away_p if backing == home else home_p
    team = home_t if backing == home else away_t
    opp_team = away_t if backing == home else home_t
    total = max(1, team.wins + team.losses)
    opp_total = max(1, opp_team.wins + opp_team.losses)
    wpct = team.wins / total
    opp_wpct = opp_team.wins / opp_total

    # --- Pitcher quality (real data when available, else default 4.50 won't trigger) ---
    on_backs, backs_data = is_on_backs_list(sp.name)
    if on_backs:
        factors.append(f"{sp.name} on permanent backs list ({backs_data.get('note','')})")

    if sp.siera < 3.80:
        factors.append(f"SP elite SIERA: {sp.siera:.2f}")
    if sp.era < 3.50:
        factors.append(f"SP ERA {sp.era:.2f} (strong)")
    if sp.k9 >= 9.0:
        factors.append(f"SP K/9 {sp.k9:.1f} (above avg strikeouts)")
    if sp.whip < 1.20:
        factors.append(f"SP WHIP {sp.whip:.2f} (good command)")

    fraud = is_era_fraud(opp_sp.era, opp_sp.xera, opp_sp.fip, opp_sp.siera)
    if fraud["is_fraud"]:
        factors.append(f"Opposing {opp_sp.name} ERA fraud confirmed ({' | '.join(fraud['confirmations'][:2])})")

    on_fade, fade_data = is_on_fade_list(opp_sp.name)
    if on_fade:
        factors.append(f"Opposing {opp_sp.name} on permanent fade list ({fade_data.get('note','')})")

    # Opposing pitcher ERA vulnerability
    if opp_sp.era >= 4.50 and opp_sp.name not in ("TBD", ""):
        factors.append(f"Opposing SP {opp_sp.name} ERA {opp_sp.era:.2f} (exploitable)")
    elif opp_sp.name in ("TBD", ""):
        factors.append("Opposing SP unconfirmed (TBD) — scheduling uncertainty favors prepared side")

    # --- Team record & form ---
    if wpct >= 0.520:
        factors.append(f"{backing} winning record ({team.wins}-{team.losses}, {wpct:.3f})")
    elif wpct >= 0.480:
        factors.append(f"{backing} near-.500 record ({team.wins}-{team.losses}) — competitive")

    if opp_wpct < 0.460:
        factors.append(f"Opponent {away if backing == home else home} below .500 ({opp_team.wins}-{opp_team.losses})")

    if team.run_diff > 0:
        factors.append(f"Positive run differential ({team.run_diff:+d})")
    elif team.run_diff > -10:
        factors.append(f"Near-even run differential ({team.run_diff:+d}) — competitive offense")

    if team.momentum >= 3:
        factors.append(f"Hot streak: {team.momentum}-game winning streak")
    elif team.momentum >= 2:
        factors.append(f"Winning momentum: {team.momentum} straight wins")

    if opp_team.momentum <= -3:
        factors.append(f"Opponent on {abs(opp_team.momentum)}-game losing streak")

    # --- Home/Away ---
    if backing == home:
        factors.append("Home field advantage (+3-4% win probability boost)")
    else:
        if away_t.wins > away_t.losses:
            factors.append(f"{away} road record supports away backing")

    # --- Weather ---
    if weather.is_dome:
        factors.append("Dome game — no weather variance, consistent conditions")
    elif weather.temperature >= 80:
        factors.append(f"Warm weather ({weather.temperature:.0f}°F) — ball carries")
    elif weather.temperature < 55:
        factors.append(f"Cold ({weather.temperature:.0f}°F) — pitcher-friendly conditions")

    if weather.wind_speed >= 10 and weather.wind_direction == "in":
        factors.append(f"Wind {weather.wind_speed:.0f}mph in — suppresses scoring")

    return factors
