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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
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
    PickCandidate, PitcherProfile, TeamProfile, WeatherProfile, run_all_layers,
    update_tier_thresholds, update_factor_weights,
)
from ..models.outcome_grader import grade_all_pending
from ..models.weight_trainer import run_full_retrain, load_learned_weights
from .parlay_builder import picks_to_legs, build_full_parlay_card
from .nrfi_yrfi import NrfiProfile, rank_games_for_nrfi_parlay, build_nrfi_parlay
from .pick_card import render_pick_card
from ..collectors.hr_props_collector import run_hr_parlay_analysis
from ..collectors.fangraphs_collector import (
    get_pitcher_velocity_trends, check_velocity_trend,
    get_pitcher_hr_vulnerability, get_team_xwoba_luck,
)
from ..collectors.xstats_collector import (
    get_pitcher_xstats, get_team_xwoba,
    get_pitcher_csw, get_stuff_plus,
)
from ..collectors.velocity_tracker import get_velocity_data
from ..collectors.catcher_framing_collector import get_framing_by_team
from ..collectors.umpire_collector import get_todays_umpires
from ..collectors.bullpen_fatigue_collector import get_bullpen_fatigue
from ..collectors.air_density_collector import get_air_density_for_game
from ..collectors.travel_fatigue_collector import get_travel_fatigue
from ..collectors.defensive_metrics_collector import get_team_oaa
from ..collectors.pitch_mix_collector import get_pitch_mix_changes
from ..collectors.luck_metrics_collector import get_luck_metrics
from ..collectors.platoon_splits_collector import get_platoon_splits
from ..collectors.lineup_monitor import get_lineups
from ..collectors.bat_speed_collector import get_bat_speed_metrics
from ..signals.aggregator import run_all_signals, build_signal_factors

logger = logging.getLogger(__name__)


def _timed(fn, *args, timeout: int = 8, default=None, **kwargs):
    """Call fn(*args, **kwargs) with a hard wall-clock timeout. Returns default on timeout/error."""
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn, *args, **kwargs)
            return future.result(timeout=timeout)
    except FuturesTimeout:
        logger.warning("_timed: %s timed out after %ds — using default", getattr(fn, "__name__", fn), timeout)
        return default if default is not None else {}
    except Exception as exc:
        logger.warning("_timed: %s failed: %s — using default", getattr(fn, "__name__", fn), exc)
        return default if default is not None else {}


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

    # Grade pending bets with real MLB scores
    logger.info("Phase 0: Grading pending bets from MLB API...")
    try:
        graded = grade_all_pending(days_back=3)
        if graded:
            logger.info("Graded %d bets from real game results", graded)
    except Exception as e:
        logger.warning("Outcome grader error: %s", e)

    # Self-improvement: retrain weights from graded picks
    try:
        retrain_result = run_full_retrain()
        logger.info("Weight retrain: %s", retrain_result.get("status"))
    except Exception as e:
        logger.warning("Weight retrain error: %s", e)

    # Load learned weights (dynamic thresholds + factor multipliers)
    learned_weights = load_learned_weights()
    logger.info("Loaded learned weights: %d factors, %d markets, sample_size=%d",
                len(learned_weights.get("factors", {})),
                len(learned_weights.get("markets", {})),
                learned_weights.get("sample_size", 0))

    # Apply dynamic tier thresholds to layer engine
    update_tier_thresholds(learned_weights.get("thresholds", {}))
    # Apply learned factor weights to the scoring pipeline
    update_factor_weights(learned_weights.get("factors", {}))

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

    # All enrichment data fetched in parallel — keeps boot time under 60s.
    # Only calls that hit blocked external sites (Savant/Covers/InsideThePen) use _timed().
    # MLB Stats API calls run directly — they work from cloud and need up to 20s for 500 pitchers.
    logger.info("Fetching all enrichment data in parallel...")
    with ThreadPoolExecutor(max_workers=20) as _pool:
        _f_fg       = _pool.submit(get_pitcher_stats_fangraphs)       # MLB API — no timeout needed
        _f_sc       = _pool.submit(get_statcast_pitcher_metrics)       # MLB API — no timeout needed
        _f_vel_fg   = _pool.submit(_timed, get_pitcher_velocity_trends, timeout=8, default={})  # Savant — blocked
        _f_hr       = _pool.submit(get_pitcher_hr_vulnerability)       # MLB API — no timeout needed
        _f_xwoba_lk = _pool.submit(get_team_xwoba_luck)               # MLB API — no timeout needed
        _f_px       = _pool.submit(get_pitcher_xstats)
        _f_tx       = _pool.submit(get_team_xwoba)
        _f_vel      = _pool.submit(get_velocity_data)
        _f_frm      = _pool.submit(get_framing_by_team)
        _f_ump      = _pool.submit(_timed, get_todays_umpires, date_str,  timeout=10, default={})
        _f_bull     = _pool.submit(_timed, get_bullpen_fatigue,           timeout=10, default={})
        _f_csw      = _pool.submit(get_pitcher_csw)
        _f_stuff    = _pool.submit(get_stuff_plus)
        _f_travel   = _pool.submit(get_travel_fatigue, date_str)
        _f_defense  = _pool.submit(get_team_oaa)
        _f_pitch    = _pool.submit(get_pitch_mix_changes)
        _f_luck     = _pool.submit(get_luck_metrics)
        _f_platoon  = _pool.submit(get_platoon_splits)
        _f_lineup   = _pool.submit(_timed, get_lineups, date_str, timeout=15, default={})
        _f_bat      = _pool.submit(get_bat_speed_metrics)

        fg_stats           = _f_fg.result()
        sc_stats           = _f_sc.result()
        velocity_data      = _f_vel_fg.result()
        hr_vuln_data       = _f_hr.result()
        xwoba_luck         = _f_xwoba_lk.result()
        t1_pitcher_xstats  = _f_px.result()
        t1_team_xwoba      = _f_tx.result()
        t1_velocity        = _f_vel.result()
        t1_framing_by_team = _f_frm.result()
        t1_umpires         = _f_ump.result()
        t1_bullpen_fatigue = _f_bull.result()
        t2_csw_data        = _f_csw.result()
        t2_stuff_data      = _f_stuff.result()
        t2_travel_data     = _f_travel.result()
        t2_defense_data    = _f_defense.result()
        t2_pitch_mix_data  = _f_pitch.result()
        t3_luck_data       = _f_luck.result()
        t3_platoon_data    = _f_platoon.result()
        t3_lineup_data     = _f_lineup.result()
        t3_bat_speed_data  = _f_bat.result()

    all_signals = run_all_signals(games, standings_by_name, team_id_map)
    logger.info("All enrichment data loaded.")

    # Pre-fetch team streaks and weather in parallel (avoids 30+ sequential MLB API calls)
    all_team_ids: set[tuple] = {
        (game["home_team"], team_id_map.get(game["home_team"]))
        for game in games
    } | {
        (game["away_team"], team_id_map.get(game["away_team"]))
        for game in games
    }
    streaks_cache: dict[str, dict] = {}
    weather_cache: dict[str, dict] = {}

    def _fetch_streak(name: str, team_id: int | None) -> tuple[str, dict]:
        if team_id:
            return name, compute_team_streak(team_id, name)
        return name, {}

    def _fetch_weather(home: str, game_id: str) -> tuple[str, dict]:
        return game_id, get_game_weather(home, game_id)

    with ThreadPoolExecutor(max_workers=20) as _prep_pool:
        _sf = {_prep_pool.submit(_fetch_streak, n, tid): n for n, tid in all_team_ids if n}
        _wf = {
            _prep_pool.submit(_fetch_weather, g["home_team"], g["game_id"]): g["game_id"]
            for g in games
        }
        for fut in _sf:
            name, streak = fut.result()
            streaks_cache[name] = streak
        for fut in _wf:
            gid, w = fut.result()
            weather_cache[gid] = w

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

        # Weather (pre-fetched)
        weather_raw = weather_cache.get(game_id, {})
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
        home_streak_raw = streaks_cache.get(home, {})
        away_streak_raw = streaks_cache.get(away, {})

        home_profile = _build_team_profile(home, home_stand, home_streak_raw)
        away_profile = _build_team_profile(away, away_stand, away_streak_raw)

        # Odds from parsed games
        odds_entry = odds_by_teams.get((home, away), {})
        dk_h2h = odds_entry.get("odds_by_book", {}).get("draftkings", {}).get("h2h", {})
        home_price = dk_h2h.get("home_price", -120)
        away_price = dk_h2h.get("away_price", +105)

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

        # Factor list (base + all 13 intelligence signals)
        game_key = f"{away} @ {home}"
        confirmed_factors = _build_factor_list(
            backing_team, home, away,
            home_pitcher, away_pitcher,
            home_profile, away_profile,
            weather,
        )
        # Append signal-derived factors
        confirmed_factors += build_signal_factors(
            game_key, backing_team, home, away, all_signals
        )
        # Signal 7: velocity trend
        sp = home_pitcher if backing_team == home else away_pitcher
        opp_sp = away_pitcher if backing_team == home else home_pitcher
        vt = check_velocity_trend(opp_sp.name, velocity_data)
        if vt.get("flag"):
            confirmed_factors.append(f"Opposing {opp_sp.name} velocity down {vt['drop']:.1f}mph — arm fatigue signal")
        # Signal 13: HR/FB vulnerability of opposing pitcher
        opp_last = opp_sp.name.split()[-1].lower() if opp_sp.name not in ("TBD","") else ""
        opp_hr_vuln = hr_vuln_data.get(opp_last, {})
        if opp_hr_vuln.get("hr_fb_rate", 0) >= 0.14:
            confirmed_factors.append(f"Opposing {opp_sp.name} HR/FB {opp_hr_vuln['hr_fb_rate']:.0%} — homer-prone")
        # Signal 3: xwOBA luck (away team name abbrev won't match well so skip for now)
        opp_team = away if backing_team == home else home
        opp_xwoba = xwoba_luck.get(opp_team, {})
        if opp_xwoba.get("label") == "lucky":
            confirmed_factors.append(f"Opponent {opp_team} hitting above xwOBA ({opp_xwoba.get('gap',0):+.3f}) — regression due")

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

        # ── Tier 1: inject enrichment data into pick/profiles ────────
        _inject_tier1_data(
            pick=pick,
            t1_pitcher_xstats=t1_pitcher_xstats,
            t1_team_xwoba=t1_team_xwoba,
            t1_velocity=t1_velocity,
            t1_framing_by_team=t1_framing_by_team,
            t1_umpires=t1_umpires,
            t1_bullpen_fatigue=t1_bullpen_fatigue,
        )

        # ── Tier 2: inject enrichment data into pick/profiles ────────
        _inject_tier2_data(
            pick=pick,
            home=home,
            game_id=game_id,
            t2_csw_data=t2_csw_data,
            t2_stuff_data=t2_stuff_data,
            t2_travel_data=t2_travel_data,
            t2_defense_data=t2_defense_data,
            t2_pitch_mix_data=t2_pitch_mix_data,
        )

        # ── Tier 3: inject enrichment data into pick/profiles ────────
        _inject_tier3_data(
            pick=pick,
            t3_luck_data=t3_luck_data,
            t3_platoon_data=t3_platoon_data,
            t3_lineup_data=t3_lineup_data,
            t3_bat_speed_data=t3_bat_speed_data,
        )

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
        "all_signals": all_signals,
        "xwoba_luck": xwoba_luck,
        "roi": roi,
        "learned_weights": learned_weights,
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
    if sp.xfip < 3.50:
        factors.append(f"SP xFIP {sp.xfip:.2f} (elite, true skill indicator)")
    if sp.hard_hit_rate < 0.33:
        factors.append(f"SP hard hit% {sp.hard_hit_rate:.0%} (elite soft contact)")
    if sp.gb_pct >= 0.52:
        factors.append(f"SP ground ball% {sp.gb_pct:.0%} (limits HR damage)")
    if sp.ip_per_start >= 6.2:
        factors.append(f"SP averaging {sp.ip_per_start:.1f} IP/start (goes deep, limits bullpen use)")
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
    if opp_sp.xfip >= 4.80 and opp_sp.name not in ("TBD", ""):
        factors.append(f"Opposing SP {opp_sp.name} xFIP {opp_sp.xfip:.2f} (HR-prone, skills worse than ERA)")
    if opp_sp.hard_hit_rate >= 0.42 and opp_sp.name not in ("TBD", ""):
        factors.append(f"Opposing SP {opp_sp.name} hard hit% {opp_sp.hard_hit_rate:.0%} (hitters squaring up)")
    if opp_sp.fly_ball_pct >= 0.42 and opp_sp.name not in ("TBD", ""):
        factors.append(f"Opposing SP {opp_sp.name} fly ball% {opp_sp.fly_ball_pct:.0%} (HR risk in hitter park)")
    if opp_sp.name in ("TBD", ""):
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


# ------------------------------------------------------------------ #
#  Tier 1 enrichment injection                                        #
# ------------------------------------------------------------------ #

def _inject_tier1_data(
    pick,
    t1_pitcher_xstats: dict,
    t1_team_xwoba: dict,
    t1_velocity: dict,
    t1_framing_by_team: dict,
    t1_umpires: dict,
    t1_bullpen_fatigue: dict,
) -> None:
    """
    Mutates PitcherProfile fields and TeamProfile fields on the pick with
    Tier 1 enrichment data fetched before the per-game loop.
    Also attaches umpire data as a custom attribute on the pick object.
    """
    # ── 1. Pitcher xStats (xERA into PitcherProfile) ─────────────────
    for pitcher in (pick.home_pitcher, pick.away_pitcher):
        if pitcher.name in ("TBD", ""):
            continue
        # Try full name first, then last name only
        xdata = t1_pitcher_xstats.get(pitcher.name)
        if not xdata:
            last = pitcher.name.split()[-1]
            xdata = t1_pitcher_xstats.get(last)
        if xdata:
            # Only overwrite if we got real data (non-default)
            if xdata.get("xera", 4.50) != 4.50 or xdata.get("era", 4.50) != 4.50:
                pitcher.xera  = xdata.get("xera",  pitcher.xera)
                pitcher.xfip  = xdata.get("xfip",  pitcher.xfip)
                pitcher.siera = xdata.get("siera", pitcher.siera)

    # ── 2. Team xwOBA gap (stored as custom attr for layer_13) ───────
    for team_profile in (pick.home_team_profile, pick.away_team_profile):
        team_abbr = team_profile.name[:3].upper()
        xwoba_entry = t1_team_xwoba.get(team_abbr, {})
        xwoba = xwoba_entry.get("xwoba", 0.0)
        woba  = xwoba_entry.get("woba",  0.0)
        if xwoba > 0 and woba > 0:
            gap = round(xwoba - woba, 3)  # positive = hitting below expected (rebound due)
            team_profile._xwoba_gap = gap  # type: ignore[attr-defined]
        else:
            team_profile._xwoba_gap = None  # type: ignore[attr-defined]

    # ── 3. Velocity & spin data into PitcherProfile ───────────────────
    for pitcher in (pick.home_pitcher, pick.away_pitcher):
        if pitcher.name in ("TBD", ""):
            continue
        vdata = t1_velocity.get(pitcher.name)
        if not vdata:
            last = pitcher.name.split()[-1]
            vdata = t1_velocity.get(last)
        if vdata:
            pitcher.velocity_season = vdata.get("season_velo",   pitcher.velocity_season)
            pitcher.velocity_7d     = vdata.get("recent_velo",   pitcher.velocity_7d)
            pitcher.spin_rate_season = vdata.get("season_spin",  pitcher.spin_rate_season)
            pitcher.spin_rate_7d    = vdata.get("recent_spin",   pitcher.spin_rate_7d)

    # ── 4. Catcher framing into TeamProfile ───────────────────────────
    for team_profile in (pick.home_team_profile, pick.away_team_profile):
        team_abbr = team_profile.name[:3].upper()
        framing = t1_framing_by_team.get(team_abbr)
        if framing is not None:
            team_profile.framing_runs = framing

    # ── 5. Umpire data attached to pick as custom attr ────────────────
    ump_data = t1_umpires.get(str(pick.game_id), {})
    pick._umpire_data = ump_data  # type: ignore[attr-defined]

    # ── 6. Bullpen fatigue into TeamProfile ───────────────────────────
    for team_profile in (pick.home_team_profile, pick.away_team_profile):
        team_abbr = team_profile.name[:3].upper()
        fatigue_entry = t1_bullpen_fatigue.get(team_abbr, {})
        if fatigue_entry:
            team_profile.bullpen_fatigue_score  = fatigue_entry.get("fatigue_score",   0.0)
            team_profile.bullpen_arms_available = fatigue_entry.get("arms_available",  3)


# ------------------------------------------------------------------ #
#  Tier 2 enrichment injection                                        #
# ------------------------------------------------------------------ #

def _inject_tier2_data(
    pick,
    home: str,
    game_id: str,
    t2_csw_data: dict,
    t2_stuff_data: dict,
    t2_travel_data: dict,
    t2_defense_data: dict,
    t2_pitch_mix_data: dict,
) -> None:
    """
    Mutates PitcherProfile fields, TeamProfile fields, and WeatherProfile on the pick
    with Tier 2 enrichment data fetched before the per-game loop.

    Follows the exact same pattern as _inject_tier1_data: try full name first,
    fall back to last name only; only overwrite when real data is present.
    """
    # ── 1. CSW% and Stuff+ into PitcherProfile ────────────────────────
    for pitcher in (pick.home_pitcher, pick.away_pitcher):
        if pitcher.name in ("TBD", ""):
            continue

        # CSW rate
        csw = t2_csw_data.get(pitcher.name)
        if csw is None:
            last = pitcher.name.split()[-1]
            csw = t2_csw_data.get(last)
        if csw is not None and csw > 0:
            pitcher.csw_rate = csw

        # Stuff+
        stuff = t2_stuff_data.get(pitcher.name)
        if stuff is None:
            last = pitcher.name.split()[-1]
            stuff = t2_stuff_data.get(last)
        if stuff is not None and stuff > 0:
            pitcher.stuff_plus = stuff

    # ── 2. Air density into WeatherProfile ────────────────────────────
    # Air density is fetched per game (home team + approximate game time)
    try:
        air = get_air_density_for_game(home_team=home)
        if air:
            pick.weather.air_density_ratio = air.get("air_density_ratio", 1.0)
            pick.weather.carry_boost_pct   = air.get("carry_boost_pct",   0.0)
    except Exception as exc:
        logger.debug("_inject_tier2_data: air density failed for %s: %s", home, exc)

    # ── 3. Travel fatigue into TeamProfile ────────────────────────────
    for team_profile in (pick.home_team_profile, pick.away_team_profile):
        travel_entry = t2_travel_data.get(team_profile.name, {})
        if travel_entry:
            team_profile.travel_tz_change = travel_entry.get("tz_change_hours", 0.0)
            team_profile.travel_fatigue   = travel_entry.get("fatigue_flag",    False)

    # ── 4. Defensive OAA into TeamProfile ─────────────────────────────
    for team_profile in (pick.home_team_profile, pick.away_team_profile):
        # Try 3-letter abbreviation first (matches Baseball Savant format)
        team_abbr = team_profile.name[:3].upper()
        defense_entry = t2_defense_data.get(team_abbr, {})
        if defense_entry:
            team_profile.team_oaa             = defense_entry.get("total_oaa",           0.0)
            team_profile.defensive_run_value   = defense_entry.get("defensive_run_value", 0.0)

    # ── 5. Pitch mix changes into PitcherProfile ──────────────────────
    for pitcher in (pick.home_pitcher, pick.away_pitcher):
        if pitcher.name in ("TBD", ""):
            continue
        mix_entry = t2_pitch_mix_data.get(pitcher.name)
        if mix_entry is None:
            last = pitcher.name.split()[-1]
            mix_entry = t2_pitch_mix_data.get(last)
        if mix_entry:
            pitcher.pitch_mix_change  = mix_entry.get("mix_change",  False)
            pitcher.pitch_mix_details = mix_entry.get("details",     "")
            # Store change_type as a custom attr for layer_21 to read
            pitcher._pitch_mix_change_type = mix_entry.get("change_type", None)  # type: ignore[attr-defined]


# ------------------------------------------------------------------ #
#  Tier 3 enrichment injection                                        #
# ------------------------------------------------------------------ #

def _inject_tier3_data(
    pick,
    t3_luck_data: dict,
    t3_platoon_data: dict,
    t3_lineup_data: dict,
    t3_bat_speed_data: dict,
) -> None:
    """
    Mutates TeamProfile fields on the pick with Tier 3 enrichment data
    fetched before the per-game loop.

    Follows the exact same pattern as _inject_tier1_data and _inject_tier2_data:
    try 3-letter abbreviation first; only overwrite when real data is present.
    """
    # ── 1. Luck metrics (BABIP, LOB%, luck_score) into TeamProfile ────
    for team_profile in (pick.home_team_profile, pick.away_team_profile):
        team_abbr = team_profile.name[:3].upper()
        luck_entry = t3_luck_data.get(team_abbr, {})
        if luck_entry:
            team_profile.babip      = luck_entry.get("babip",      team_profile.babip)
            team_profile.lob_pct    = luck_entry.get("lob_pct",    team_profile.lob_pct)
            team_profile.luck_score = luck_entry.get("luck_score",  team_profile.luck_score)

    # ── 2. Platoon splits (wRC+ vs LHP / RHP) into TeamProfile ───────
    for team_profile in (pick.home_team_profile, pick.away_team_profile):
        team_abbr = team_profile.name[:3].upper()
        platoon_entry = t3_platoon_data.get(team_abbr, {})
        if platoon_entry:
            team_profile.wrc_vs_lhp = platoon_entry.get("wrc_vs_lhp", team_profile.wrc_vs_lhp)
            team_profile.wrc_vs_rhp = platoon_entry.get("wrc_vs_rhp", team_profile.wrc_vs_rhp)

    # ── 3. Day-of lineup data into TeamProfile ────────────────────────
    for team_profile in (pick.home_team_profile, pick.away_team_profile):
        # Try 3-letter abbreviation (RotoWire / MLB API use these)
        team_abbr = team_profile.name[:3].upper()
        lineup_entry = t3_lineup_data.get(team_abbr, {})
        if not lineup_entry:
            # Fallback: full team name (some sources return full names)
            lineup_entry = t3_lineup_data.get(team_profile.name, {})
        if lineup_entry:
            team_profile.lineup_confirmed   = lineup_entry.get("lineup_confirmed",    team_profile.lineup_confirmed)
            team_profile.lineup_value_score = lineup_entry.get("value_score",          team_profile.lineup_value_score)

    # ── 4. Bat speed & sprint speed into TeamProfile ──────────────────
    for team_profile in (pick.home_team_profile, pick.away_team_profile):
        team_abbr = team_profile.name[:3].upper()
        bat_entry = t3_bat_speed_data.get(team_abbr, {})
        if bat_entry:
            team_profile.avg_bat_speed    = bat_entry.get("avg_bat_speed",    team_profile.avg_bat_speed)
            team_profile.avg_sprint_speed = bat_entry.get("avg_sprint_speed", team_profile.avg_sprint_speed)
            team_profile.is_speed_team    = bat_entry.get("is_speed_team",    team_profile.is_speed_team)
