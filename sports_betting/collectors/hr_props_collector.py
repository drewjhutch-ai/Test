"""
HR Prop data collector.
Pulls batter HR prop odds from The Odds API and Statcast batter metrics via pybaseball.
Builds ranked HR candidates and 3-leg parlay recommendations.
"""
from __future__ import annotations
import logging
import requests
from datetime import date, datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Statcast batter data ──────────────────────────────────────────────

def get_statcast_batter_data(season: int = None) -> dict[str, dict]:
    """
    Pull batter Statcast data (barrel %, hard-hit %, xwOBA, HR rate) via pybaseball.
    Returns dict keyed by lowercase last name.
    """
    if season is None:
        season = date.today().year

    cache_key = f"statcast_batters_{date.today().isoformat()}"
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            logger.info("Statcast batter data loaded from cache (%d batters)", len(data))
            return data
        except Exception:
            pass

    try:
        import pybaseball
        pybaseball.cache.enable()
        df = pybaseball.statcast_batter_exitvelo_barrels(season, minBBE=50)

        result = {}
        for _, row in df.iterrows():
            name = str(row.get("last_name, first_name", "")).strip()
            if not name:
                continue
            parts = name.split(", ")
            last = parts[0].lower().strip() if parts else name.lower()
            first = parts[1].lower().strip() if len(parts) > 1 else ""

            result[last] = {
                "name": name,
                "first": first,
                "barrel_rate": float(row.get("brl_percent", 0) or 0) / 100,
                "hard_hit_rate": float(row.get("hard_hit_percent", 0) or 0) / 100,
                "avg_exit_velo": float(row.get("avg_hit_speed", 88) or 88),
                "xwoba": float(row.get("xwoba", 0.320) or 0.320),
                "bbs": int(row.get("brl_pa", 0) or 0),
            }

        with open(cache_path, "w") as f:
            json.dump(result, f)

        logger.info("Statcast batter data fetched: %d batters", len(result))
        return result

    except Exception as e:
        logger.warning("Could not fetch Statcast batter data: %s", e)
        return {}


def get_batting_stats(season: int = None) -> dict[str, dict]:
    """Pull FanGraphs batting stats for HR rate, wOBA splits via pybaseball."""
    if season is None:
        season = date.today().year

    cache_key = f"fg_batting_{date.today().isoformat()}"
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            return data
        except Exception:
            pass

    try:
        from pybaseball import batting_stats_bref
        df = batting_stats_bref(season)

        result = {}
        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            if not name:
                continue
            last  = name.split()[-1].lower()
            first = name.split()[0].lower()

            games = float(row.get("G", 1) or 1)
            hrs   = float(row.get("HR", 0) or 0)
            ba    = float(row.get("BA", 0.250) or 0.250)
            slg   = float(row.get("SLG", 0.400) or 0.400)
            obp   = float(row.get("OBP", 0.320) or 0.320)
            iso   = round(slg - ba, 3)

            result[last] = {
                "name":     name,
                "first":    first,
                "team":     str(row.get("Tm", "")),
                "hr_rate":  round(hrs / max(1, games), 4),
                "hrs":      int(hrs),
                "games":    int(games),
                "woba":     round(obp * 0.45 + slg * 0.55, 4),  # approximation
                "iso":      iso,
                "pull_pct": 0.40,  # not in BRef; use league average default
            }

        with open(cache_path, "w") as f:
            json.dump(result, f)

        logger.info("BRef batting stats fetched: %d batters", len(result))
        return result

    except Exception as e:
        logger.warning("Could not fetch FanGraphs batting stats: %s", e)
        return {}


# ── HR prop odds from The Odds API ────────────────────────────────────

def get_hr_prop_odds() -> list[dict]:
    """
    Fetch batter HR prop odds from The Odds API.
    Returns list of dicts: {batter, team, game, price_over, book, event_id}.
    """
    import os
    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        logger.warning("No ODDS_API_KEY — cannot pull HR prop odds")
        return _mock_hr_odds()

    cache_key = f"hr_odds_{date.today().isoformat()}"
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            logger.info("HR odds loaded from cache (%d props)", len(data))
            return data
        except Exception:
            pass

    try:
        # First get today's events
        events_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
        events_resp = requests.get(events_url, params={"apiKey": api_key}, timeout=10)
        events_resp.raise_for_status()
        events = events_resp.json()

        today = date.today().isoformat()
        today_events = [
            e for e in events
            if e.get("commence_time", "")[:10] == today
        ]

        props = []
        for event in today_events[:8]:  # limit API calls
            event_id = event.get("id")
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            try:
                odds_url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
                resp = requests.get(odds_url, params={
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": "batter_home_runs",
                    "oddsFormat": "american",
                    "bookmakers": "draftkings,fanduel",
                }, timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()

                for book in data.get("bookmakers", []):
                    book_name = book.get("key", "")
                    for market in book.get("markets", []):
                        if market.get("key") != "batter_home_runs":
                            continue
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == "Over":
                                props.append({
                                    "batter": outcome.get("description", "Unknown"),
                                    "game": f"{away} @ {home}",
                                    "home_team": home,
                                    "away_team": away,
                                    "price": int(outcome.get("price", -130)),
                                    "point": float(outcome.get("point", 0.5)),
                                    "book": book_name,
                                    "event_id": event_id,
                                })
            except Exception as e:
                logger.debug("HR prop fetch failed for event %s: %s", event_id, e)
                continue

        if props:
            with open(cache_path, "w") as f:
                json.dump(props, f)

        logger.info("HR prop odds fetched: %d props", len(props))
        return props

    except Exception as e:
        logger.warning("HR prop odds fetch failed: %s", e)
        return _mock_hr_odds()


def _mock_hr_odds() -> list[dict]:
    """Fallback mock HR props when API unavailable."""
    return [
        {"batter": "Aaron Judge",   "game": "NYY @ BAL", "home_team": "Baltimore Orioles",
         "away_team": "New York Yankees", "price": +130, "point": 0.5, "book": "draftkings"},
        {"batter": "Shohei Ohtani", "game": "SF @ LAD",  "home_team": "Los Angeles Dodgers",
         "away_team": "San Francisco Giants", "price": +115, "point": 0.5, "book": "draftkings"},
        {"batter": "Kyle Schwarber","game": "PHI @ BOS", "home_team": "Boston Red Sox",
         "away_team": "Philadelphia Phillies","price": +140, "point": 0.5, "book": "draftkings"},
        {"batter": "Pete Alonso",   "game": "NYM @ ATL", "home_team": "Atlanta Braves",
         "away_team": "New York Mets",   "price": +155, "point": 0.5, "book": "draftkings"},
        {"batter": "Yordan Alvarez","game": "HOU @ SEA", "home_team": "Seattle Mariners",
         "away_team": "Houston Astros",  "price": +125, "point": 0.5, "book": "draftkings"},
        {"batter": "Matt Olson",    "game": "ATL @ NYM", "home_team": "New York Mets",
         "away_team": "Atlanta Braves",  "price": +135, "point": 0.5, "book": "draftkings"},
        {"batter": "Bryce Harper",  "game": "PHI @ BOS", "home_team": "Boston Red Sox",
         "away_team": "Philadelphia Phillies","price": +145, "point": 0.5, "book": "draftkings"},
        {"batter": "Vladimir Guerrero Jr.", "game": "TOR @ TB", "home_team": "Tampa Bay Rays",
         "away_team": "Toronto Blue Jays","price": +140, "point": 0.5, "book": "draftkings"},
    ]


# ── HR candidate scoring ──────────────────────────────────────────────

# Known power hitters with reliable stats as fallback when pybaseball unavailable
KNOWN_HR_HITTERS: dict[str, dict] = {
    "aaron judge":              {"hr_rate": 0.28, "barrel_rate": 0.22, "hard_hit_rate": 0.57, "hand": "R"},
    "shohei ohtani":            {"hr_rate": 0.24, "barrel_rate": 0.20, "hard_hit_rate": 0.53, "hand": "L"},
    "kyle schwarber":           {"hr_rate": 0.22, "barrel_rate": 0.18, "hard_hit_rate": 0.50, "hand": "L"},
    "pete alonso":              {"hr_rate": 0.20, "barrel_rate": 0.17, "hard_hit_rate": 0.48, "hand": "R"},
    "yordan alvarez":           {"hr_rate": 0.23, "barrel_rate": 0.21, "hard_hit_rate": 0.56, "hand": "L"},
    "matt olson":               {"hr_rate": 0.21, "barrel_rate": 0.17, "hard_hit_rate": 0.49, "hand": "L"},
    "bryce harper":             {"hr_rate": 0.19, "barrel_rate": 0.16, "hard_hit_rate": 0.47, "hand": "L"},
    "vladimir guerrero jr.":    {"hr_rate": 0.18, "barrel_rate": 0.15, "hard_hit_rate": 0.46, "hand": "R"},
    "vladimir guerrero":        {"hr_rate": 0.18, "barrel_rate": 0.15, "hard_hit_rate": 0.46, "hand": "R"},
    "freddie freeman":          {"hr_rate": 0.17, "barrel_rate": 0.14, "hard_hit_rate": 0.44, "hand": "L"},
    "manny machado":            {"hr_rate": 0.17, "barrel_rate": 0.14, "hard_hit_rate": 0.44, "hand": "R"},
    "rafael devers":            {"hr_rate": 0.19, "barrel_rate": 0.16, "hard_hit_rate": 0.47, "hand": "L"},
    "paul goldschmidt":         {"hr_rate": 0.17, "barrel_rate": 0.15, "hard_hit_rate": 0.45, "hand": "R"},
    "nolan arenado":            {"hr_rate": 0.18, "barrel_rate": 0.15, "hard_hit_rate": 0.46, "hand": "R"},
    "cody bellinger":           {"hr_rate": 0.15, "barrel_rate": 0.13, "hard_hit_rate": 0.42, "hand": "L"},
    "juan soto":                {"hr_rate": 0.16, "barrel_rate": 0.14, "hard_hit_rate": 0.43, "hand": "L"},
    "default":                  {"hr_rate": 0.12, "barrel_rate": 0.09, "hard_hit_rate": 0.38, "hand": "R"},
}


def score_hr_candidates(
    hr_odds: list[dict],
    batting_stats: dict,
    statcast_data: dict,
    weather_by_game: dict,
) -> list[dict]:
    """
    Score each HR prop candidate using all available data.
    Returns list sorted by composite score descending.
    """
    try:
        from ..model_v4.hr_props import HRPropInput, evaluate_hr_prop
    except Exception:
        from sports_betting.model_v4.hr_props import HRPropInput, evaluate_hr_prop

    scored = []
    for prop in hr_odds:
        try:
            batter_name = prop.get("batter", "")
            home_team   = prop.get("home_team", "")
            game        = prop.get("game", "")
            price       = prop.get("price", -130)

            lower_name = batter_name.lower()
            last       = batter_name.split()[-1].lower() if batter_name else ""

            # Use pybaseball data when available, fall back to known-hitter DB, then defaults
            bat = batting_stats.get(last, {})
            sc  = statcast_data.get(last, {})
            known = KNOWN_HR_HITTERS.get(lower_name) or KNOWN_HR_HITTERS.get(last) or KNOWN_HR_HITTERS["default"]

            hr_rate     = bat.get("hr_rate")     or known["hr_rate"]
            barrel_rate = sc.get("barrel_rate")  or known["barrel_rate"]
            hard_hit    = sc.get("hard_hit_rate") or known["hard_hit_rate"]
            xwoba       = sc.get("xwoba", 0.340)
            batter_hand = known["hand"]

            wx         = weather_by_game.get(game, {})
            temp       = wx.get("temperature", 72)
            wind_speed = wx.get("wind_speed", 5)
            wind_dir   = wx.get("wind_direction", "calm")

            inp = HRPropInput(
                batter_name=batter_name,
                batter_hand=batter_hand,
                team=prop.get("away_team", ""),
                pitcher_name="Unknown",
                pitcher_hand="R",
                home_team=home_team,
                hr_rate=hr_rate,
                pitcher_hr9=1.25,
                pitcher_barrel_rate=0.09,
                pitcher_hard_hit_rate=0.37,
                pitcher_hr_fb_rate=0.13,
                temperature=temp,
                wind_speed=wind_speed,
                wind_direction=wind_dir,
                batter_woba_vs_hand=xwoba,
                price=price,
            )

            result        = evaluate_hr_prop(inp)
            composite     = result["composite_probability"]
            ev            = result["ev_pct"]
            factors_passed = result["factors_passed"]

            factor_notes = [v["note"] for v in result["factors"].values() if v["score"] > 0]

            scored.append({
                "batter":        batter_name,
                "game":          game,
                "home_team":     home_team,
                "price":         price,
                "book":          prop.get("book", "draftkings"),
                "hr_rate":       hr_rate,
                "barrel_rate":   barrel_rate,
                "hard_hit_rate": hard_hit,
                "composite_prob": composite,
                "ev_pct":        ev,
                "factors_passed": factors_passed,
                "recommendation": result["recommendation"],
                "factor_notes":  factor_notes[:3],
                "starred":       factors_passed >= 4 and ev > 0,
            })
        except Exception as e:
            logger.debug("HR scoring failed for %s: %s", prop.get("batter","?"), e)
            continue

    scored.sort(key=lambda x: (x["factors_passed"], x["composite_prob"]), reverse=True)
    return scored


# ── 3-leg HR parlay builder ───────────────────────────────────────────

def build_hr_parlays(candidates: list[dict]) -> list[dict]:
    """
    Build 3-leg HR parlays from top candidates.
    Prioritizes: different games, highest composite probability, positive EV.
    Returns list of parlay dicts.
    """
    import math

    # Take top candidates from different games (max 1 per game)
    seen_games = set()
    top = []
    for c in candidates:
        if c["game"] not in seen_games and c["composite_prob"] >= 0.10:
            top.append(c)
            seen_games.add(c["game"])
        if len(top) >= 6:
            break

    if len(top) < 3:
        top = candidates[:6]  # fallback: use all if not enough unique games

    def american_to_decimal(p):
        return (p / 100 + 1) if p > 0 else (100 / abs(p) + 1)

    def decimal_to_american(d):
        if d >= 2.0:
            return int((d - 1) * 100)
        return int(-100 / (d - 1))

    parlays = []

    # Best 3-leg: top 3 picks by composite prob
    if len(top) >= 3:
        legs = top[:3]
        combined_dec = math.prod(american_to_decimal(l["price"]) for l in legs)
        combined_prob = math.prod(l["composite_prob"] for l in legs)
        ev = (combined_prob * combined_dec) - 1
        all_starred = all(l.get("starred") for l in legs)
        parlays.append({
            "label": "HR Parlay A — Best 3",
            "legs": legs,
            "combined_odds": decimal_to_american(combined_dec),
            "combined_prob": round(combined_prob, 4),
            "ev_pct": round(ev, 4),
            "starred": all_starred or sum(l.get("starred", False) for l in legs) >= 2,
            "stake_rec": "$5–$10",
        })

    # Value 3-leg: best EV legs
    ev_sorted = sorted(top, key=lambda x: x["ev_pct"], reverse=True)
    if len(ev_sorted) >= 3:
        legs = ev_sorted[:3]
        combined_dec = math.prod(american_to_decimal(l["price"]) for l in legs)
        combined_prob = math.prod(l["composite_prob"] for l in legs)
        ev = (combined_prob * combined_dec) - 1
        parlays.append({
            "label": "HR Parlay B — Best EV",
            "legs": legs,
            "combined_odds": decimal_to_american(combined_dec),
            "combined_prob": round(combined_prob, 4),
            "ev_pct": round(ev, 4),
            "starred": ev > 0.15,
            "stake_rec": "$5",
        })

    # Moonshot 3-leg: highest odds (most +money legs)
    price_sorted = sorted(top, key=lambda x: x["price"], reverse=True)
    if len(price_sorted) >= 3:
        legs = price_sorted[:3]
        combined_dec = math.prod(american_to_decimal(l["price"]) for l in legs)
        combined_prob = math.prod(l["composite_prob"] for l in legs)
        ev = (combined_prob * combined_dec) - 1
        parlays.append({
            "label": "HR Parlay C — Moonshot",
            "legs": legs,
            "combined_odds": decimal_to_american(combined_dec),
            "combined_prob": round(combined_prob, 4),
            "ev_pct": round(ev, 4),
            "starred": False,
            "stake_rec": "$3–$5",
        })

    return parlays


def run_hr_parlay_analysis(games: list[dict], weather_by_game: dict) -> dict:
    """
    Full HR parlay pipeline. Returns candidates + parlays for the web app.
    Never raises — always returns a valid dict.
    """
    try:
        batting  = get_batting_stats()
        statcast = get_statcast_batter_data()
        hr_odds  = get_hr_prop_odds()

        if not hr_odds:
            return {"candidates": [], "parlays": [], "data_note": "No HR prop odds available today."}

        candidates = score_hr_candidates(hr_odds, batting, statcast, weather_by_game)
        parlays    = build_hr_parlays(candidates)

        return {
            "candidates": candidates,
            "parlays":    parlays,
            "data_note":  f"Scored {len(candidates)} HR props · {len(parlays)} parlays built",
        }
    except Exception as e:
        logger.warning("HR parlay analysis failed: %s", e)
        return {"candidates": [], "parlays": [], "data_note": f"HR analysis error: {e}"}
