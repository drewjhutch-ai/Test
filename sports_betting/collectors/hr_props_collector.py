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
        import pybaseball
        pybaseball.cache.enable()
        df = pybaseball.batting_stats(season, qual=50)

        result = {}
        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            if not name:
                continue
            last = name.split()[-1].lower() if name else ""
            first = name.split()[0].lower() if name else ""

            games = float(row.get("G", 1) or 1)
            hrs = float(row.get("HR", 0) or 0)

            result[last] = {
                "name": name,
                "first": first,
                "team": str(row.get("Team", "")),
                "hr_rate": round(hrs / max(1, games), 4),
                "hrs": int(hrs),
                "games": int(games),
                "woba": float(row.get("wOBA", 0.320) or 0.320),
                "iso": float(row.get("ISO", 0.150) or 0.150),
                "pull_pct": float(row.get("Pull%", 0.40) or 0.40),
            }

        with open(cache_path, "w") as f:
            json.dump(result, f)

        logger.info("FanGraphs batting stats fetched: %d batters", len(result))
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
    from ..model_v4.hr_props import HRPropInput, evaluate_hr_prop
    from ..model_v4.park_database import get_hr_factor

    scored = []
    for prop in hr_odds:
        batter_name = prop.get("batter", "")
        home_team = prop.get("home_team", "")
        game = prop.get("game", "")
        price = prop.get("price", -130)

        last = batter_name.split()[-1].lower() if batter_name else ""
        first = batter_name.split()[0].lower() if batter_name else ""

        # Match batter in batting stats
        bat = batting_stats.get(last, {})
        sc = statcast_data.get(last, {})

        hr_rate = bat.get("hr_rate", 0.10)
        barrel_rate = sc.get("barrel_rate", 0.08)
        hard_hit = sc.get("hard_hit_rate", 0.38)
        xwoba = sc.get("xwoba", 0.320)
        iso = bat.get("iso", 0.150)

        # Weather for this game
        wx = weather_by_game.get(game, {})
        temp = wx.get("temperature", 72)
        wind_speed = wx.get("wind_speed", 5)
        wind_dir = wx.get("wind_direction", "calm")

        # Handedness — default RHB vs RHP if unknown
        batter_hand = "R"
        woba_vs_hand = xwoba

        inp = HRPropInput(
            batter_name=batter_name,
            batter_hand=batter_hand,
            team=prop.get("away_team", ""),
            pitcher_name="Unknown",
            pitcher_hand="R",
            home_team=home_team,
            hr_rate=hr_rate,
            pitcher_hr9=1.20,          # league avg default
            pitcher_barrel_rate=barrel_rate * 0.8,
            pitcher_hard_hit_rate=hard_hit * 0.8,
            pitcher_hr_fb_rate=0.14,
            temperature=temp,
            wind_speed=wind_speed,
            wind_direction=wind_dir,
            batter_woba_vs_hand=woba_vs_hand,
            price=price,
        )

        result = evaluate_hr_prop(inp)
        composite = result["composite_probability"]
        ev = result["ev_pct"]
        factors_passed = result["factors_passed"]

        # Build readable factor summary
        factor_notes = []
        for k, v in result["factors"].items():
            if v["score"] > 0:
                factor_notes.append(v["note"])

        scored.append({
            "batter": batter_name,
            "game": game,
            "home_team": home_team,
            "price": price,
            "book": prop.get("book", "draftkings"),
            "hr_rate": hr_rate,
            "barrel_rate": barrel_rate,
            "hard_hit_rate": hard_hit,
            "composite_prob": composite,
            "ev_pct": ev,
            "factors_passed": factors_passed,
            "recommendation": result["recommendation"],
            "factor_notes": factor_notes[:3],
            "starred": factors_passed >= 4 and ev > 0,
        })

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
    """
    batting = get_batting_stats()
    statcast = get_statcast_batter_data()
    hr_odds = get_hr_prop_odds()

    if not hr_odds:
        return {"candidates": [], "parlays": [], "data_note": "No HR prop odds available today."}

    candidates = score_hr_candidates(hr_odds, batting, statcast, weather_by_game)
    parlays = build_hr_parlays(candidates)

    return {
        "candidates": candidates,
        "parlays": parlays,
        "data_note": f"Scored {len(candidates)} HR props · {len(parlays)} parlays built",
    }
