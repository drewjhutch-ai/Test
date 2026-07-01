"""
Outcome grader — pulls real MLB game results and grades all bet types.
Handles: ML, F5 ML, NRFI, YRFI, K Over/Under, Outs Recorded, Run Line,
         Game Over/Under, F5 Over/Under, Earned Runs.
"""
from __future__ import annotations
import logging
import requests
from datetime import date, datetime, timedelta
from ..database import get_db

logger = logging.getLogger(__name__)

MLB_BASE = "https://statsapi.mlb.com/api/v1"


def fetch_game_linescore(game_pk: int) -> dict:
    """Pull full linescore from MLB StatsAPI."""
    try:
        r = requests.get(f"{MLB_BASE}/game/{game_pk}/linescore", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug("Linescore fetch failed for %s: %s", game_pk, e)
        return {}


def fetch_game_boxscore(game_pk: int) -> dict:
    """Pull full boxscore (pitcher stats, etc.)."""
    try:
        r = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug("Boxscore fetch failed for %s: %s", game_pk, e)
        return {}


def fetch_completed_games(game_date: str | None = None) -> list[dict]:
    """
    Fetch all completed games for a given date (YYYY-MM-DD).
    Returns list of game dicts with game_pk, home, away, status, scores.
    """
    if game_date is None:
        game_date = (date.today() - timedelta(days=1)).isoformat()
    try:
        r = requests.get(
            f"{MLB_BASE}/schedule",
            params={"sportId": 1, "date": game_date, "hydrate": "linescore,probablePitcher,boxscore"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        games = []
        for date_entry in data.get("dates", []):
            for g in date_entry.get("games", []):
                status = g.get("status", {}).get("abstractGameState", "")
                if status != "Final":
                    continue
                ls = g.get("linescore", {})
                teams = g.get("teams", {})
                games.append({
                    "game_pk":    g.get("gamePk"),
                    "game_date":  game_date,
                    "status":     status,
                    "home_team":  teams.get("home", {}).get("team", {}).get("name", ""),
                    "away_team":  teams.get("away", {}).get("team", {}).get("name", ""),
                    "home_score": teams.get("home", {}).get("score", 0) or 0,
                    "away_score": teams.get("away", {}).get("score", 0) or 0,
                    "innings":    ls.get("innings", []),
                })
        return games
    except Exception as e:
        logger.warning("Completed games fetch failed: %s", e)
        return []


def get_f5_scores(innings: list) -> tuple[int, int]:
    """Sum runs through first 5 innings. Returns (home_f5, away_f5)."""
    home_f5 = sum(inn.get("home", {}).get("runs", 0) or 0 for inn in innings[:5])
    away_f5 = sum(inn.get("away", {}).get("runs", 0) or 0 for inn in innings[:5])
    return home_f5, away_f5


def get_inning1_scores(innings: list) -> tuple[int, int]:
    """Return runs scored in inning 1. Returns (home_i1, away_i1)."""
    if not innings:
        return 0, 0
    i1 = innings[0]
    return (i1.get("home", {}).get("runs", 0) or 0,
            i1.get("away", {}).get("runs", 0) or 0)


def get_pitcher_ks(game_pk: int, pitcher_name: str) -> int | None:
    """Pull strikeout total for a named pitcher from boxscore."""
    try:
        box = fetch_game_boxscore(game_pk)
        for side in ("home", "away"):
            pitchers = box.get("teams", {}).get(side, {}).get("pitchers", [])
            player_info = box.get("teams", {}).get(side, {}).get("players", {})
            for pid in pitchers:
                pkey = f"ID{pid}"
                p = player_info.get(pkey, {})
                name = p.get("person", {}).get("fullName", "")
                if pitcher_name.lower() in name.lower() or name.lower() in pitcher_name.lower():
                    ks = p.get("stats", {}).get("pitching", {}).get("strikeOuts", None)
                    if ks is not None:
                        return int(ks)
    except Exception as e:
        logger.debug("Pitcher K lookup failed: %s", e)
    return None


def grade_bet(bet: dict, game: dict) -> bool | None:
    """
    Determine WIN/LOSS for a single bet given a completed game dict.
    Returns True=WIN, False=LOSS, None=cannot grade.
    """
    side   = (bet.get("side") or "").lower().strip()
    market = (bet.get("market") or "").lower().replace(" ", "_").replace("-", "_")
    home_score = game["home_score"]
    away_score = game["away_score"]
    innings = game.get("innings", [])
    game_pk = game.get("game_pk")

    home_team = game["home_team"].lower()
    away_team = game["away_team"].lower()

    # Resolve side to home/away
    def is_home(s):
        return s in ("home", home_team) or any(tok in home_team for tok in s.split()[:2] if len(tok) > 3)
    def is_away(s):
        return s in ("away", away_team) or any(tok in away_team for tok in s.split()[:2] if len(tok) > 3)

    # Full game ML
    if any(kw in market for kw in ("full_game_ml", "h2h", "ml", "moneyline")):
        if is_home(side):  return home_score > away_score
        if is_away(side):  return away_score > home_score

    # Run line
    if "run_line" in market:
        spread = -1.5 if "_1.5" in market and "plus" not in market else 1.5
        if is_home(side):  return (home_score - away_score) > spread
        if is_away(side):  return (away_score - home_score) > spread

    # F5 ML
    if "f5_ml" in market or ("f5" in market and "ml" in market):
        h5, a5 = get_f5_scores(innings)
        if is_home(side):  return h5 > a5
        if is_away(side):  return a5 > h5

    # NRFI
    if "nrfi" in market:
        h1, a1 = get_inning1_scores(innings)
        return (h1 + a1) == 0

    # YRFI
    if "yrfi" in market:
        h1, a1 = get_inning1_scores(innings)
        return (h1 + a1) > 0

    # Game over/under (need line from factors or book_price field)
    if "game_over" in market or ("over" in market and "f5" not in market and "k" not in market):
        import json
        try:
            factors = json.loads(bet.get("factors") or "[]")
            for f in factors:
                if "total_line" in str(f).lower():
                    line = float(str(f).split(":")[-1].strip())
                    return (home_score + away_score) > line
        except Exception:
            pass
        return None

    if "game_under" in market or ("under" in market and "f5" not in market and "k" not in market):
        import json
        try:
            factors = json.loads(bet.get("factors") or "[]")
            for f in factors:
                if "total_line" in str(f).lower():
                    line = float(str(f).split(":")[-1].strip())
                    return (home_score + away_score) < line
        except Exception:
            pass
        return None

    # F5 Over/Under
    if "f5_over" in market:
        h5, a5 = get_f5_scores(innings)
        import json
        try:
            factors = json.loads(bet.get("factors") or "[]")
            for f in factors:
                if "total_line" in str(f).lower():
                    line = float(str(f).split(":")[-1].strip())
                    return (h5 + a5) > line
        except Exception:
            pass
        return None

    if "f5_under" in market:
        h5, a5 = get_f5_scores(innings)
        import json
        try:
            factors = json.loads(bet.get("factors") or "[]")
            for f in factors:
                if "total_line" in str(f).lower():
                    line = float(str(f).split(":")[-1].strip())
                    return (h5 + a5) < line
        except Exception:
            pass
        return None

    # K Over/Under
    if "k_over" in market or "strikeout" in market:
        import json
        import re
        pitcher = ""
        try:
            factors = json.loads(bet.get("factors") or "[]")
            for f in factors:
                if "pitcher:" in str(f).lower():
                    pitcher = str(f).split(":")[-1].strip()
        except Exception:
            pass
        if not pitcher:
            pitcher = side  # side might be pitcher name

        # Extract K line from market string e.g. "k_over_7.5"
        m = re.search(r"(\d+\.?\d*)", market)
        if m and game_pk:
            line = float(m.group(1))
            ks = get_pitcher_ks(game_pk, pitcher)
            if ks is not None:
                return ks > line if "over" in market else ks < line
        return None

    return None


def grade_all_pending(days_back: int = 3) -> int:
    """
    Main grading function. Pulls completed games for the past N days,
    matches them to pending (result IS NULL) value_bets, grades each.
    Returns count of bets graded.
    """
    graded = 0
    for offset in range(days_back):
        d = (date.today() - timedelta(days=offset)).isoformat()
        completed = fetch_completed_games(d)
        if not completed:
            continue

        # Build lookup: normalized team name → game
        game_lookup: dict[str, dict] = {}
        for g in completed:
            for name in (g["home_team"], g["away_team"]):
                # Store by last word of team name (e.g. "Yankees", "Dodgers")
                key = name.split()[-1].lower()
                game_lookup[key] = g
                # Also by game_date + both teams
                game_lookup[f"{g['home_team'].split()[-1].lower()}_{g['away_team'].split()[-1].lower()}"] = g

        with get_db() as conn:
            # Update games table with real scores
            for g in completed:
                conn.execute("""
                    UPDATE games SET status='Final', home_score=?, away_score=?
                    WHERE game_date=? AND (
                        home_team LIKE ? OR home_team LIKE ?
                    )
                """, (
                    g["home_score"], g["away_score"], d,
                    f"%{g['home_team'].split()[-1]}%",
                    f"%{g['home_team'][:8]}%",
                ))

            # Grade pending bets
            pending = conn.execute("""
                SELECT vb.id, vb.game_id, vb.market, vb.side, vb.book_price,
                       vb.recommended_bet, vb.factors, vb.confidence
                FROM value_bets vb
                WHERE vb.result IS NULL
                AND vb.confidence IN ('PLACED', 'MODEL_PICK', 'MODEL_PARLAY')
                AND DATE(vb.detected_at) = ?
            """, (d,)).fetchall()

            for bet in pending:
                bet = dict(bet)
                gid = bet.get("game_id", "")

                # Try to find matching completed game.
                matched_game = None

                # Most reliable: model game_ids are "mlb_<gamePk>" — match on the
                # numeric pk directly. (Team-name matching below is the fallback
                # for legacy rows whose game_id embeds a team name.)
                gid_digits = "".join(ch for ch in gid if ch.isdigit())
                if gid_digits:
                    for game in completed:
                        if str(game.get("game_pk")) == gid_digits:
                            matched_game = game
                            break

                if not matched_game:
                    for key, game in game_lookup.items():
                        if key in gid.lower() or gid.lower() in key:
                            matched_game = game
                            break
                # Broader match: any team name token in game_id
                if not matched_game:
                    for game in completed:
                        for team in (game["home_team"], game["away_team"]):
                            last = team.split()[-1].lower()
                            if last in gid.lower():
                                matched_game = game
                                break
                        if matched_game:
                            break

                if not matched_game:
                    continue

                won = grade_bet(bet, matched_game)
                if won is None:
                    continue

                price = bet.get("book_price", -110) or -110
                stake = bet.get("recommended_bet", 5) or 5
                if won:
                    profit = stake * (price / 100) if price > 0 else stake * (100 / abs(price))
                else:
                    profit = -stake

                conn.execute("""
                    UPDATE value_bets SET result=?, profit_loss=? WHERE id=?
                """, ("WIN" if won else "LOSS", round(profit, 2), bet["id"]))
                graded += 1

    if graded:
        logger.info("Outcome grader: graded %d bets", graded)
    return graded
