"""
Sharp money tracker.
Monitors betting percentages, line movements against public money,
steam moves, and reverse line movement to identify sharp action.
"""
import logging
import requests
from bs4 import BeautifulSoup
from ..config import SHARP_TICKET_PCT_THRESHOLD, SHARP_MONEY_PCT_THRESHOLD
from ..database import get_db, save_sharp_play

logger = logging.getLogger(__name__)


def detect_reverse_line_movement(game_id: str) -> list[dict]:
    """
    Reverse line movement (RLM): line moves AGAINST the side getting
    the majority of public bets. This signals sharp money on the other side.
    """
    with get_db() as conn:
        movements = conn.execute("""
            SELECT * FROM line_movements
            WHERE game_id=?
            ORDER BY movement_time DESC
            LIMIT 20
        """, (game_id,)).fetchall()

    rlm_signals = []
    for m in movements:
        m = dict(m)
        if not m.get("sharp_flag"):
            continue

        # If line moved toward a team (better odds) but public is on the other side
        # This implies sharp money caused the move
        # Without real ticket % data, we flag significant moves as potential sharp action
        move_size = abs((m.get("new_price") or 0) - (m.get("old_price") or 0))
        if move_size >= 3:  # 3+ cent move is noteworthy
            rlm_signals.append({
                "game_id": game_id,
                "side": m.get("side", "unknown"),
                "market": m.get("market"),
                "signal_type": "LINE_MOVEMENT",
                "signal_strength": min(1.0, move_size / 15),
                "opening_line": m.get("old_price"),
                "current_line": m.get("new_price"),
                "ticket_pct": None,
                "money_pct": None,
                "notes": f"Line moved {move_size:.1f} pts (potential sharp action)",
            })

    return rlm_signals


def scrape_action_network_percentages(game_id: str) -> dict | None:
    """
    Attempt to scrape public betting percentages from Action Network.
    Returns ticket% and money% for home team.
    """
    try:
        url = "https://www.actionnetwork.com/mlb/public-betting"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        # Action Network uses dynamic JS, so static scraping is limited
        # This is a placeholder — in production, integrate AN's API or use Selenium
        logger.debug("Action Network scrape returned %d bytes", len(resp.text))
        return None
    except Exception as e:
        logger.debug("Action Network scrape failed: %s", e)
        return None


def analyze_steam_move(game_id: str, book: str, market: str, threshold_minutes: int = 5) -> bool:
    """
    Steam move: multiple books move the same direction in rapid succession.
    This is one of the strongest sharp money indicators.
    """
    with get_db() as conn:
        recent_moves = conn.execute("""
            SELECT book, side, old_price, new_price, movement_time
            FROM line_movements
            WHERE game_id=? AND market=?
            AND movement_time >= datetime('now', ? || ' minutes')
            ORDER BY movement_time DESC
        """, (game_id, market, f"-{threshold_minutes}")).fetchall()

    if len(recent_moves) < 2:
        return False

    moves = [dict(m) for m in recent_moves]
    books_moved = set(m["book"] for m in moves)

    # Check if multiple books moved in the same direction
    directions = []
    for m in moves:
        if m.get("new_price") and m.get("old_price"):
            directions.append("up" if m["new_price"] > m["old_price"] else "down")

    if len(books_moved) >= 2 and len(set(directions)) == 1:
        logger.info("STEAM MOVE detected on %s %s: %d books moved %s",
                    game_id, market, len(books_moved), directions[0])
        return True
    return False


def detect_wiseguy_moves(parsed_games: list[dict]) -> list[dict]:
    """
    Comprehensive sharp money detection.
    Looks for:
    1. Significant line moves (3+ cents moneyline, 0.5+ spread)
    2. Steam moves across multiple books
    3. Pinnacle line leadership (sharps bet Pinnacle first)
    4. Line moves against public betting %
    """
    sharp_plays = []

    for game in parsed_games:
        game_id = game["game_id"]
        odds_by_book = game.get("odds_by_book", {})

        if not odds_by_book:
            continue

        # Pinnacle as sharp indicator: if Pinnacle has a notably different line
        pinnacle_h2h = odds_by_book.get("pinnacle", {}).get("h2h", {})
        dk_h2h = odds_by_book.get("draftkings", {}).get("h2h", {})

        if pinnacle_h2h and dk_h2h:
            pin_home = pinnacle_h2h.get("home_price")
            dk_home = dk_h2h.get("home_price")

            if pin_home and dk_home:
                diff = abs(pin_home - dk_home)
                if diff >= 5:
                    # Pinnacle is sharper than DK — indicates value discrepancy
                    favored_side = "home" if pin_home < dk_home else "away"
                    sharp_play = {
                        "game_id": game_id,
                        "side": favored_side,
                        "market": "h2h",
                        "signal_type": "PINNACLE_DISCREPANCY",
                        "signal_strength": min(1.0, diff / 20),
                        "opening_line": dk_home,
                        "current_line": pin_home,
                        "ticket_pct": None,
                        "money_pct": None,
                        "notes": f"Pinnacle ({pin_home}) vs DraftKings ({dk_home}): {diff} pt gap suggests sharp value on {favored_side}",
                    }
                    sharp_plays.append(sharp_play)
                    save_sharp_play(sharp_play)

        # Detect RLM signals
        rlm_signals = detect_reverse_line_movement(game_id)
        for signal in rlm_signals:
            sharp_plays.append(signal)
            save_sharp_play(signal)

        # Steam move check across all markets
        for market in ["h2h", "spreads", "totals"]:
            if analyze_steam_move(game_id, "multi", market):
                steam_play = {
                    "game_id": game_id,
                    "side": "steam",
                    "market": market,
                    "signal_type": "STEAM_MOVE",
                    "signal_strength": 0.85,
                    "opening_line": None,
                    "current_line": None,
                    "ticket_pct": None,
                    "money_pct": None,
                    "notes": f"Steam move detected across multiple books on {market}",
                }
                sharp_plays.append(steam_play)
                save_sharp_play(steam_play)

    logger.info("Sharp money analysis complete. %d signals detected.", len(sharp_plays))
    return sharp_plays


def get_consensus_sharp_side(game_id: str) -> dict:
    """Aggregate all sharp signals for a game into a consensus recommendation."""
    with get_db() as conn:
        signals = conn.execute("""
            SELECT side, market, signal_type, signal_strength, notes, detected_at
            FROM sharp_plays
            WHERE game_id=?
            AND detected_at >= datetime('now', '-12 hours')
            ORDER BY signal_strength DESC
        """, (game_id,)).fetchall()

    if not signals:
        return {"has_sharp_action": False}

    signals = [dict(s) for s in signals]
    side_strength = {}
    for s in signals:
        side = s["side"]
        side_strength[side] = side_strength.get(side, 0) + s["signal_strength"]

    consensus_side = max(side_strength, key=side_strength.get) if side_strength else None

    return {
        "has_sharp_action": True,
        "consensus_side": consensus_side,
        "total_signal_strength": round(sum(side_strength.values()), 2),
        "signal_count": len(signals),
        "signals": signals[:5],
    }
