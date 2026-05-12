"""
Line movement analysis.
Tracks how lines move from open to close and identifies
which direction sharp money is pushing.
"""
import logging
from datetime import datetime
from ..database import get_db

logger = logging.getLogger(__name__)


def get_line_movement_report(game_id: str) -> dict:
    """Full line movement report for a game across all books."""
    with get_db() as conn:
        movements = conn.execute("""
            SELECT * FROM line_movements
            WHERE game_id=?
            ORDER BY movement_time ASC
        """, (game_id,)).fetchall()

        snapshots = conn.execute("""
            SELECT * FROM odds_snapshots
            WHERE game_id=?
            ORDER BY snapshot_time ASC
        """, (game_id,)).fetchall()

    movements = [dict(m) for m in movements]
    snapshots = [dict(s) for s in snapshots]

    # Find opening and current lines per book
    books = {}
    for s in snapshots:
        book = s["book"]
        market = s["market"]
        key = (book, market)
        if key not in books:
            books[key] = {"opening": s, "current": s}
        else:
            books[key]["current"] = s

    movement_summary = {}
    for (book, market), data in books.items():
        opening = data["opening"]
        current = data["current"]

        summary = {
            "book": book,
            "market": market,
            "opening_home_price": opening.get("home_price"),
            "current_home_price": current.get("home_price"),
            "opening_away_price": opening.get("away_price"),
            "current_away_price": current.get("away_price"),
            "opening_total": opening.get("total"),
            "current_total": current.get("total"),
        }

        if opening.get("home_price") and current.get("home_price"):
            summary["home_price_move"] = current["home_price"] - opening["home_price"]
        if opening.get("total") and current.get("total"):
            summary["total_move"] = current["total"] - opening["total"]

        movement_summary[f"{book}_{market}"] = summary

    # Identify consensus move direction
    home_moves = [v.get("home_price_move", 0) for v in movement_summary.values()
                  if v.get("home_price_move")]
    avg_home_move = sum(home_moves) / len(home_moves) if home_moves else 0

    sharp_direction = "NEUTRAL"
    if avg_home_move > 3:
        sharp_direction = "AGAINST HOME (sharp on away)"
    elif avg_home_move < -3:
        sharp_direction = "TOWARD HOME (sharp on home)"

    return {
        "game_id": game_id,
        "movement_count": len(movements),
        "books_tracked": len(set(s["book"] for s in snapshots)),
        "avg_home_price_move": round(avg_home_move, 2),
        "sharp_direction": sharp_direction,
        "details": movement_summary,
        "raw_movements": movements[-20:],  # Last 20 moves
    }


def identify_sharp_patterns(game_id: str) -> list[str]:
    """Identify specific sharp betting patterns."""
    report = get_line_movement_report(game_id)
    patterns = []

    # Pattern 1: Consistent multi-book move
    if report["books_tracked"] >= 3 and abs(report["avg_home_price_move"]) >= 5:
        patterns.append(
            f"CONSENSUS MOVE: {report['books_tracked']} books moved "
            f"{'toward' if report['avg_home_move_price'] > 0 else 'away from'} home"
        )

    # Pattern 2: Rapid succession moves (steam)
    movements = report.get("raw_movements", [])
    if len(movements) >= 3:
        times = []
        for m in movements[-5:]:
            try:
                t = datetime.fromisoformat(m["movement_time"].replace("Z", ""))
                times.append(t)
            except (ValueError, TypeError):
                pass
        if times and len(times) >= 2:
            time_span = (times[-1] - times[0]).total_seconds()
            if time_span < 300:  # 5 minutes
                patterns.append(f"STEAM: {len(times)} moves in {time_span:.0f}s — rapid sharp action")

    # Pattern 3: Line moved but small (sharp positioning)
    for key, detail in report.get("details", {}).items():
        move = detail.get("home_price_move", 0)
        if 2 <= abs(move) <= 8:
            patterns.append(f"SHARP POSITION: {detail['book']} moved {move:+.0f} cents")

    return patterns


def get_opening_vs_current_summary(parsed_games: list[dict]) -> list[dict]:
    """Summary table of opening vs current lines for all today's games."""
    summaries = []
    for game in parsed_games:
        game_id = game["game_id"]
        with get_db() as conn:
            # Opening line (earliest snapshot)
            opening = conn.execute("""
                SELECT home_price, away_price, total, book FROM odds_snapshots
                WHERE game_id=? AND market='h2h'
                ORDER BY snapshot_time ASC LIMIT 1
            """, (game_id,)).fetchone()

            # Current line per book
            current_dk = conn.execute("""
                SELECT home_price, away_price FROM odds_snapshots
                WHERE game_id=? AND market='h2h' AND book='draftkings'
                ORDER BY snapshot_time DESC LIMIT 1
            """, (game_id,)).fetchone()

        if opening:
            summary = {
                "game_id": game_id,
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "opening_home": dict(opening).get("home_price"),
                "opening_away": dict(opening).get("away_price"),
                "opening_book": dict(opening).get("book"),
                "current_dk_home": dict(current_dk).get("home_price") if current_dk else None,
                "current_dk_away": dict(current_dk).get("away_price") if current_dk else None,
            }
            if summary["opening_home"] and summary["current_dk_home"]:
                summary["home_move"] = summary["current_dk_home"] - summary["opening_home"]
            summaries.append(summary)

    return summaries
