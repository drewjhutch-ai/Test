"""
Signal 5: Closing Line Value (CLV) Tracker
The gold standard metric for measuring betting edge.
Compares our bet price to Pinnacle's closing line.
+2% CLV consistently over 200+ bets = genuine edge confirmed.
"""
from __future__ import annotations
import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)


def get_pinnacle_closing_odds(game_id: str, backing_team: str) -> int | None:
    """
    Fetch Pinnacle's closing odds for a given game/team from The Odds API.
    Returns American odds integer or None if unavailable.
    """
    import os, requests
    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        return None

    try:
        from datetime import datetime
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds-history"
        resp = requests.get(url, params={
            "apiKey": api_key,
            "regions": "eu",  # Pinnacle available in EU region
            "markets": "h2h",
            "oddsFormat": "american",
            "bookmakers": "pinnacle",
            "date": yesterday,
        }, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        for event in data.get("data", []):
            for bm in event.get("bookmakers", []):
                if bm.get("key") != "pinnacle":
                    continue
                for market in bm.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        if backing_team.lower() in outcome.get("name", "").lower():
                            return int(outcome.get("price", 0))
    except Exception as e:
        logger.debug("CLV fetch failed: %s", e)
    return None


def compute_clv(our_price: int, closing_price: int) -> float:
    """
    Compute CLV as percentage edge vs closing line.
    Positive = we got a better price than the market closed at.
    """
    def to_prob(american: int) -> float:
        if american > 0:
            return 100 / (american + 100)
        return abs(american) / (abs(american) + 100)

    if our_price == 0 or closing_price == 0:
        return 0.0

    our_prob = to_prob(our_price)
    close_prob = to_prob(closing_price)
    # CLV = how much better our implied prob is vs close (lower our_prob = better odds for us)
    return round(close_prob - our_prob, 4)


def get_clv_summary() -> dict:
    """
    Compute CLV summary from bet history in the database.
    Returns average CLV, count, and assessment.
    """
    try:
        from ..database import get_db
        with get_db() as conn:
            rows = conn.execute("""
                SELECT our_price, closing_price
                FROM value_bets
                WHERE closing_price IS NOT NULL AND our_price IS NOT NULL
                  AND date >= date('now', '-90 days')
            """).fetchall()
    except Exception:
        rows = []

    if not rows:
        return {
            "avg_clv": 0.0,
            "count": 0,
            "assessment": "Insufficient data — CLV builds as bets are recorded and graded",
            "is_sharp": False,
        }

    clvs = [compute_clv(r[0], r[1]) for r in rows if r[0] and r[1]]
    if not clvs:
        return {"avg_clv": 0.0, "count": 0, "assessment": "No CLV data yet", "is_sharp": False}

    avg = sum(clvs) / len(clvs)

    if avg >= 0.025 and len(clvs) >= 50:
        assessment = f"⭐ SHARP: +{avg:.1%} avg CLV over {len(clvs)} bets — model has genuine edge"
        is_sharp = True
    elif avg >= 0.010:
        assessment = f"Positive CLV trend (+{avg:.1%}) — keep building sample size"
        is_sharp = False
    elif avg >= 0:
        assessment = f"Marginally positive CLV (+{avg:.1%}) — need more data"
        is_sharp = False
    else:
        assessment = f"Negative CLV ({avg:.1%}) — model needs calibration"
        is_sharp = False

    return {
        "avg_clv": round(avg, 4),
        "count": len(clvs),
        "assessment": assessment,
        "is_sharp": is_sharp,
    }
