"""
Twilio SMS sender.
Formats the daily pick card as a clean text message and sends it.
Free trial at twilio.com gives ~$15 credit — enough for months of daily texts.
"""
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

UNIT_SIZE = 5  # $5 per unit


def send_daily_picks(results: dict) -> bool:
    """
    Format and send the full daily pick card as an SMS.
    Returns True if sent successfully, False otherwise.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token  = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_number = os.getenv("TWILIO_FROM_NUMBER", "")
    to_number   = os.getenv("TWILIO_TO_NUMBER", "")

    if not all([account_sid, auth_token, from_number, to_number]):
        logger.warning("Twilio credentials not set in .env — skipping SMS.")
        return False

    if account_sid == "your_twilio_account_sid_here":
        logger.warning("Twilio credentials are still placeholders — skipping SMS.")
        return False

    message = _format_message(results)

    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        msg = client.messages.create(
            body=message,
            from_=from_number,
            to=to_number,
        )
        logger.info("SMS sent successfully. SID: %s", msg.sid)
        return True
    except ImportError:
        logger.error("Twilio not installed. Run: pip install twilio")
        return False
    except Exception as e:
        logger.error("SMS send failed: %s", e)
        return False


def _format_message(results: dict) -> str:
    """Build a concise, readable SMS from the daily model results."""
    today = results.get("date", datetime.now().strftime("%Y-%m-%d"))
    picks = [p for p in results.get("picks", []) if p.tier != "SKIP"]
    parlays = results.get("parlays", [])
    nrfi_parlay = results.get("nrfi_parlay")
    roi = results.get("roi", {})

    lines = []
    lines.append(f"⚾ MLB PICKS — {today}")
    lines.append("=" * 32)

    # ROI line
    if roi and roi.get("total_bets"):
        roi_sign = "+" if roi.get("roi", 0) >= 0 else ""
        lines.append(
            f"Season: {roi.get('wins',0)}-{roi.get('losses',0)} | "
            f"ROI: {roi_sign}{roi.get('roi',0):.1f}%"
        )
        lines.append("")

    # Picks
    if not picks:
        lines.append("No picks today — pitchers TBD or no edge found.")
    else:
        tier_icons = {"STRONG": "🔥", "MEDIUM": "✅", "LEAN": "📌"}
        unit_map   = {"STRONG": 3,    "MEDIUM": 2,    "LEAN": 1}
        for i, p in enumerate(picks, 1):
            icon  = tier_icons.get(p.tier, "•")
            units = unit_map.get(p.tier, 1)
            bet   = units * UNIT_SIZE
            mkt   = _shorten_market(p.recommended_market or p.proposed_market)
            game  = f"{_short_name(p.away_team)} @ {_short_name(p.home_team)}"
            lines.append(
                f"{icon} {i}. {game}\n"
                f"   {mkt} | {p.tier} | {units}u/${bet} | Lose: {p.losing_pct:.0%}"
            )
            if p.factors:
                lines.append(f"   ↳ {p.factors[0][:55]}")

    lines.append("")
    lines.append("-" * 32)

    # Best parlay
    if parlays:
        p1 = parlays[0]
        p1.compute()
        legs_str = " + ".join(
            f"{_short_name(leg.pick.backing_team)} {_shorten_market(leg.market)}"
            for leg in p1.legs
        )
        lines.append(
            f"🎰 P1 ANCHOR: {legs_str}\n"
            f"   +{p1.american_odds} | ${p1.stake_low}-${p1.stake_high} | "
            f"Win ~${p1.payout_per_unit:.0f} | EV {p1.ev_pct:+.0%}"
        )

    # NRFI parlay
    if nrfi_parlay and isinstance(nrfi_parlay, dict):
        games_str = ", ".join(
            g.get("game", "?").split(" @ ")[-1]
            for g in nrfi_parlay.get("legs", [])[:3]
        )
        lines.append(
            f"\n🚫 NRFI {nrfi_parlay.get('type','')}: {games_str}...\n"
            f"   {nrfi_parlay.get('american_odds','?')} | "
            f"{nrfi_parlay.get('recommended_stake','?')} | "
            f"Win ~${nrfi_parlay.get('potential_win',0):.0f}"
        )

    lines.append("")
    lines.append("-" * 32)
    lines.append("HARD RULES REMINDER:")
    lines.append("❌ No totals in parlays")
    lines.append("❌ No LAA in parlays")
    lines.append("❌ No sweep G3 in parlays")
    lines.append("")
    lines.append("Good luck 🤞")

    full = "\n".join(lines)

    # SMS has a 1600 char limit — truncate gracefully if needed
    if len(full) > 1580:
        full = full[:1577] + "..."

    return full


def _short_name(team: str) -> str:
    """Shorten team name to city or abbreviation for SMS."""
    shorts = {
        "New York Yankees": "NYY", "New York Mets": "NYM",
        "Boston Red Sox": "BOS", "Chicago Cubs": "CHC",
        "Chicago White Sox": "CWS", "Los Angeles Dodgers": "LAD",
        "Los Angeles Angels": "LAA", "San Francisco Giants": "SF",
        "Oakland Athletics": "ATH", "Seattle Mariners": "SEA",
        "Houston Astros": "HOU", "Texas Rangers": "TEX",
        "Arizona Diamondbacks": "AZ", "Colorado Rockies": "COL",
        "Minnesota Twins": "MIN", "Detroit Tigers": "DET",
        "Cleveland Guardians": "CLE", "Kansas City Royals": "KC",
        "Milwaukee Brewers": "MIL", "St. Louis Cardinals": "STL",
        "Pittsburgh Pirates": "PIT", "Cincinnati Reds": "CIN",
        "Atlanta Braves": "ATL", "Miami Marlins": "MIA",
        "Tampa Bay Rays": "TB", "Baltimore Orioles": "BAL",
        "Washington Nationals": "WSH", "Philadelphia Phillies": "PHI",
        "Toronto Blue Jays": "TOR", "San Diego Padres": "SD",
    }
    return shorts.get(team, team.split()[-1][:3].upper())


def _shorten_market(market: str) -> str:
    if not market:
        return "ML"
    m = market.lower()
    if "f5" in m:
        return "F5 ML"
    if "full" in m or "h2h" in m:
        return "ML"
    if "k over" in m or "strikeout" in m:
        return "K Over"
    if "er under" in m:
        return "ER Undr"
    if "nrfi" in m:
        return "NRFI"
    if "yrfi" in m:
        return "YRFI"
    if "run line" in m:
        return "RL"
    return market[:8]
