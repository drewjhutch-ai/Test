"""
MLB Betting Model v4.0 — Web Dashboard
Runs on Streamlit Cloud (free). Works on phone, tablet, any browser.
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd

# ── Streamlit Cloud: load secrets into env vars ──────────────────────
if hasattr(st, "secrets"):
    for key, val in st.secrets.items():
        if isinstance(val, str):
            os.environ.setdefault(key, val)

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="MLB Betting Model v4.0",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (avoids slow startup before UI appears) ─────────────
@st.cache_resource
def get_db_connection():
    from sports_betting.database import init_db
    init_db()

@st.cache_data(ttl=300)  # Cache 5 minutes
def cached_run_model(date_str: str):
    """Run the full model — cached so button presses don't re-run instantly."""
    from sports_betting.model_v4.daily_runner import run_daily_model
    return run_daily_model(date_str=date_str, verbose=False)

@st.cache_data(ttl=60)
def cached_roi():
    from sports_betting.models.trainer import compute_roi_summary
    return compute_roi_summary()

@st.cache_data(ttl=300)
def cached_signals():
    from sports_betting.analysis.signal_tracker import get_signal_performance_report
    return get_signal_performance_report()


# ── Helpers ───────────────────────────────────────────────────────────

TIER_EMOJI  = {"STRONG": "🔥", "MEDIUM": "✅", "LEAN": "📌", "SKIP": "❌"}
TIER_COLOR  = {"STRONG": "#00FF88", "MEDIUM": "#FFD700", "LEAN": "#FFFFFF", "SKIP": "#666"}
UNIT_MAP    = {"STRONG": 3, "MEDIUM": 2, "LEAN": 1}
UNIT_SIZE   = 5

def _short(team: str) -> str:
    shorts = {
        "New York Yankees":"NYY","New York Mets":"NYM","Boston Red Sox":"BOS",
        "Chicago Cubs":"CHC","Chicago White Sox":"CWS","Los Angeles Dodgers":"LAD",
        "Los Angeles Angels":"LAA","San Francisco Giants":"SF","Seattle Mariners":"SEA",
        "Houston Astros":"HOU","Texas Rangers":"TEX","Arizona Diamondbacks":"AZ",
        "Colorado Rockies":"COL","Minnesota Twins":"MIN","Detroit Tigers":"DET",
        "Cleveland Guardians":"CLE","Kansas City Royals":"KC","Milwaukee Brewers":"MIL",
        "St. Louis Cardinals":"STL","Pittsburgh Pirates":"PIT","Cincinnati Reds":"CIN",
        "Atlanta Braves":"ATL","Miami Marlins":"MIA","Tampa Bay Rays":"TB",
        "Baltimore Orioles":"BAL","Washington Nationals":"WSH",
        "Philadelphia Phillies":"PHI","Toronto Blue Jays":"TOR","San Diego Padres":"SD",
        "Oakland Athletics":"ATH",
    }
    return shorts.get(team, team.split()[-1][:3].upper())


# ── Sidebar ───────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Major_League_Baseball_logo.svg/320px-Major_League_Baseball_logo.svg.png",
                     width=120)
    st.sidebar.title("⚾ MLB Model v4.0")
    st.sidebar.markdown("---")

    date_input = st.sidebar.date_input(
        "Game Date",
        value=datetime.now().date(),
        min_value=datetime.now().date() - timedelta(days=1),
        max_value=datetime.now().date() + timedelta(days=3),
    )
    date_str = date_input.strftime("%Y-%m-%d")

    run_btn = st.sidebar.button("🔄 Run Model", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Hard Rules Active**")
    st.sidebar.markdown(
        "✅ Totals included when 3+ signals\n\n"
        "❌ No Sweep G3 in parlays\n\n"
        "❌ No debut K props\n\n"
        "❌ Max 3 picks/team/7 days"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Unit size:** $5  \n"
        "🔥 STRONG = 3u/$15  \n"
        "✅ MEDIUM = 2u/$10  \n"
        "📌 LEAN = 1u/$5"
    )

    return date_str, run_btn


# ── Main content ──────────────────────────────────────────────────────

def render_header(roi: dict):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Today", datetime.now().strftime("%b %d, %Y"))
    with col2:
        bets = roi.get("total_bets", 0)
        st.metric("Total Bets", bets)
    with col3:
        wins = roi.get("wins", 0)
        losses = roi.get("losses", 0)
        st.metric("Record", f"{wins}-{losses}")
    with col4:
        hit = roi.get("hit_rate", 0)
        st.metric("Hit Rate", f"{hit:.1%}")
    with col5:
        r = roi.get("roi", 0)
        delta_color = "normal" if r >= 0 else "inverse"
        st.metric("ROI", f"{r:+.2f}%", delta=f"{r:+.2f}%", delta_color=delta_color)


def render_picks_tab(picks: list):
    active = [p for p in picks if p.tier != "SKIP"]

    if not active:
        st.warning("⚠️ No picks today — pitchers may be TBD or no edge found. Try again after noon ET.")
        return

    # POTD banner
    potd = next((p for p in active if p.tier == "STRONG"), active[0])
    units = UNIT_MAP.get(potd.tier, 1)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1A1D26,#0E1117);
                border:2px solid #FFD700;border-radius:12px;padding:20px;margin-bottom:20px">
        <h2 style="color:#FFD700;margin:0">⭐ Pick of the Day</h2>
        <h3 style="color:#FAFAFA;margin:8px 0">
            {_short(potd.away_team)} @ {_short(potd.home_team)}
        </h3>
        <p style="color:#00D4AA;font-size:22px;font-weight:bold;margin:4px 0">
            BET: {_short(potd.backing_team)} &nbsp;({potd.recommended_market or potd.proposed_market})
        </p>
        <p style="color:#FFD700;font-size:16px;margin:4px 0">
            {potd.tier} — {units}u / ${units * UNIT_SIZE} &nbsp;|&nbsp;
            Lose probability: {potd.losing_pct:.0%}
        </p>
        <p style="color:#AAA;font-size:13px;margin:4px 0">
            {" &nbsp;·&nbsp; ".join(potd.factors[:4])}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Picks table
    rows = []
    for i, p in enumerate(active, 1):
        u = UNIT_MAP.get(p.tier, 1)
        rows.append({
            "":         TIER_EMOJI.get(p.tier, "•"),
            "Game":     f"{_short(p.away_team)} @ {_short(p.home_team)}",
            "BET THIS": f"➜ {_short(p.backing_team)}",
            "Market":   p.recommended_market or p.proposed_market or "ML",
            "Tier":     p.tier,
            "Bet":      f"{u}u / ${u * UNIT_SIZE}",
            "Factors":  p.factor_count,
            "Lose %":   f"{p.losing_pct:.0%}",
            "Top Edge": (p.factors[0][:55] if p.factors else "—"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "":          st.column_config.TextColumn("", width=30),
            "BET THIS":  st.column_config.TextColumn("BET THIS", width=110),
            "Tier":      st.column_config.TextColumn("Tier", width=80),
            "Bet":       st.column_config.TextColumn("Bet", width=90),
            "Factors":   st.column_config.NumberColumn("✓", width=50),
            "Lose %":    st.column_config.TextColumn("Lose %", width=70),
        }
    )

    # Losing scenarios expander
    with st.expander("📋 Losing Scenarios"):
        for p in active:
            u = UNIT_MAP.get(p.tier, 1)
            st.markdown(
                f"**{_short(p.away_team)} @ {_short(p.home_team)}** "
                f"({p.tier}, {u}u)  \n"
                f"_{p.losing_scenario}_"
            )


def render_parlays_tab(parlays: list, nrfi_parlay: dict | None):
    st.markdown("### 🎰 Daily Parlay Card")
    st.caption("All 5 parlays generated daily. ⭐ = model's highest-confidence selection. ⚠️ = below optimal EV threshold but best available legs.")

    if not parlays:
        st.info("No picks available to build parlays — run the model first.")
        return

    parlay_meta = [
        ("P1 — Anchor 2-Leg",  "🥇", "$35–$40"),
        ("P2 — Core 3-Leg",    "🥈", "$15–$20"),
        ("P3 — Science 4-Leg", "🥉", "$10–$15"),
        ("P4 — Push 5-Leg",    "🎯", "$5–$10"),
        ("P5 — Moonshot 6-Leg","🌙", "$5"),
    ]

    for i, parlay in enumerate(parlays):
        parlay.compute()
        label, icon, stake = parlay_meta[i] if i < len(parlay_meta) else (f"Parlay {i+1}", "🎰", "$5")

        # Star if EV is solidly positive, warn if forced below threshold
        star = "⭐ " if parlay.ev_pct >= 0.10 and not parlay.below_threshold else ""
        warn = " ⚠️ Below EV Threshold — best available legs" if parlay.below_threshold else ""

        with st.container():
            st.markdown(f"### {icon} {star}{label}{warn}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Odds",  f"+{parlay.american_odds:,}")
            col2.metric("Stake", stake)
            col3.metric("Win",   f"~${parlay.payout_per_unit:.0f}")
            ev_color = "normal" if parlay.ev_pct >= 0 else "inverse"
            col4.metric("EV", f"{parlay.ev_pct:+.1%}", delta_color=ev_color)

            leg_rows = []
            for j, leg in enumerate(parlay.legs, 1):
                leg_rows.append({
                    "Leg":      j,
                    "BET":      f"➜ {_short(leg.pick.backing_team)}",
                    "Game":     f"{_short(leg.pick.away_team)} @ {_short(leg.pick.home_team)}",
                    "Market":   leg.market[:20],
                    "Price":    f"{leg.price:+d}",
                    "Win Prob": f"{leg.true_prob:.0%}",
                })
            st.dataframe(pd.DataFrame(leg_rows), hide_index=True, use_container_width=True)

            note = parlay.independence_notes[0] if parlay.independence_notes else ""
            color = "green" if "PASS" in note else "orange"
            st.markdown(f":{color}[{note}]")
            st.markdown("---")

    # NRFI parlay bonus
    if nrfi_parlay and isinstance(nrfi_parlay, dict):
        st.markdown("### 🚫 NRFI Bonus Parlay")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Type",  nrfi_parlay.get("type", ""))
        col2.metric("Odds",  nrfi_parlay.get("american_odds", ""))
        col3.metric("Stake", nrfi_parlay.get("recommended_stake", ""))
        col4.metric("Win",   f"~${nrfi_parlay.get('potential_win', 0):.0f}")
        legs = nrfi_parlay.get("legs", [])
        if legs:
            rows = [{"Game": g.get("game","?"),
                     "NRFI %": f"{g.get('nrfi_probability',0):.1%}",
                     "Tier": g.get("tier","?")} for g in legs]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_nrfi_tab(nrfi_ranked: list, nrfi_parlay: dict | None = None):
    if not nrfi_ranked:
        st.info("No NRFI data available.")
        return

    # Top 2 starred picks
    nrfi_games = [g for g in nrfi_ranked if g.get("lean") == "NRFI"]
    yrfi_games = [g for g in nrfi_ranked if g.get("lean") == "YRFI"]
    top_nrfi = nrfi_games[:2]

    if top_nrfi:
        st.markdown("### ⭐ Top NRFI Picks Today")
        cols = st.columns(len(top_nrfi))
        for i, g in enumerate(top_nrfi):
            with cols[i]:
                prob = g.get("nrfi_probability", 0)
                st.markdown(f"""
                <div style="background:#1A2B1A;border:2px solid #FFD700;border-radius:10px;padding:14px;text-align:center">
                    <div style="color:#FFD700;font-size:22px">⭐</div>
                    <div style="color:#FAFAFA;font-weight:bold;font-size:15px">{g.get('game','?')}</div>
                    <div style="color:#00D4AA;font-size:24px;font-weight:bold">{prob:.0%} NRFI</div>
                    <div style="color:#AAA;font-size:12px">{g.get('home_pitcher','?')} vs {g.get('away_pitcher','?')}</div>
                    <div style="color:#888;font-size:11px">{g.get('tier','?')}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("")

    # NRFI/YRFI parlay
    eligible = [g for g in nrfi_ranked if g.get("parlay_eligible")]
    if len(eligible) >= 3:
        import math
        def nrfi_to_american(p):
            if p <= 0 or p >= 1:
                return -110
            dec = 1 / p
            if dec >= 2.0:
                return int((dec - 1) * 100)
            return int(-100 / (dec - 1))

        for parlay_size in [5, 4, 3]:
            legs = eligible[:parlay_size]
            if len(legs) < parlay_size:
                continue
            combined_prob = math.prod(g.get("nrfi_probability", 0.6) for g in legs)
            combined_dec = math.prod(1 / max(0.01, g.get("nrfi_probability", 0.6)) for g in legs)
            ev = (combined_prob * combined_dec) - 1
            am_odds = nrfi_to_american(combined_prob)
            if am_odds < 0:
                payout = abs(100 / am_odds) * 10 + 10
            else:
                payout = (am_odds / 100) * 10 + 10

            star = "⭐ " if ev > 0.05 else ""
            st.markdown(f"### 🚫 {star}NRFI {parlay_size}-Leg Parlay")
            c1, c2, c3 = st.columns(3)
            c1.metric("Combined Odds", f"+{abs(am_odds):,}" if am_odds > 0 else f"{am_odds:,}")
            c2.metric("Win Prob",  f"{combined_prob:.1%}")
            c3.metric("EV",       f"{ev:+.1%}")
            prows = [{"Game": g.get("game","?"), "NRFI %": f"{g.get('nrfi_probability',0):.1%}",
                      "SP (Home)": g.get("home_pitcher","?"), "SP (Away)": g.get("away_pitcher","?")}
                     for g in legs]
            st.dataframe(pd.DataFrame(prows), hide_index=True, use_container_width=True)
            st.caption("Stake recommendation: $10 for 3-leg · $7 for 4-leg · $5 for 5-leg")
            break  # show only the best size
        st.markdown("---")

    # Full rankings table
    st.markdown("### 📊 Full NRFI/YRFI Rankings")
    rows = []
    for i, g in enumerate(nrfi_ranked):
        nrfi = g.get("nrfi_probability", 0)
        lean = g.get("lean", "?")
        star = "⭐ " if i < 2 and lean == "NRFI" else ""
        rows.append({
            "":             star,
            "Game":         g.get("game", "?"),
            "Home SP":      g.get("home_pitcher", "?"),
            "Away SP":      g.get("away_pitcher", "?"),
            "NRFI %":       f"{nrfi:.1%}",
            "YRFI %":       f"{g.get('yrfi_probability', 0):.1%}",
            "Lean":         lean,
            "Parlay?":      "✅" if g.get("parlay_eligible") else "❌",
            "Tier":         g.get("tier", "?"),
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("""
    **Guide** · ✅ Parlay eligible = 70%+ · 🏟️ Dome = most reliable · ❄️ Cold = NRFI lean · 💨 Wind in = NRFI lean
    """)


def render_intelligence_tab(all_signals: dict, xwoba_luck: dict, games: list, sharp_plays: list = None):
    """Signal 📡 Intelligence — all 13 signals summarized per game."""
    from sports_betting.signals.clv_tracker import get_clv_summary

    st.markdown("### 📡 Model Intelligence Dashboard")
    st.caption("All 13 advanced signals running on today's slate. These feed directly into pick factor counts.")

    # CLV summary at top
    clv = get_clv_summary()
    if clv["count"] > 0:
        color = "green" if clv["is_sharp"] else "orange"
        st.markdown(f"**Closing Line Value (CLV):** :{color}[{clv['assessment']}]")
    else:
        st.info("📈 CLV tracking starts as soon as you record your first bet and games complete. It will tell you if the model is genuinely sharp over time.")

    if not all_signals:
        st.info("Signal data will appear after the model runs. Press Run Model.")
        return

    # Per-game signal breakdown
    st.markdown("---")
    for game_key, signals in all_signals.items():

        with st.expander(f"🔬 {game_key}", expanded=False):
            col1, col2 = st.columns(2)

            # Bullpen fatigue
            with col1:
                st.markdown("**💪 Bullpen Fatigue**")
                for side in ("home", "away"):
                    bp = signals.get("bullpen", {}).get(side, {})
                    level = bp.get("fatigue_level", "unknown")
                    score = bp.get("score", 0)
                    color = "red" if score >= 0.6 else "orange" if score >= 0.4 else "green"
                    st.markdown(f":{color}[{side.title()}: {level} ({score:.0%})]")
                    if bp.get("note"):
                        st.caption(bp["note"])

            # Pythagorean luck
            with col2:
                st.markdown("**🍀 Pythagorean Luck**")
                for side in ("home", "away"):
                    py = signals.get("pythag", {}).get(side, {})
                    luck = py.get("luck_score", 0)
                    label = py.get("label", "neutral")
                    color = "red" if "lucky" in label else "green" if "unlucky" in label else "gray"
                    team = py.get("team", side)
                    st.markdown(f":{color}[{team}: {luck:+.1f} wins ({label})]")

            # Umpire
            ump = signals.get("umpire", {})
            if ump.get("name") and ump["name"] != "Unknown":
                lean = ump.get("lean", "NEUTRAL")
                color = "green" if lean == "OVER" else "red" if lean == "UNDER" else "gray"
                st.markdown(f"**⚖️ Umpire:** {ump['name']} — :{color}[{lean} lean ({ump.get('over_rate',0.5):.0%} career over rate)]")
                st.caption(ump.get("note", ""))

            # Travel
            travel = signals.get("travel", {})
            for side in ("home", "away"):
                tv = travel.get(side, {})
                if tv.get("travel_flag"):
                    st.markdown(f"**✈️ Travel:** :red[{tv['note']}]")

            # Opener
            opener = signals.get("opener", {})
            for flag in opener.get("flags", []):
                st.markdown(f"**🔄 Opener:** :orange[{flag}]")

    # xwOBA team luck table
    if xwoba_luck:
        st.markdown("---")
        st.markdown("### 🎯 Team xwOBA Luck Scores")
        st.caption("Teams hitting above their expected wOBA are 'lucky' and due for regression. Teams below are buying low opportunities.")
        rows = []
        for team, data in sorted(xwoba_luck.items(), key=lambda x: x[1].get("gap", 0), reverse=True):
            if data.get("woba", 0) > 0:
                rows.append({
                    "Team":   team,
                    "wOBA":   f"{data.get('woba',0):.3f}",
                    "xwOBA":  f"{data.get('xwoba',0):.3f}",
                    "Gap":    f"{data.get('gap',0):+.3f}",
                    "Status": data.get("label","?").upper(),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Sharp money signals
    if sharp_plays:
        st.markdown("---")
        st.markdown("### 🔍 Sharp Money Signals")
        for sp in sharp_plays[:8]:
            strength = sp.get("signal_strength", 0)
            color = "red" if strength >= 0.7 else "orange" if strength >= 0.4 else "gray"
            st.markdown(
                f"**{sp.get('signal_type','?')}** — "
                f":{color}[Strength: {strength:.0%}]  \n"
                f"_{sp.get('notes','')}_"
            )


def _build_hr_parlays_direct(hr_results: dict) -> list[dict]:
    """
    Build 3 HR parlays directly from candidates. Always returns 3 parlays —
    no EV or factor threshold, just the model's best available picks.
    Labeled ⚠️ when below model confidence thresholds.
    """
    import math

    # Pull candidates from results; fall back to KNOWN_HR_HITTERS if empty
    candidates = (hr_results or {}).get("candidates", [])
    if not candidates:
        try:
            from sports_betting.collectors.hr_props_collector import (
                KNOWN_HR_HITTERS, _mock_hr_odds,
            )
            mock_odds = _mock_hr_odds()
            for prop in mock_odds:
                batter = prop["batter"]
                lower  = batter.lower()
                last   = batter.split()[-1].lower()
                known  = KNOWN_HR_HITTERS.get(lower) or KNOWN_HR_HITTERS.get(last) or KNOWN_HR_HITTERS["default"]
                candidates.append({
                    "batter":        batter,
                    "game":          prop["game"],
                    "home_team":     prop["home_team"],
                    "price":         prop["price"],
                    "book":          prop.get("book", "draftkings"),
                    "hr_rate":       known["hr_rate"],
                    "barrel_rate":   known["barrel_rate"],
                    "hard_hit_rate": known["hard_hit_rate"],
                    "composite_prob": round(0.10 + known["hr_rate"] * 1.2, 3),
                    "ev_pct":        0.0,
                    "factors_passed": 2,
                    "recommendation": "BEST AVAILABLE",
                })
        except Exception:
            pass

    if not candidates:
        return []

    def a2d(p):
        return (p / 100 + 1) if p > 0 else (100 / abs(p) + 1)

    def d2a(d):
        if d >= 2.0:
            return int((d - 1) * 100)
        return int(-100 / (d - 1))

    def make_parlay(legs, label, stake):
        dec  = math.prod(a2d(l["price"]) for l in legs)
        prob = math.prod(l["composite_prob"] for l in legs)
        ev   = (prob * dec) - 1
        high_conf = all(l.get("factors_passed", 0) >= 3 for l in legs)
        return {
            "label":          label,
            "legs":           legs,
            "combined_odds":  d2a(dec),
            "combined_prob":  round(prob, 4),
            "ev_pct":         round(ev, 4),
            "stake_rec":      stake,
            "starred":        high_conf and ev > 0,
            "warning":        not high_conf or ev <= 0,
        }

    # Best 3 by composite_prob (1 per game where possible)
    seen, top3 = set(), []
    for c in sorted(candidates, key=lambda x: x["composite_prob"], reverse=True):
        if c["game"] not in seen or len(top3) < 3:
            top3.append(c)
            seen.add(c["game"])
        if len(top3) == 3:
            break
    while len(top3) < 3 and len(candidates) >= 3:
        top3 = candidates[:3]
        break

    # Best 3 by price (highest +money = moonshot)
    price3 = sorted(candidates, key=lambda x: x["price"], reverse=True)[:3]

    # Best 3 by EV
    ev3 = sorted(candidates, key=lambda x: x["ev_pct"], reverse=True)[:3]

    parlays = []
    if len(top3) == 3:
        parlays.append(make_parlay(top3, "HR Parlay A — Best 3", "$5–$10"))
    if len(ev3) == 3:
        parlays.append(make_parlay(ev3,  "HR Parlay B — Best EV", "$5"))
    if len(price3) == 3:
        parlays.append(make_parlay(price3, "HR Parlay C — Moonshot", "$3–$5"))

    return parlays


def render_hr_parlay_tab(hr_results: dict):
    st.markdown("### 💣 Home Run Parlay")
    st.caption("Best 3-leg HR parlays — always shown daily. HR props are high-variance by nature; see warning labels.")

    parlays   = _build_hr_parlays_direct(hr_results)
    candidates = (hr_results or {}).get("candidates", [])

    if not parlays:
        st.warning("Could not build HR parlays today. Model will retry on next refresh.")
        return

    for parlay in parlays:
        odds     = parlay["combined_odds"]
        odds_str = f"+{odds:,}" if odds > 0 else f"{odds:,}"
        ev       = parlay["ev_pct"]
        star     = "⭐ " if parlay.get("starred") else ""
        warn     = "⚠️ " if parlay.get("warning") else ""

        st.markdown(f"### 🎯 {star}{warn}{parlay['label']}")

        if parlay.get("warning"):
            st.caption("⚠️ Below model confidence threshold — treat as lottery-ticket stake only ($3–$5 max)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Odds",     odds_str)
        c2.metric("Win Prob", f"{parlay['combined_prob']:.1%}")
        c3.metric("EV",       f"{ev:+.1%}")
        c4.metric("Stake",    parlay["stake_rec"])

        leg_rows = []
        for j, leg in enumerate(parlay["legs"], 1):
            lp = leg.get("price", 0)
            leg_rows.append({
                "Leg":        j,
                "Batter":     leg.get("batter", "?"),
                "Game":       leg.get("game", "?"),
                "Price":      f"{'+' if lp > 0 else ''}{lp}",
                "HR Prob":    f"{leg.get('composite_prob', 0):.1%}",
                "HR Rate":    f"{leg.get('hr_rate', 0):.1%}",
                "Barrel %":   f"{leg.get('barrel_rate', 0):.1%}",
                "Hard Hit":   f"{leg.get('hard_hit_rate', 0):.1%}",
                "Book":       leg.get("book", "DK"),
            })
        st.dataframe(pd.DataFrame(leg_rows), hide_index=True, use_container_width=True)
        st.markdown("---")

    # Ranked candidates table
    if candidates:
        with st.expander(f"📋 All HR Candidates Ranked ({len(candidates)} batters scored)"):
            c_rows = []
            for i, c in enumerate(candidates[:20], 1):
                flag = "⭐" if c.get("factors_passed", 0) >= 4 else ("✅" if c.get("factors_passed", 0) >= 3 else "")
                c_rows.append({
                    "":       flag,
                    "Batter": c.get("batter", "?"),
                    "Game":   c.get("game", "?"),
                    "Price":  f"{c.get('price', 0):+d}",
                    "HR Prob":f"{c.get('composite_prob', 0):.1%}",
                    "EV":     f"{c.get('ev_pct', 0):+.1%}",
                    "Factors":c.get("factors_passed", 0),
                    "Rec":    c.get("recommendation", "?"),
                })
            st.dataframe(pd.DataFrame(c_rows), hide_index=True, use_container_width=True)

    st.markdown("""
    **HR Factor Scoring**
    - **F1** Batter HR rate this season (HRs/game)  ·  **F2** Pitcher HR vulnerability
    - **F3** Park HR factor  ·  **F4** Weather (wind out / hot temps)  ·  **F5** Handedness split

    ⭐ = 4+ factors confirmed, positive EV  ·  ⚠️ = model best-available pick, no EV threshold met
    Max $5–$10 stake on any HR parlay — these are lottery tickets by design
    """)


def render_skipped_tab(skipped: list):
    if not skipped:
        st.success("No games skipped today.")
        return

    st.markdown(f"**{len(skipped)} games eliminated by the 12-layer filter:**")
    for s in skipped:
        st.markdown(f"- **{s.get('game','?')}** — {s.get('reason','?')}")


def render_signals_tab():
    report = cached_signals()
    if not report:
        st.info("Signal data builds up as bets are graded. Check back after the first week of picks.")
        return

    rows = []
    for s in report:
        hr = s["hit_rate"]
        w  = s["current_weight"]
        rows.append({
            "Signal":   s["signal"],
            "Bets":     s["bets"],
            "Wins":     s["wins"],
            "Hit Rate": hr,
            "Weight":   w,
            "Status":   "↑ Boosted" if w > 1.05 else "↓ Penalized" if w < 0.95 else "→ Neutral",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Hit Rate": st.column_config.ProgressColumn(
                "Hit Rate", min_value=0, max_value=1, format="%.1%"
            ),
            "Weight": st.column_config.NumberColumn("Weight", format="%.3f"),
        }
    )
    st.caption("Weights auto-adjust after every game cycle. Signals hitting >60% get boosted.")


def render_record_bet_tab(picks: list = None, parlays: list = None, nrfi_ranked: list = None, hr_results: dict = None):
    st.markdown("### 📝 Record a Bet You Placed")
    st.caption("Logging your actual bets teaches the model which signals work best over time.")

    picks       = picks or []
    parlays     = parlays or []
    nrfi_ranked = nrfi_ranked or []
    hr_results  = hr_results or {}

    # Build game options from today's active picks
    active_picks = [p for p in picks if p.tier != "SKIP"]
    game_options = [
        f"{_short(p.away_team)} @ {_short(p.home_team)}"
        for p in active_picks
    ]
    # Add NRFI games not already in picks
    nrfi_games = [g.get("game", "") for g in nrfi_ranked if g.get("game") not in game_options]
    game_options = list(dict.fromkeys(game_options + nrfi_games))  # dedupe, preserve order
    if not game_options:
        game_options = ["No games loaded — run model first"]

    # Map game → pick for auto-fill
    pick_by_game = {
        f"{_short(p.away_team)} @ {_short(p.home_team)}": p
        for p in active_picks
    }

    from sports_betting.analysis.signal_tracker import record_placed_bet

    # ── Bet type selector ──────────────────────────────────────────────
    bet_type = st.radio(
        "What are you recording?",
        ["🎯 Single Bet", "🎰 Parlay", "💣 HR Parlay"],
        horizontal=True,
    )
    st.markdown("---")

    # ── SINGLE BET ─────────────────────────────────────────────────────
    if bet_type == "🎯 Single Bet":
        with st.form("record_single_bet", clear_on_submit=True):
            st.markdown("#### Single Bet Details")

            col1, col2 = st.columns(2)
            with col1:
                selected_game = st.selectbox("Game", game_options)

                # Auto-detect teams from game string
                pick = pick_by_game.get(selected_game)
                if pick:
                    team_opts = [
                        f"{_short(pick.backing_team)} (Model Pick ⭐)",
                        _short(pick.home_team) if pick.backing_team != pick.home_team else _short(pick.away_team),
                        "Over",
                        "Under",
                    ]
                elif " @ " in selected_game:
                    parts = selected_game.split(" @ ")
                    team_opts = [parts[1].strip(), parts[0].strip(), "Over", "Under"]
                else:
                    team_opts = ["Home", "Away", "Over", "Under"]

                side = st.selectbox("Bet On", team_opts)
                # Strip the model tag if present
                side_clean = side.replace(" (Model Pick ⭐)", "").strip()

                market = st.selectbox("Market", [
                    "Full Game ML", "F5 ML",
                    "Run Line -1.5", "Run Line +1.5",
                    "NRFI", "YRFI",
                    "Game Over", "Game Under",
                    "F5 Over", "F5 Under",
                    "K Over", "K Under",
                    "Outs Recorded Over", "Outs Recorded Under",
                    "Earned Runs Under", "Earned Runs Over",
                    "Hits Over", "Hits Under",
                    "Total Bases Over", "Total Bases Under",
                    "HR (batter prop)",
                    "RBI Over", "RBI Under",
                    "First Inning Over", "First Inning Under",
                ])

            with col2:
                # Auto-suggest price from model pick
                default_price = -130
                if pick:
                    default_price = getattr(pick, "backing_price", -130) or -130
                price = st.number_input("DraftKings Price (American odds)", value=int(default_price), step=5)

                units = st.selectbox("Units", [0.5, 1, 1.5, 2, 3], index=1)
                dollar_amount = units * UNIT_SIZE
                st.caption(f"💵 Dollar amount: ${dollar_amount:.2f}")

                book = st.selectbox("Book", ["DraftKings", "FanDuel", "BetMGM", "Caesars", "Other"])

            # Auto-populate signals from model pick
            default_signals = ""
            if pick and pick.factors:
                default_signals = ", ".join(pick.factors[:3])
            signals = st.text_area(
                "Signals / Reason (auto-filled from model — edit if needed)",
                value=default_signals,
                height=80,
            )

            submitted = st.form_submit_button("✅ Record Single Bet", type="primary", use_container_width=True)
            if submitted and selected_game != "No games loaded — run model first":
                signal_list = [s.strip() for s in signals.split(",")] if signals else []
                game_id = selected_game.replace(" @ ", "-").replace(" ", "_")
                record_placed_bet(
                    game_id=game_id,
                    book=book.lower().replace(" ", ""),
                    market=market.lower().replace(" ", "_"),
                    side=side_clean.lower(),
                    price=int(price),
                    units=float(units),
                    signals_present=signal_list,
                )
                st.success(f"✅ Recorded: **{side_clean}** {market} @ {int(price):+d}  ·  {units}u / ${dollar_amount:.2f}")
                st.caption("The model will grade this automatically after the game completes and update signal weights.")

    # ── HR PARLAY RECORDER ─────────────────────────────────────────────
    elif bet_type == "💣 HR Parlay":
        st.markdown("#### Record an HR Parlay")
        st.caption("Select one of today's model-built HR parlays to record, or enter manually.")

        # Get HR parlays — use same direct builder as the tab so they always match
        hr_parlays = _build_hr_parlays_direct(hr_results)

        if hr_parlays:
            parlay_labels = [p.get("label", f"HR Parlay {i+1}") for i, p in enumerate(hr_parlays)]
            selected_hr_label = st.selectbox("Select HR Parlay", parlay_labels)
            selected_hr = next((p for p in hr_parlays if p.get("label") == selected_hr_label), hr_parlays[0])

            # Show parlay details
            legs = selected_hr.get("legs", [])
            odds = selected_hr.get("combined_odds", 0)
            odds_str = f"+{odds:,}" if odds > 0 else f"{odds:,}"
            prob = selected_hr.get("combined_prob", 0)
            ev = selected_hr.get("ev_pct", 0)

            c1, c2, c3 = st.columns(3)
            c1.metric("Odds", odds_str)
            c2.metric("Win Prob", f"{prob:.1%}")
            c3.metric("EV", f"{ev:+.1%}")

            if legs:
                st.markdown("**Legs:**")
                for j, leg in enumerate(legs, 1):
                    lp = leg.get("price", 0)
                    st.markdown(
                        f"  **{j}.** {leg.get('batter','?')} — {leg.get('game','?')}  "
                        f"| Price: {'+' if lp > 0 else ''}{lp}  "
                        f"| HR Prob: {leg.get('composite_prob',0):.1%}"
                    )
        else:
            st.info("No HR parlays loaded. Run the model first or they'll be shown here automatically.")
            selected_hr = None
            legs = []
            odds_str = "+0"
            odds = 0

        with st.form("record_hr_parlay", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                hr_stake = st.number_input("Stake ($)", value=5, min_value=1, step=1)
                if odds > 0:
                    potential = round(odds / 100 * hr_stake, 2)
                elif odds < 0:
                    potential = round(100 / abs(odds) * hr_stake, 2)
                else:
                    potential = 0.0
                st.caption(f"💵 Potential win: ${potential:,.2f}  (total return ${potential + hr_stake:,.2f})")
            with col2:
                hr_book = st.selectbox("Book", ["DraftKings", "FanDuel", "BetMGM", "Caesars", "Other"])
                hr_note = st.text_input("Note (optional)", placeholder="e.g. HR Parlay A — Judge/Alvarez/Schwarber")

            hr_submitted = st.form_submit_button("✅ Record HR Parlay", type="primary", use_container_width=True)
            if hr_submitted:
                parlay_id = f"hr_parlay_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
                for j, leg in enumerate(legs, 1):
                    batter = leg.get("batter", f"batter_{j}").lower().replace(" ", "_")
                    game_id = f"{parlay_id}_leg{j}_{batter}"
                    record_placed_bet(
                        game_id=game_id,
                        book=hr_book.lower().replace(" ", ""),
                        market="hr_prop",
                        side=batter,
                        price=leg.get("price", -130),
                        units=round(hr_stake / UNIT_SIZE, 2),
                        signals_present=[
                            f"hr_parlay:{parlay_id}",
                            f"leg:{j}of{len(legs)}",
                            f"barrel:{leg.get('barrel_rate',0):.1%}",
                        ],
                    )
                st.success(
                    f"✅ Recorded HR parlay ({len(legs)} legs)  ·  "
                    f"Odds: {odds_str}  ·  Stake: ${hr_stake}  ·  To win: ${potential:,.2f}"
                )
                if hr_note:
                    st.caption(f"Note: {hr_note}")

    # ── PARLAY BUILDER ─────────────────────────────────────────────────
    else:
        st.markdown("#### Build Your Parlay")
        st.caption("Add each leg of your parlay below. Combined odds update automatically.")

        # Let user choose how many legs
        num_legs = st.selectbox("Number of Legs", [2, 3, 4, 5, 6], index=0)

        legs_data = []
        combined_decimal = 1.0
        all_valid = True

        for i in range(num_legs):
            st.markdown(f"**Leg {i+1}**")
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            with c1:
                game = st.selectbox(f"Game", game_options, key=f"pg_{i}")
            with c2:
                pick = pick_by_game.get(game)
                if pick:
                    t_opts = [
                        f"{_short(pick.backing_team)} ⭐",
                        _short(pick.home_team) if pick.backing_team != pick.home_team else _short(pick.away_team),
                        "Over", "Under",
                    ]
                elif " @ " in game:
                    parts = game.split(" @ ")
                    t_opts = [parts[1].strip(), parts[0].strip(), "Over", "Under"]
                else:
                    t_opts = ["Home", "Away", "Over", "Under"]
                side = st.selectbox("Side", t_opts, key=f"ps_{i}")
            with c3:
                mkt = st.selectbox("Market", [
                    "Full Game ML", "F5 ML", "Run Line -1.5", "Run Line +1.5",
                    "NRFI", "YRFI",
                    "Game Over", "Game Under", "F5 Over", "F5 Under",
                    "K Over", "K Under",
                    "Outs Recorded Over",
                    "Total Bases Over", "HR (batter prop)",
                ], key=f"pm_{i}")
            with c4:
                default_p = -130
                if pick:
                    default_p = getattr(pick, "backing_price", -130) or -130
                leg_price = st.number_input("Price", value=int(default_p), step=5, key=f"pp_{i}")

            # Running combined odds
            if leg_price > 0:
                dec = leg_price / 100 + 1
            else:
                dec = 100 / abs(leg_price) + 1
            combined_decimal *= dec
            legs_data.append({
                "game": game, "side": side.replace(" ⭐","").strip(),
                "market": mkt, "price": int(leg_price),
            })

        # Show combined odds live
        st.markdown("---")
        if combined_decimal >= 2.0:
            combined_american = int((combined_decimal - 1) * 100)
        else:
            combined_american = int(-100 / (combined_decimal - 1)) if combined_decimal > 1 else -9999

        c1, c2, c3 = st.columns(3)
        c1.metric("Combined Odds", f"+{combined_american:,}" if combined_american > 0 else f"{combined_american:,}")

        with st.form("record_parlay_bet", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                parlay_stake = st.number_input("Stake ($)", value=10, min_value=1, step=5)
                potential_win = round((combined_decimal - 1) * parlay_stake, 2)
                st.caption(f"💵 Potential win: ${potential_win:,.2f} (total return ${potential_win + parlay_stake:,.2f})")
            with col2:
                parlay_book = st.selectbox("Book", ["DraftKings", "FanDuel", "BetMGM", "Caesars", "Other"])
                parlay_label = st.text_input("Parlay name / note (optional)", placeholder="e.g. P2 Core 3-leg")

            submitted_parlay = st.form_submit_button("✅ Record Parlay", type="primary", use_container_width=True)
            if submitted_parlay:
                # Record each leg as a linked parlay bet
                parlay_id = f"parlay_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
                for j, leg in enumerate(legs_data):
                    game_id = leg["game"].replace(" @ ","-").replace(" ","_")
                    record_placed_bet(
                        game_id=f"{parlay_id}_leg{j+1}_{game_id}",
                        book=parlay_book.lower().replace(" ",""),
                        market=leg["market"].lower().replace(" ","_"),
                        side=leg["side"].lower(),
                        price=leg["price"],
                        units=round(parlay_stake / UNIT_SIZE, 2),
                        signals_present=[f"parlay:{parlay_id}", f"leg:{j+1}of{num_legs}"],
                    )
                st.success(
                    f"✅ Recorded {num_legs}-leg parlay  ·  "
                    f"Odds: {'+' if combined_american > 0 else ''}{combined_american:,}  ·  "
                    f"Stake: ${parlay_stake}  ·  "
                    f"To win: ${potential_win:,.2f}"
                )
                if parlay_label:
                    st.caption(f"Label: {parlay_label}")

    # ── Recent bets history ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Recent Recorded Bets")
    try:
        from sports_betting.database import get_db
        with get_db() as conn:
            rows = conn.execute("""
                SELECT game_id, market, side, book_price, recommended_bet, result, detected_at
                FROM value_bets
                WHERE confidence = 'PLACED'
                ORDER BY detected_at DESC
                LIMIT 15
            """).fetchall()
        if rows:
            hist = []
            for r in rows:
                result_str = r[5] or "Pending"
                icon = "🟢" if result_str == "WIN" else "🔴" if result_str == "LOSS" else "⏳"
                units_val = round((r[4] or 0) / UNIT_SIZE, 1)
                hist.append({
                    "":       icon,
                    "Game":   str(r[0])[:28],
                    "Market": r[1] or "?",
                    "Side":   str(r[2]).upper() if r[2] else "?",
                    "Price":  f"{int(r[3]):+d}" if r[3] else "?",
                    "Units":  units_val,
                    "Stake":  f"${r[4]:.2f}" if r[4] else "?",
                    "Result": result_str,
                    "Date":   str(r[6])[:10] if r[6] else "?",
                })
            st.dataframe(pd.DataFrame(hist), hide_index=True, use_container_width=True)
        else:
            st.caption("No bets recorded yet — your history appears here once you submit a bet above.")
    except Exception as e:
        st.caption(f"Bet history unavailable: {e}")


# ── App entry point ───────────────────────────────────────────────────

def main():
    get_db_connection()

    date_str, run_btn = render_sidebar()

    st.title("⚾ MLB Betting Model v4.0")
    st.caption(f"12-Layer Framework  ·  Built from 24 days live data  ·  Platform: DraftKings  ·  Unit: $5")

    # ROI header
    roi = cached_roi()
    render_header(roi)
    st.markdown("---")

    # Run model
    if run_btn:
        st.cache_data.clear()

    with st.spinner("Running 12-layer analysis... (~30 seconds)"):
        try:
            results = cached_run_model(date_str)
        except Exception as e:
            st.error(f"Model error: {e}")
            st.info("Check that your API keys are set in the Streamlit secrets panel.")
            return

    picks        = results.get("picks", [])
    parlays      = results.get("parlays", [])
    nrfi_parlay  = results.get("nrfi_parlay")
    nrfi_ranked  = results.get("nrfi_ranked", [])
    skipped      = results.get("skipped", [])
    sharp_plays  = results.get("sharp_plays", [])
    hr_results   = results.get("hr_results", {})
    all_signals  = results.get("all_signals", {})
    xwoba_luck   = results.get("xwoba_luck", {})

    active_count = len([p for p in picks if p.tier != "SKIP"])

    # Tabs
    tabs = st.tabs([
        f"🎯 Picks ({active_count})",
        f"🎰 Parlays ({len(parlays)})",
        f"🚫 NRFI/YRFI ({len(nrfi_ranked)})",
        "💣 HR Parlays",
        "📡 Intelligence",
        f"⏭️ Skipped ({len(skipped)})",
        "📊 Signal Performance",
        "📝 Record a Bet",
    ])

    with tabs[0]: render_picks_tab(picks)
    with tabs[1]: render_parlays_tab(parlays, nrfi_parlay)
    with tabs[2]: render_nrfi_tab(nrfi_ranked, nrfi_parlay)
    with tabs[3]: render_hr_parlay_tab(hr_results)
    with tabs[4]: render_intelligence_tab(all_signals, xwoba_luck, picks, sharp_plays)
    with tabs[5]: render_skipped_tab(skipped)
    with tabs[6]: render_signals_tab()
    with tabs[7]: render_record_bet_tab(picks, parlays, nrfi_ranked, hr_results)

    # Footer
    st.markdown("---")
    st.caption(
        "⚠️ For research purposes only. Sports betting involves financial risk. "
        "Always bet responsibly and within legal jurisdictions.  \n"
        f"Last run: {datetime.now().strftime('%I:%M %p')}  ·  "
        "Refreshes when you press Run Model"
    )


if __name__ == "__main__":
    main()
