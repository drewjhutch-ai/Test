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
        "❌ No totals in parlays\n\n"
        "❌ No LAA in parlays\n\n"
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


def render_hr_parlay_tab(hr_results: dict):
    st.markdown("### 💣 Home Run Parlay")
    st.caption("3-leg HR parlays built from batter barrel rate, park factor, pitcher vulnerability, weather, and odds value.")

    if not hr_results or not hr_results.get("candidates"):
        st.info(hr_results.get("data_note", "No HR prop data available. Add ODDS_API_KEY in Streamlit secrets to enable live odds."))
        st.markdown("""
        **How this works once enabled:**
        - Pulls live HR prop odds from DraftKings via The Odds API
        - Scores each batter on 5 factors: HR rate, pitcher vulnerability, park factor, weather, handedness splits
        - Builds 3 different 3-leg parlays: Best overall · Best EV · Moonshot (highest odds)
        - ⭐ = model's highest confidence selection
        """)
        return

    st.caption(hr_results.get("data_note", ""))

    # Parlays first
    for parlay in hr_results.get("parlays", []):
        star = "⭐ " if parlay.get("starred") else ""
        odds = parlay.get("combined_odds", 0)
        odds_str = f"+{odds:,}" if odds > 0 else f"{odds:,}"
        ev = parlay.get("ev_pct", 0)

        st.markdown(f"### 🎯 {star}{parlay.get('label','Parlay')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Odds",     odds_str)
        c2.metric("Win Prob", f"{parlay.get('combined_prob',0):.1%}")
        c3.metric("EV",       f"{ev:+.1%}")
        c4.metric("Stake",    parlay.get("stake_rec", "$5"))

        leg_rows = []
        for j, leg in enumerate(parlay.get("legs", []), 1):
            leg_rows.append({
                "Leg":      j,
                "Batter":   leg.get("batter", "?"),
                "Game":     leg.get("game", "?"),
                "Price":    f"{leg.get('price',0):+d}",
                "HR Prob":  f"{leg.get('composite_prob',0):.1%}",
                "Barrel %": f"{leg.get('barrel_rate',0):.1%}",
                "Hard Hit": f"{leg.get('hard_hit_rate',0):.1%}",
                "Book":     leg.get("book", "DK"),
            })
        st.dataframe(pd.DataFrame(leg_rows), hide_index=True, use_container_width=True)
        st.markdown("---")

    # Top HR candidates table
    candidates = hr_results.get("candidates", [])
    if candidates:
        with st.expander(f"📋 All HR Candidates Ranked ({len(candidates)} batters scored)"):
            c_rows = []
            for i, c in enumerate(candidates[:20], 1):
                star_c = "⭐" if c.get("starred") else ""
                c_rows.append({
                    "":         star_c,
                    "Batter":   c.get("batter","?"),
                    "Game":     c.get("game","?"),
                    "Price":    f"{c.get('price',0):+d}",
                    "HR Prob":  f"{c.get('composite_prob',0):.1%}",
                    "EV":       f"{c.get('ev_pct',0):+.1%}",
                    "Factors":  c.get("factors_passed",0),
                    "Rec":      c.get("recommendation","?"),
                })
            st.dataframe(pd.DataFrame(c_rows), hide_index=True, use_container_width=True)

    st.markdown("""
    **HR Factor Scoring**
    - **F1** Batter HR rate this season (HRs/game)
    - **F2** Pitcher HR vulnerability (HR/9, barrel rate, HR/FB rate)
    - **F3** Park HR factor (Coors, Yankee Stadium, etc.)
    - **F4** Weather (wind out, temperature)
    - **F5** Handedness wOBA split (batter vs pitcher hand)

    ⭐ = 4+ factors confirmed · Stake max $10 on any single HR parlay · HR props are lottery tickets by nature
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


def render_record_bet_tab():
    st.markdown("### 📝 Record a Bet You Placed")
    st.markdown("Logging your actual bets teaches the model which signals work best for you.")

    with st.form("record_bet"):
        col1, col2 = st.columns(2)
        with col1:
            game_id  = st.text_input("Game (e.g. NYY-BOS or mlb_12345)")
            market   = st.selectbox("Market", ["ml","f5_ml","k_over","er_under","nrfi","yrfi","run_line"])
            price    = st.number_input("Price (American odds)", value=-130, step=5)
        with col2:
            side     = st.selectbox("Side", ["home","away","over","under"])
            units    = st.selectbox("Units", [1, 2, 3])
            signals  = st.text_input("Signals (comma-separated)", placeholder="era fraud, hot streak")

        submitted = st.form_submit_button("✅ Record Bet", type="primary")
        if submitted and game_id:
            from sports_betting.analysis.signal_tracker import record_placed_bet
            signal_list = [s.strip() for s in signals.split(",")] if signals else []
            record_placed_bet(
                game_id=game_id, book="draftkings",
                market=market, side=side,
                price=int(price), units=float(units),
                signals_present=signal_list,
            )
            st.success(f"✅ Recorded: {side.upper()} {market} @ {int(price):+d} ({units}u / ${int(units)*UNIT_SIZE})")
            st.caption("Result will be graded automatically after the game completes.")


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

    active_count = len([p for p in picks if p.tier != "SKIP"])

    # Tabs
    tabs = st.tabs([
        f"🎯 Picks ({active_count})",
        f"🎰 Parlays ({len(parlays)})",
        f"🚫 NRFI/YRFI ({len(nrfi_ranked)})",
        "💣 HR Parlays",
        f"⏭️ Skipped ({len(skipped)})",
        "📊 Signal Performance",
        "📝 Record a Bet",
    ])

    with tabs[0]: render_picks_tab(picks)
    with tabs[1]: render_parlays_tab(parlays, nrfi_parlay)
    with tabs[2]: render_nrfi_tab(nrfi_ranked, nrfi_parlay)
    with tabs[3]: render_hr_parlay_tab(hr_results)
    with tabs[4]: render_skipped_tab(skipped)
    with tabs[5]: render_signals_tab()
    with tabs[6]: render_record_bet_tab()

    # Sharp money alerts
    if sharp_plays:
        with st.expander(f"🔍 Sharp Money Signals ({len(sharp_plays)})"):
            for sp in sharp_plays[:8]:
                st.markdown(
                    f"**{sp.get('signal_type','?')}** — "
                    f"Strength: {sp.get('signal_strength',0):.0%}  \n"
                    f"_{sp.get('notes','')}_"
                )

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
