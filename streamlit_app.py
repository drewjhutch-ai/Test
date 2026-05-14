"""
MLB Betting Model v4.0 — Web Dashboard
Runs on Streamlit Cloud (free). Works on phone, tablet, any browser.
"""
import os
import sys
import json
import html as html_lib
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

@st.cache_data(ttl=60)
def cached_model_roi():
    """ROI computed only from MODEL_PICK / MODEL_PARLAY rows."""
    from sports_betting.database import get_db
    try:
        with get_db() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                                              AS total,
                    SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END)       AS wins,
                    SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END)       AS losses,
                    SUM(recommended_bet)                                  AS staked,
                    SUM(profit_loss)                                      AS profit,
                    AVG(CASE WHEN result='WIN' THEN 1.0 ELSE 0.0 END)    AS hit_rate
                FROM value_bets
                WHERE result IS NOT NULL
                  AND confidence IN ('MODEL_PICK', 'MODEL_PARLAY')
            """).fetchone()
        if not row or not row["total"]:
            return {}
        staked = row["staked"] or 1
        profit = row["profit"] or 0
        return {
            "total": row["total"], "wins": row["wins"], "losses": row["losses"],
            "hit_rate": round(row["hit_rate"] or 0, 4),
            "roi": round(profit / staked * 100, 2),
            "profit": round(profit, 2),
        }
    except Exception:
        return {}

@st.cache_data(ttl=300)
def cached_signals():
    from sports_betting.analysis.signal_tracker import get_signal_performance_report
    return get_signal_performance_report()


# ── Helpers ───────────────────────────────────────────────────────────

TIER_EMOJI  = {"STRONG": "🔥", "MEDIUM": "✅", "LEAN": "📌", "SKIP": "❌"}
TIER_COLOR  = {"STRONG": "#ef4444", "MEDIUM": "#f59e0b", "LEAN": "#94a3b8", "SKIP": "#334155"}
UNIT_MAP    = {"STRONG": 3, "MEDIUM": 2, "LEAN": 1}
UNIT_SIZE   = 5


def _format_market(raw: str, pick=None) -> str:
    """
    Convert a raw market string into a clear, human-readable bet description.
    Optionally uses the pick object to add pitcher name for K props.
    """
    if not raw:
        return "Moneyline"
    r = raw.strip()

    # Pitcher K prop — most common ambiguous case
    # Format: "K Over prop (5.9)" or "k_over_5.9" etc.
    import re
    k_match = re.search(r'[Kk]\s*[Oo]ver\s*prop\s*\(?([\d.]+)\)?', r)
    if k_match or "k_over" in r.lower():
        line = k_match.group(1) if k_match else re.search(r'[\d.]+', r)
        line = line if isinstance(line, str) else (line.group() if line else "?")
        # Try to get the pitcher's name from the pick
        pitcher_name = ""
        if pick:
            backing = getattr(pick, "backing_team", "")
            bp = getattr(pick, "backing_pitcher", None) or getattr(pick, "home_pitcher", None)
            ap = getattr(pick, "away_pitcher", None)
            # Use the starter for the team we're backing
            home_team = getattr(pick, "home_team", "")
            if backing and home_team and backing.lower() in home_team.lower():
                sp = getattr(pick, "home_pitcher", None)
            else:
                sp = getattr(pick, "away_pitcher", None)
            if sp:
                pitcher_name = getattr(sp, "name", "") or ""
        if pitcher_name and pitcher_name.lower() not in ("tbd", "unknown", ""):
            return f"{pitcher_name} K Over {line} (Pitcher Strikeout Prop)"
        return f"Pitcher K Over {line} (Strikeout Prop)"

    # K under
    if re.search(r'[Kk]\s*[Uu]nder', r) or "k_under" in r.lower():
        line = re.search(r'[\d.]+', r)
        line = line.group() if line else "?"
        if pick:
            home_team = getattr(pick, "home_team", "")
            backing   = getattr(pick, "backing_team", "")
            sp = (getattr(pick, "home_pitcher", None) if backing and home_team and backing.lower() in home_team.lower()
                  else getattr(pick, "away_pitcher", None))
            pname = (getattr(sp, "name", "") or "") if sp else ""
            if pname and pname.lower() not in ("tbd", "unknown", ""):
                return f"{pname} K Under {line} (Pitcher Strikeout Prop)"
        return f"Pitcher K Under {line} (Strikeout Prop)"

    lower = r.lower()
    # Moneyline variants
    if lower in ("h2h", "full_game_ml", "moneyline", "ml"):
        return "Moneyline (Full Game)"
    if "f5" in lower and "ml" in lower:
        return "First 5 Innings ML"
    if "f5" in lower and "over" in lower:
        return "First 5 Innings Over"
    if "f5" in lower and "under" in lower:
        return "First 5 Innings Under"
    if "f5" in lower:
        return "First 5 Innings ML"
    # Run line
    if "run_line" in lower or lower == "rl":
        if "-1.5" in r:
            return "Run Line -1.5 (Favorite)"
        if "+1.5" in r:
            return "Run Line +1.5 (Underdog)"
        return "Run Line"
    # Totals
    if "nrfi" in lower:
        return "No Run First Inning (NRFI)"
    if "yrfi" in lower:
        return "Yes Run First Inning (YRFI)"
    if "game" in lower and "over" in lower:
        return "Game Total Over"
    if "game" in lower and "under" in lower:
        return "Game Total Under"
    # HR prop
    if "hr" in lower and "prop" in lower:
        return "Home Run Prop"
    # ERA fraud fade
    if "era_fraud" in lower or "fade" in lower:
        return "Fade Pitcher (ERA Fraud)"
    # Opponent ML
    if "opponent" in lower or "opp" in lower:
        return "Opponent Moneyline"
    # Outs recorded
    if "outs" in lower:
        return "Outs Recorded Prop"
    # Default: title-case the raw string and strip underscores
    return r.replace("_", " ").title()


def inject_css():
    st.markdown("""
    <style>
    /* ── Global ── */
    .stApp { background: #0a0e1a; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1440px; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e293b; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 { color: #f1f5f9 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827; border-radius: 10px; padding: 4px; gap: 2px; border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px; color: #64748b;
        font-weight: 500; font-size: 13px; padding: 8px 14px; transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important; color: #f59e0b !important; font-weight: 700;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 11px !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 22px !important; font-weight: 800; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: #0a0e1a; font-weight: 800; border: none; border-radius: 8px;
        padding: 10px 20px; letter-spacing: 0.03em; transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        box-shadow: 0 4px 16px rgba(245,158,11,0.35); transform: translateY(-1px);
    }

    /* ── Inputs ── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stTextInput > div > div > input {
        background: #1e293b !important; border-color: #334155 !important;
        color: #f1f5f9 !important; border-radius: 8px !important;
    }

    /* ── Forms ── */
    .stForm { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: #111827 !important; border: 1px solid #1e293b !important;
        border-radius: 8px !important; color: #94a3b8 !important; font-weight: 500;
    }
    .streamlit-expanderContent {
        background: #0d1117 !important; border: 1px solid #1e293b !important;
        border-top: none !important; border-radius: 0 0 8px 8px !important;
    }

    /* ── Typography ── */
    h1, h2, h3 { color: #f1f5f9 !important; font-weight: 800 !important; }
    p, .stMarkdown p { color: #94a3b8; }
    hr { border-color: #1e293b !important; margin: 1.5rem 0; }
    .stCaption, small { color: #475569 !important; font-size: 12px; }

    /* ── Radio ── */
    .stRadio > div { gap: 8px; }
    .stRadio label {
        background: #111827; border: 1px solid #1e293b; border-radius: 8px;
        padding: 8px 18px; color: #94a3b8 !important; cursor: pointer; transition: all 0.2s;
    }

    /* ── Alerts ── */
    .stSuccess { background: #052e16 !important; border-color: #10b981 !important; border-radius: 8px !important; }
    .stWarning { background: #1c1207 !important; border-color: #f59e0b !important; border-radius: 8px !important; }
    .stInfo    { background: #0c1a2e !important; border-color: #3b82f6 !important; border-radius: 8px !important; }
    .stError   { background: #1f0707 !important; border-color: #ef4444 !important; border-radius: 8px !important; }

    /* ── DataFrames ── */
    .stDataFrame { border: 1px solid #1e293b !important; border-radius: 10px !important; overflow: hidden; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


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
    st.sidebar.markdown("""
    <div style="text-align:center;padding:16px 0 8px">
        <div style="font-size:36px">⚾</div>
        <div style="color:#f59e0b;font-size:18px;font-weight:800;letter-spacing:0.05em">MLB MODEL</div>
        <div style="color:#475569;font-size:11px;letter-spacing:0.1em">v4.0 · 12-LAYER AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")

    date_input = st.sidebar.date_input(
        "Game Date",
        value=datetime.now().date(),
        min_value=datetime.now().date() - timedelta(days=1),
        max_value=datetime.now().date() + timedelta(days=3),
    )
    date_str = date_input.strftime("%Y-%m-%d")

    run_btn = st.sidebar.button("⚡ Run Model", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="color:#475569;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">Active Rules</div>
    """, unsafe_allow_html=True)
    rules = [
        ("✅", "Totals allowed · 3+ signals"),
        ("✅", "All DK markets active"),
        ("🚫", "No debut K props"),
        ("🚫", "Max 3 picks · team · 7d"),
    ]
    for icon, text in rules:
        st.sidebar.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:4px 0;color:#94a3b8;font-size:12px">
            <span>{icon}</span><span>{text}</span>
        </div>""", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="color:#475569;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">Unit Sizing · $5/unit</div>
    <div style="display:grid;gap:6px">
        <div style="background:#111827;border:1px solid #1e293b;border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between">
            <span style="color:#ef4444;font-size:13px">🔥 STRONG</span>
            <span style="color:#f1f5f9;font-size:13px;font-weight:700">3u · $15</span>
        </div>
        <div style="background:#111827;border:1px solid #1e293b;border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between">
            <span style="color:#f59e0b;font-size:13px">✅ MEDIUM</span>
            <span style="color:#f1f5f9;font-size:13px;font-weight:700">2u · $10</span>
        </div>
        <div style="background:#111827;border:1px solid #1e293b;border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between">
            <span style="color:#94a3b8;font-size:13px">📌 LEAN</span>
            <span style="color:#f1f5f9;font-size:13px;font-weight:700">1u · $5</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return date_str, run_btn


# ── Main content ──────────────────────────────────────────────────────

def render_header(roi: dict):
    wins   = roi.get("wins", 0)
    losses = roi.get("losses", 0)
    bets   = roi.get("total_bets", 0)
    hit    = roi.get("hit_rate", 0)
    r      = roi.get("roi", 0)
    roi_color = "#10b981" if r >= 0 else "#ef4444"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#111827,#0d1117);
                border:1px solid #1e293b;border-radius:16px;
                padding:24px 32px;margin-bottom:24px">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px">
            <div>
                <div style="color:#475569;font-size:11px;text-transform:uppercase;letter-spacing:0.1em">Season Record</div>
                <div style="color:#f1f5f9;font-size:32px;font-weight:800;line-height:1">{wins}–{losses}</div>
                <div style="color:#475569;font-size:12px;margin-top:2px">{bets} graded bets</div>
            </div>
            <div style="text-align:center">
                <div style="color:#475569;font-size:11px;text-transform:uppercase;letter-spacing:0.1em">Hit Rate</div>
                <div style="color:#f59e0b;font-size:32px;font-weight:800;line-height:1">{hit:.1%}</div>
            </div>
            <div style="text-align:center">
                <div style="color:#475569;font-size:11px;text-transform:uppercase;letter-spacing:0.1em">ROI</div>
                <div style="color:{roi_color};font-size:32px;font-weight:800;line-height:1">{r:+.1f}%</div>
            </div>
            <div style="text-align:right">
                <div style="color:#475569;font-size:11px;text-transform:uppercase;letter-spacing:0.1em">Model</div>
                <div style="color:#f1f5f9;font-size:14px;font-weight:700">12-Layer AI</div>
                <div style="color:#475569;font-size:12px">{datetime.now().strftime("%b %d, %Y")}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_model_header(mroi: dict):
    """Separate header tracking the model's own autonomous pick record."""
    if not mroi:
        st.html(
            '<div style="background:#0d1117;border:1px solid #1e293b;border-radius:12px;'
            'padding:14px 24px;margin-bottom:16px;display:flex;align-items:center;gap:12px">'
            '<span style="color:#8b5cf6;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em">🤖 AI Model Record</span>'
            '<span style="color:#334155;font-size:12px;margin-left:8px">No graded model picks yet — grades update automatically after each game ends</span>'
            '</div>'
        )
        return
    wins   = mroi.get("wins", 0)
    losses = mroi.get("losses", 0)
    total  = mroi.get("total", 0)
    hit    = mroi.get("hit_rate", 0)
    roi    = mroi.get("roi", 0)
    profit = mroi.get("profit", 0)
    roi_color    = "#10b981" if roi >= 0 else "#ef4444"
    profit_color = "#10b981" if profit >= 0 else "#ef4444"
    st.html(
        f'<div style="background:#0d1117;border:1px solid #1e293b;border-left:3px solid #8b5cf6;'
        f'border-radius:12px;padding:16px 28px;margin-bottom:16px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px">'
        f'<div>'
        f'<div style="color:#8b5cf6;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.12em">🤖 AI Model Record</div>'
        f'<div style="color:#f1f5f9;font-size:26px;font-weight:800;line-height:1.1;margin-top:2px">{wins}–{losses}</div>'
        f'<div style="color:#475569;font-size:11px;margin-top:2px">{total} graded picks</div>'
        f'</div>'
        f'<div style="text-align:center">'
        f'<div style="color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:0.1em">Hit Rate</div>'
        f'<div style="color:#f59e0b;font-size:24px;font-weight:800;line-height:1.1">{hit:.1%}</div>'
        f'</div>'
        f'<div style="text-align:center">'
        f'<div style="color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:0.1em">ROI</div>'
        f'<div style="color:{roi_color};font-size:24px;font-weight:800;line-height:1.1">{roi:+.1f}%</div>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<div style="color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:0.1em">Net P/L</div>'
        f'<div style="color:{profit_color};font-size:24px;font-weight:800;line-height:1.1">{profit:+.0f}u</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def render_picks_tab(picks: list):
    tier_order = {"STRONG": 0, "MEDIUM": 1, "LEAN": 2}
    active = sorted(
        [p for p in picks if p.tier != "SKIP"],
        key=lambda p: (tier_order.get(p.tier, 9), -(p.factor_count or 0)),
    )

    if not active:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:12px;
                    padding:40px;text-align:center">
            <div style="font-size:48px;margin-bottom:12px">⏳</div>
            <div style="color:#f1f5f9;font-size:18px;font-weight:600">No picks today</div>
            <div style="color:#475569;font-size:14px;margin-top:8px">Pitchers may be TBD or no statistical edge found. Try again after noon ET.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # POTD banner
    potd = next((p for p in active if p.tier == "STRONG"), active[0])
    units = UNIT_MAP.get(potd.tier, 1)
    mkt = html_lib.escape(_format_market(potd.recommended_market or potd.proposed_market or "", potd))
    factors_html = "  ·  ".join(html_lib.escape(str(f)) for f in potd.factors[:3])

    tier_badge_color = {"STRONG": "#ef4444", "MEDIUM": "#f59e0b", "LEAN": "#94a3b8"}.get(potd.tier, "#94a3b8")

    st.html(f"""
    <div style="background:linear-gradient(135deg,#111827 0%,#1a1208 100%);border:1px solid #f59e0b44;border-radius:16px;padding:28px 32px;margin-bottom:24px;position:relative;overflow:hidden">
        <div style="position:absolute;top:0;right:0;width:200px;height:200px;background:radial-gradient(circle,#f59e0b08,transparent);border-radius:50%;transform:translate(30%,-30%)"></div>
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap">
            <div>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                    <span style="font-size:20px">⭐</span>
                    <span style="color:#f59e0b;font-size:11px;text-transform:uppercase;letter-spacing:0.12em;font-weight:700">Pick of the Day</span>
                    <span style="background:{tier_badge_color}22;color:{tier_badge_color};font-size:11px;font-weight:700;padding:2px 10px;border-radius:999px;letter-spacing:0.05em">{potd.tier}</span>
                </div>
                <div style="color:#f1f5f9;font-size:22px;font-weight:800;margin-bottom:6px">{_short(potd.away_team)} <span style="color:#475569;font-weight:400">@</span> {_short(potd.home_team)}</div>
                <div style="color:#10b981;font-size:28px;font-weight:800;margin-bottom:8px">BET: {_short(potd.backing_team)} <span style="color:#475569;font-size:16px;font-weight:500">· {mkt}</span></div>
                <div style="color:#475569;font-size:12px">{factors_html}</div>
            </div>
            <div style="text-align:right">
                <div style="color:#f59e0b;font-size:36px;font-weight:800">{units}u</div>
                <div style="color:#475569;font-size:14px">${units * UNIT_SIZE} stake</div>
                <div style="color:#475569;font-size:12px;margin-top:4px">Lose prob: {potd.losing_pct:.0%}</div>
            </div>
        </div>
    </div>
    """)

    # Picks cards grouped by tier
    tier_colors = {"STRONG": "#ef4444", "MEDIUM": "#f59e0b", "LEAN": "#94a3b8"}
    tier_labels = {"STRONG": "🔴 Strong Plays", "MEDIUM": "🟡 Medium Plays", "LEAN": "⚪ Lean Plays"}
    current_tier = None
    for p in active:
        if p.tier != current_tier:
            current_tier = p.tier
            label = tier_labels.get(p.tier, p.tier)
            tc_hdr = tier_colors.get(p.tier, "#94a3b8")
            st.html(f'<div style="color:{tc_hdr};font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin:18px 0 8px 2px">{label}</div>')

        u = UNIT_MAP.get(p.tier, 1)
        mkt = html_lib.escape(_format_market(p.recommended_market or p.proposed_market or "", p))
        tc = tier_colors.get(p.tier, "#94a3b8")
        factors_str = "  ·  ".join(html_lib.escape(str(f)) for f in p.factors[:3]) if p.factors else "—"

        # Get additional markets from layer 8
        add_markets = []
        for lo in (p.layer_outputs or []):
            if lo.layer == 8:
                add_markets = lo.data.get("additional_markets", [])
                break
        also_str = "  ·  ".join(html_lib.escape(str(m)) for m in add_markets[:3]) if add_markets else ""

        also_row = f'<div style="color:#475569;font-size:11px;margin-top:4px">Also: {also_str}</div>' if also_str else ''
        st.html(f"""
        <div style="background:#111827;border:1px solid #1e293b;border-left:3px solid {tc};border-radius:12px;padding:20px 24px;margin-bottom:12px">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
                <div style="flex:1;min-width:200px">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                        <span style="background:{tc}22;color:{tc};font-size:11px;font-weight:700;padding:2px 10px;border-radius:999px">{p.tier}</span>
                        <span style="color:#f59e0b;font-size:12px">✓ {p.factor_count} factors</span>
                    </div>
                    <div style="color:#f1f5f9;font-size:16px;font-weight:700">{_short(p.away_team)} <span style="color:#475569">@</span> {_short(p.home_team)}</div>
                    <div style="color:#10b981;font-size:13px;font-weight:600;margin-top:2px">➜ {_short(p.backing_team)} · {mkt}</div>
                    {also_row}
                    <div style="color:#475569;font-size:11px;margin-top:6px">{factors_str}</div>
                </div>
                <div style="text-align:right">
                    <div style="color:#f59e0b;font-size:22px;font-weight:800">{u}u</div>
                    <div style="color:#94a3b8;font-size:12px">${u * UNIT_SIZE}</div>
                    <div style="color:#475569;font-size:11px;margin-top:4px">Lose: {p.losing_pct:.0%}</div>
                </div>
            </div>
        </div>
        """)

    with st.expander("📋 Losing Scenarios & Full Analysis"):
        for p in active:
            u = UNIT_MAP.get(p.tier, 1)
            st.markdown(f"**{_short(p.away_team)} @ {_short(p.home_team)}** ({p.tier}, {u}u)  \n_{p.losing_scenario}_")


def render_parlays_tab(parlays: list, nrfi_parlay: dict | None):
    st.markdown("""
    <div style="margin-bottom:20px">
        <div style="color:#f1f5f9;font-size:20px;font-weight:800">🎰 Daily Parlay Card</div>
        <div style="color:#475569;font-size:12px;margin-top:2px">5 parlays built daily · ⭐ high confidence · ⚠️ best available</div>
    </div>
    """, unsafe_allow_html=True)

    if not parlays:
        st.info("No picks available to build parlays — run the model first.")
        return

    parlay_meta = [
        ("P1 — Anchor",   "🥇", "$35–$40", "#f59e0b"),
        ("P2 — Core",     "🥈", "$15–$20", "#94a3b8"),
        ("P3 — Science",  "🥉", "$10–$15", "#cd7c3a"),
        ("P4 — Push",     "🎯", "$5–$10",  "#3b82f6"),
        ("P5 — Moonshot", "🌙", "$5",      "#8b5cf6"),
    ]

    for i, parlay in enumerate(parlays):
        parlay.compute()
        label, icon, stake, accent = parlay_meta[i] if i < len(parlay_meta) else (f"Parlay {i+1}", "🎰", "$5", "#64748b")
        star   = "⭐ " if parlay.ev_pct >= 0.10 and not parlay.below_threshold else ""
        warn   = "⚠️ best-available legs" if parlay.below_threshold else ""
        ev_col = "#10b981" if parlay.ev_pct >= 0 else "#ef4444"
        legs_n = len(parlay.legs)

        warn_str = f' · {warn}' if warn else ''
        st.html(f"""
        <div style="background:#111827;border:1px solid #1e293b;border-top:2px solid {accent};border-radius:12px;padding:20px 24px;margin-bottom:14px">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:14px">
                <div style="display:flex;align-items:center;gap:10px">
                    <span style="font-size:18px">{icon}</span>
                    <span style="color:#f1f5f9;font-size:15px;font-weight:700">{star}{label}</span>
                    <span style="color:#475569;font-size:12px">{legs_n}-leg{warn_str}</span>
                </div>
                <div style="display:flex;gap:24px">
                    <div style="text-align:center"><div style="color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:0.08em">Odds</div><div style="color:#f59e0b;font-size:18px;font-weight:800">+{parlay.american_odds:,}</div></div>
                    <div style="text-align:center"><div style="color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:0.08em">Stake</div><div style="color:#f1f5f9;font-size:15px;font-weight:600">{stake}</div></div>
                    <div style="text-align:center"><div style="color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:0.08em">Win</div><div style="color:#10b981;font-size:15px;font-weight:600">~${parlay.payout_per_unit:.0f}</div></div>
                    <div style="text-align:center"><div style="color:#475569;font-size:10px;text-transform:uppercase;letter-spacing:0.08em">EV</div><div style="color:{ev_col};font-size:15px;font-weight:600">{parlay.ev_pct:+.1%}</div></div>
                </div>
            </div>
        """)

        leg_rows = []
        for j, leg in enumerate(parlay.legs, 1):
            leg_rows.append({
                "#":       j,
                "BET":     f"➜ {_short(leg.pick.backing_team)}",
                "Game":    f"{_short(leg.pick.away_team)} @ {_short(leg.pick.home_team)}",
                "Market":  leg.market[:22],
                "Price":   f"{leg.price:+d}",
                "Win %":   f"{leg.true_prob:.0%}",
                "Lose %":  f"{leg.lose_pct:.0%}",
            })
        st.dataframe(pd.DataFrame(leg_rows), hide_index=True, width='stretch')

        note = html_lib.escape(parlay.independence_notes[0] if parlay.independence_notes else "")
        nc = "#10b981" if "PASS" in note else "#f59e0b"
        st.html(f'<div style="color:{nc};font-size:11px;margin-top:6px">{note}</div>')

    if nrfi_parlay and isinstance(nrfi_parlay, dict):
        am = nrfi_parlay.get("american_odds", "")
        pw = nrfi_parlay.get("potential_win", 0)
        st.html(f"""
        <div style="background:linear-gradient(135deg,#0a1f0e,#111827);border:1px solid #10b98133;border-radius:12px;padding:20px 24px;margin-top:8px">
            <div style="color:#10b981;font-size:11px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">🚫 NRFI Bonus Parlay</div>
            <div style="display:flex;gap:28px;flex-wrap:wrap;margin-bottom:12px">
                <div><div style="color:#475569;font-size:10px">TYPE</div><div style="color:#f1f5f9;font-weight:700">{html_lib.escape(str(nrfi_parlay.get("type","")))}</div></div>
                <div><div style="color:#475569;font-size:10px">ODDS</div><div style="color:#f59e0b;font-weight:700">{html_lib.escape(str(am))}</div></div>
                <div><div style="color:#475569;font-size:10px">STAKE</div><div style="color:#f1f5f9;font-weight:700">{html_lib.escape(str(nrfi_parlay.get("recommended_stake","")))}</div></div>
                <div><div style="color:#475569;font-size:10px">WIN</div><div style="color:#10b981;font-weight:700">~${pw:.0f}</div></div>
            </div>
        </div>
        """)
        legs = nrfi_parlay.get("legs", [])
        if legs:
            rows = [{"Game": g.get("game","?"), "NRFI %": f"{g.get('nrfi_probability',0):.1%}", "Tier": g.get("tier","?")} for g in legs]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


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
            st.dataframe(pd.DataFrame(prows), hide_index=True, width='stretch')
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

    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

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
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

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
        st.dataframe(pd.DataFrame(leg_rows), hide_index=True, width='stretch')
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
            st.dataframe(pd.DataFrame(c_rows), hide_index=True, width='stretch')

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
    st.markdown("""
    <div style="margin-bottom:16px">
        <div style="color:#f1f5f9;font-size:20px;font-weight:800">📊 Model Evolution</div>
        <div style="color:#475569;font-size:12px;margin-top:2px">
            Self-improvement engine · learns from every graded pick · updates automatically
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load learned weights
    try:
        from sports_betting.models.weight_trainer import load_learned_weights
        weights = load_learned_weights()
    except Exception:
        weights = {}

    sample = weights.get("sample_size", 0)
    needed = 20
    pct    = min(1.0, sample / needed)

    # Progress toward first retraining
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e293b;border-radius:12px;padding:20px 24px;margin-bottom:16px">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px">
            <span style="color:#f1f5f9;font-weight:700">Training Progress</span>
            <span style="color:#f59e0b;font-weight:700">{sample} / {needed} graded picks</span>
        </div>
        <div style="background:#1e293b;border-radius:999px;height:8px;overflow:hidden">
            <div style="background:linear-gradient(90deg,#f59e0b,#10b981);width:{pct*100:.0f}%;height:100%;border-radius:999px;transition:width 0.5s"></div>
        </div>
        <div style="color:#475569;font-size:11px;margin-top:6px">
            {'🎓 Model is actively retraining from real results' if sample >= needed else f'Model needs {needed - sample} more graded picks before weight adjustment begins'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Thresholds section
    thresholds = weights.get("thresholds", {})
    if thresholds:
        st.markdown("#### Current Tier Thresholds")
        tier_colors = {"STRONG": "#ef4444", "MEDIUM": "#f59e0b", "LEAN": "#94a3b8", "SKIP": "#334155"}
        cols = st.columns(4)
        for i, (tier, vals) in enumerate(thresholds.items()):
            lo, hi = vals[0], vals[1]
            with cols[i % 4]:
                c = tier_colors.get(tier, "#94a3b8")
                st.markdown(f"""
                <div style="background:#111827;border:1px solid #1e293b;border-left:3px solid {c};
                            border-radius:10px;padding:12px 16px;margin-bottom:8px">
                    <div style="color:{c};font-size:11px;font-weight:700">{tier}</div>
                    <div style="color:#f1f5f9;font-size:14px;font-weight:700">Lose {lo:.0%}–{hi:.0%}</div>
                    <div style="color:#475569;font-size:10px">{'Default' if lo in (0.0, 0.25, 0.32, 0.40) else 'Learned ✓'}</div>
                </div>
                """, unsafe_allow_html=True)

    # Market performance
    market_weights = weights.get("markets", {})
    if market_weights:
        st.markdown("#### Market Win Rates (Learned)")
        mkt_rows = []
        for mkt, win_rate in sorted(market_weights.items(), key=lambda x: x[1], reverse=True):
            status = "✅ Boosted" if win_rate >= 0.60 else ("⚠️ Penalized" if win_rate < 0.48 else "→ Neutral")
            mkt_rows.append({"Market": mkt, "Win Rate": f"{win_rate:.1%}", "Status": status})
        if mkt_rows:
            st.dataframe(pd.DataFrame(mkt_rows), hide_index=True, width='stretch')

    # Factor performance
    factor_weights = weights.get("factors", {})
    if factor_weights:
        st.markdown("#### Factor Performance (Learned)")
        f_rows = []
        for fk, mult in sorted(factor_weights.items(), key=lambda x: x[1], reverse=True):
            status = "🔥 Elite" if mult >= 1.3 else ("✅ Strong" if mult >= 1.1 else ("⚠️ Weak" if mult < 0.9 else "→ Neutral"))
            f_rows.append({"Factor Signal": fk.replace("_", " ").title(), "Weight": f"x{mult:.2f}", "Status": status})
        if f_rows:
            st.dataframe(pd.DataFrame(f_rows), hide_index=True, width='stretch')

    # Legacy signal tracker
    st.markdown("#### Signal Hit Rates (All Time)")
    report = cached_signals()
    if report:
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
        st.dataframe(df, hide_index=True, width='stretch',
            column_config={
                "Hit Rate": st.column_config.ProgressColumn("Hit Rate", min_value=0, max_value=1, format="%.1%"),
                "Weight": st.column_config.NumberColumn("Weight", format="%.3f"),
            })
    else:
        st.info("Signal data builds as bets are graded. Check back after the first week.")
    st.caption("Weights auto-adjust after every cycle. Signals hitting >60% get boosted, <48% get penalized.")


def render_model_intelligence():
    """Show what the model has learned from graded picks."""
    from sports_betting.database import get_db
    from sports_betting.models.weight_trainer import (
        analyze_factor_performance, analyze_tier_accuracy,
        analyze_market_performance, analyze_context_patterns,
    )

    st.markdown("### 🧠 Model Intelligence")
    st.caption("The model learns from every graded pick. These are the patterns it has discovered so far.")

    try:
        with get_db() as conn:
            graded_count = conn.execute(
                "SELECT COUNT(*) FROM value_bets WHERE result IN ('WIN','LOSS')"
            ).fetchone()[0]
            last_retrain = conn.execute(
                "SELECT MAX(updated_at) FROM model_weights"
            ).fetchone()[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Graded Picks", graded_count)
        c2.metric("Last Retrain", str(last_retrain or "Never")[:10])
        needed = 20 - graded_count if graded_count < 20 else 0
        c3.metric("Until Next Threshold Update", f"{max(0, 50 - graded_count)} more picks" if graded_count < 50 else "Active")

        if graded_count < 5:
            st.info("Need at least 5 graded picks to show learning insights. Grade your bets in the Record tab.")
            return

        tab_f, tab_t, tab_m, tab_c = st.tabs(["📊 Factor Performance", "🎯 Tier Accuracy", "🏪 Market Win Rates", "🔍 Context Patterns"])

        with tab_f:
            st.caption("Factors with >55% win rate get upweighted in future picks. <45% get downweighted.")
            fp = analyze_factor_performance()
            if fp:
                import pandas as pd
                rows_f = sorted(fp.items(), key=lambda x: -x[1]["win_rate"])
                df_f = pd.DataFrame([
                    {"Factor": k, "Win Rate": f"{v['win_rate']:.0%}",
                     "Sample": v["sample"], "Wins": v["wins"], "Losses": v["losses"],
                     "Weight": f"{v['weight_multiplier']:.2f}x"}
                    for k, v in rows_f if v["sample"] >= 3
                ])
                if not df_f.empty:
                    st.dataframe(df_f, hide_index=True)
                else:
                    st.caption("Not enough data per factor yet.")
            else:
                st.caption("No factor data yet.")

        with tab_t:
            st.caption("Do STRONG picks actually win more often than LEAN? This is the ground truth.")
            ta = analyze_tier_accuracy()
            if ta:
                for tier, s in sorted(ta.items(), key=lambda x: {"STRONG":0,"MEDIUM":1,"LEAN":2}.get(x[0],3)):
                    color = "#ef4444" if s["win_rate"] < 0.50 else "#f59e0b" if s["win_rate"] < 0.60 else "#10b981"
                    delta = s["vs_expected"]
                    delta_str = f"{delta:+.0%} vs expected {s['expected_win_rate']:.0%}"
                    st.html(
                        f'<div style="background:#111827;border-left:3px solid {color};border-radius:8px;padding:10px 16px;margin-bottom:8px">'
                        f'<span style="color:#f1f5f9;font-weight:700">{tier}</span>'
                        f' <span style="color:{color};font-size:18px;font-weight:800;margin:0 16px">{s["win_rate"]:.0%}</span>'
                        f'<span style="color:#64748b;font-size:12px">{delta_str} · {s["sample"]} picks · {s["wins"]}W {s["losses"]}L</span>'
                        f'</div>'
                    )
            else:
                st.caption("Need graded MODEL_PICK bets to compute tier accuracy.")

        with tab_m:
            st.caption("Markets the model bets most accurately on rise in preference over time.")
            mp = analyze_market_performance()
            if mp:
                import pandas as pd
                rows_m = sorted(mp.items(), key=lambda x: -x[1]["win_rate"])
                df_m = pd.DataFrame([
                    {"Market": k, "Win Rate": f"{v['win_rate']:.0%}",
                     "Sample": v["sample"], "Wins": v["wins"], "Losses": v["losses"]}
                    for k, v in rows_m if v["sample"] >= 2
                ])
                if not df_m.empty:
                    st.dataframe(df_m, hide_index=True)
            else:
                st.caption("No market data yet.")

        with tab_c:
            st.caption("Contextual situations the model has learned to weight more or less heavily.")
            cp = analyze_context_patterns()
            if cp:
                import pandas as pd
                rows_c = sorted(cp.items(), key=lambda x: -x[1]["signal_strength"])
                df_c = pd.DataFrame([
                    {"Context": k, "Win Rate": f"{v['win_rate']:.0%}",
                     "Signal": "Strong" if v["signal_strength"] > 0.08 else "Moderate" if v["signal_strength"] > 0.04 else "Weak",
                     "Sample": v["sample"]}
                    for k, v in rows_c if v["sample"] >= 3
                ])
                if not df_c.empty:
                    st.dataframe(df_c, hide_index=True)
            else:
                st.caption("No context pattern data yet.")

    except Exception as e:
        st.caption(f"Intelligence unavailable: {e}")


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

    # ── Bet history (My Bets + Model Picks) ───────────────────────────
    st.markdown("---")

    # Grading status + regrade button
    gs = st.session_state.get("grade_status", {})
    if gs.get("graded", 0) > 0:
        st.success(f"✅ Auto-graded {gs['graded']} picks this session.")
    if gs.get("errors"):
        with st.expander("⚠️ Grading errors (click to expand)", expanded=False):
            for err in gs["errors"]:
                st.caption(err)
    col_msg, col_btn = st.columns([5, 1])
    col_msg.caption(f"Grading status: {gs.get('message', 'Not yet run')}")
    if col_btn.button("🔄 Regrade Now"):
        try:
            from sports_betting.analysis.signal_tracker import grade_pending_picks
            st.session_state["grade_status"] = grade_pending_picks()
        except Exception as e:
            st.session_state["grade_status"] = {"graded": 0, "errors": [str(e)], "skipped": 0, "message": str(e)}
        st.cache_data.clear()
        st.rerun()

    try:
        from sports_betting.database import get_db
        import json as _json

        hist_tab_my, hist_tab_model = st.tabs(["📋 My Bets", "🤖 Model Picks"])

        # ── My Bets (manually recorded) ──────────────────────────────
        with hist_tab_my:
            # ── Backup / Restore bar ─────────────────────────────────
            with st.expander("💾 Backup & Restore Bet Data", expanded=False):
                st.caption("Export your bets to CSV so data survives app restarts. Re-import to restore.")
                import io as _io, csv as _csv
                with get_db() as _bc:
                    all_bets = _bc.execute("""
                        SELECT game_id, book, market, side, book_price, model_probability,
                               implied_probability, edge, kelly_fraction, recommended_bet,
                               confidence, factors, detected_at, result, profit_loss
                        FROM value_bets ORDER BY detected_at DESC
                    """).fetchall()
                if all_bets:
                    buf = _io.StringIO()
                    w = _csv.writer(buf)
                    w.writerow(["game_id","book","market","side","book_price","model_probability",
                                "implied_probability","edge","kelly_fraction","recommended_bet",
                                "confidence","factors","detected_at","result","profit_loss"])
                    for r in all_bets:
                        w.writerow(list(r))
                    st.download_button("⬇️ Download bets CSV", buf.getvalue(),
                                       file_name="bets_backup.csv", mime="text/csv")
                else:
                    st.caption("No bets in DB yet.")

                uploaded = st.file_uploader("⬆️ Restore from CSV", type="csv", key="restore_csv")
                if uploaded:
                    try:
                        import csv as _csv2
                        reader = _csv2.DictReader(_io.StringIO(uploaded.read().decode()))
                        restored = 0
                        with get_db() as _rc:
                            for row in reader:
                                gid = row.get("game_id","")
                                if not gid:
                                    continue
                                exists = _rc.execute(
                                    "SELECT id FROM value_bets WHERE game_id=? AND confidence=? AND detected_at=? LIMIT 1",
                                    (gid, row.get("confidence",""), row.get("detected_at",""))
                                ).fetchone()
                                if exists:
                                    continue
                                _rc.execute("""
                                    INSERT OR IGNORE INTO games (game_id, home_team, away_team, game_date, status)
                                    VALUES (?, 'Unknown','Unknown', date('now'), 'scheduled')
                                """, (gid,))
                                _rc.execute("""
                                    INSERT INTO value_bets
                                    (game_id, book, market, side, book_price, model_probability,
                                     implied_probability, edge, kelly_fraction, recommended_bet,
                                     confidence, factors, detected_at, result, profit_loss)
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                """, (
                                    gid, row.get("book","draftkings"), row.get("market",""),
                                    row.get("side",""), float(row.get("book_price") or 0),
                                    float(row.get("model_probability") or 0),
                                    float(row.get("implied_probability") or 0),
                                    float(row.get("edge") or 0),
                                    float(row.get("kelly_fraction") or 0),
                                    float(row.get("recommended_bet") or 5),
                                    row.get("confidence","PLACED"),
                                    row.get("factors","[]"),
                                    row.get("detected_at",""),
                                    row.get("result") or None,
                                    float(row.get("profit_loss") or 0) if row.get("profit_loss") else None,
                                ))
                                restored += 1
                        st.success(f"Restored {restored} bets.")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Restore failed: {e}")

            with get_db() as conn:
                rows = conn.execute("""
                    SELECT game_id, market, side, book_price, recommended_bet, result, detected_at, factors
                    FROM value_bets
                    WHERE confidence = 'PLACED'
                    ORDER BY detected_at DESC LIMIT 40
                """).fetchall()

            if not rows:
                st.caption("No bets recorded yet — submit a bet above to start tracking. Use the backup tool above to restore previous data.")
            else:
                # ── Grade pending PLACED bets ─────────────────────────────
                pending_placed = [r for r in rows if not r[5]]
                if pending_placed:
                    with st.expander(f"⏳ Grade Pending Bets ({len(pending_placed)} ungraded)", expanded=True):
                        st.caption("Mark each bet WIN or LOSS — results feed directly into the ROI header.")
                        from sports_betting.database import get_db as _get_db
                        for pr in pending_placed:
                            bet_id_row = None
                            with _get_db() as _conn:
                                bet_id_row = _conn.execute(
                                    "SELECT id, book_price, recommended_bet FROM value_bets WHERE game_id=? AND confidence='PLACED' AND result IS NULL LIMIT 1",
                                    (pr[0],)
                                ).fetchone()
                            if not bet_id_row:
                                continue
                            bid, price, stake = bet_id_row[0], bet_id_row[1] or 0, bet_id_row[2] or 5.0
                            label = f"{str(pr[2]).upper()[:16] if pr[2] else '?'}  {str(pr[1])[:18] if pr[1] else ''}  ({int(price):+d})  –  {str(pr[0])[:24]}"
                            col_l, col_w, col_x = st.columns([4, 1, 1])
                            col_l.markdown(f"<span style='color:#f1f5f9;font-size:13px'>{html_lib.escape(label)}</span>", unsafe_allow_html=True)
                            if col_w.button("✅ WIN", key=f"win_{bid}"):
                                pl = stake * price / 100 if price > 0 else stake * 100 / abs(price)
                                with _get_db() as _conn:
                                    _conn.execute("UPDATE value_bets SET result='WIN', profit_loss=? WHERE id=?", (round(pl,2), bid))
                                st.cache_data.clear()
                                st.rerun()
                            if col_x.button("❌ LOSS", key=f"loss_{bid}"):
                                with _get_db() as _conn:
                                    _conn.execute("UPDATE value_bets SET result='LOSS', profit_loss=? WHERE id=?", (-stake, bid))
                                st.cache_data.clear()
                                st.rerun()

                def _parse_parlay_id(factors_str):
                    try:
                        for f in _json.loads(factors_str or "[]"):
                            if str(f).startswith("parlay:"):
                                return str(f).replace("parlay:", "")
                    except Exception:
                        pass
                    return None

                def _result_badge(res):
                    if res == "WIN":  return "🟢", "#10b981"
                    if res == "LOSS": return "🔴", "#ef4444"
                    return "⏳", "#94a3b8"

                groups, singles = {}, []
                for r in rows:
                    pid = _parse_parlay_id(r[7])
                    if pid:
                        groups.setdefault(pid, []).append(r)
                    else:
                        singles.append(r)

                for pid, legs in groups.items():
                    date_h = str(legs[0][6])[:10] if legs[0][6] else "?"
                    results = [r[5] for r in legs]
                    if all(r == "WIN" for r in results):  g_icon, g_color, g_label = "🟢", "#10b981", "All Win"
                    elif any(r == "LOSS" for r in results): g_icon, g_color, g_label = "🔴", "#ef4444", "Loss"
                    else:                                  g_icon, g_color, g_label = "⏳", "#94a3b8", "Pending"
                    leg_html = ""
                    for leg_row in legs:
                        r_icon, _ = _result_badge(leg_row[5])
                        side_e = html_lib.escape(str(leg_row[2]).upper()[:14]) if leg_row[2] else "?"
                        mkt_e  = html_lib.escape(str(leg_row[1])[:20]) if leg_row[1] else "?"
                        price_e = f"{int(leg_row[3]):+d}" if leg_row[3] else "?"
                        leg_html += (
                            f'<div style="display:flex;gap:10px;padding:4px 0;border-top:1px solid #1e293b">'
                            f'<span>{r_icon}</span>'
                            f'<span style="color:#f1f5f9;font-size:13px;min-width:90px">{side_e}</span>'
                            f'<span style="color:#64748b;font-size:12px;min-width:110px">{mkt_e}</span>'
                            f'<span style="color:#f59e0b;font-size:12px">{price_e}</span>'
                            f'</div>'
                        )
                    with st.container():
                        st.html(
                            f'<div style="background:#111827;border:1px solid #1e293b;border-left:3px solid #8b5cf6;border-radius:10px;padding:14px 18px;margin-bottom:10px">'
                            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
                            f'<span style="color:#8b5cf6;font-size:11px;font-weight:700;text-transform:uppercase">{g_icon} {len(legs)}-Leg Parlay · {date_h}</span>'
                            f'<span style="color:{g_color};font-size:12px;font-weight:600">{g_label}</span>'
                            f'</div>{leg_html}</div>'
                        )

                for r in singles:
                    r_icon, r_color = _result_badge(r[5])
                    side_e  = html_lib.escape(str(r[2]).upper()[:16]) if r[2] else "?"
                    mkt_e   = html_lib.escape(str(r[1])[:20]) if r[1] else "?"
                    price_e = f"{int(r[3]):+d}" if r[3] else "?"
                    stake_e = f"${r[4]:.0f}" if r[4] else "?"
                    date_h  = html_lib.escape(str(r[6])[:10]) if r[6] else "?"
                    res_e   = html_lib.escape(r[5] or "Pending")
                    st.html(
                        f'<div style="background:#111827;border:1px solid #1e293b;border-left:3px solid #3b82f6;border-radius:10px;padding:12px 18px;margin-bottom:8px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">'
                        f'<span>{r_icon}</span>'
                        f'<span style="color:#f1f5f9;font-size:14px;font-weight:700;min-width:80px">{side_e}</span>'
                        f'<span style="color:#64748b;font-size:13px;min-width:110px">{mkt_e}</span>'
                        f'<span style="color:#f59e0b;font-size:13px;min-width:55px">{price_e}</span>'
                        f'<span style="color:#94a3b8;font-size:12px">{stake_e}</span>'
                        f'<span style="color:{r_color};font-size:12px;font-weight:600;margin-left:auto">{res_e}</span>'
                        f'<span style="color:#334155;font-size:11px">{date_h}</span>'
                        f'</div>'
                    )

        # ── Model Picks (auto-tracked) ────────────────────────────────
        with hist_tab_model:
            with get_db() as conn:
                # No row limit — show full lifetime history
                model_rows = conn.execute("""
                    SELECT game_id, market, side, book_price, model_probability,
                           edge, result, detected_at, factors, confidence
                    FROM value_bets
                    WHERE confidence IN ('MODEL_PICK', 'MODEL_PARLAY')
                    ORDER BY detected_at DESC
                """).fetchall()

            if not model_rows:
                st.info("No model picks tracked yet — run the model to start auto-tracking picks and parlays.")
            else:
                # ── Lifetime performance summary ──────────────────────
                graded   = [r for r in model_rows if r[6] in ("WIN", "LOSS")]
                wins     = sum(1 for r in graded if r[6] == "WIN")
                losses   = len(graded) - wins
                pending  = sum(1 for r in model_rows if not r[6])
                win_rate = wins / len(graded) if graded else 0.0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("All-Time Picks", len(model_rows))
                c2.metric("Win Rate", f"{win_rate:.0%}" if graded else "—",
                          f"{wins}W / {losses}L" if graded else None)
                c3.metric("Graded", len(graded))
                c4.metric("Pending Grade", pending)

                st.markdown("---")

                # Separate single picks and parlay legs
                single_picks, parlay_legs = [], []
                for r in model_rows:
                    if r[9] == "MODEL_PICK":
                        single_picks.append(r)
                    else:
                        parlay_legs.append(r)

                # --- Single picks ---
                if single_picks:
                    st.markdown("#### 🎯 Single Picks")
                    tier_colors_hist = {"STRONG": "#ef4444", "MEDIUM": "#f59e0b", "LEAN": "#94a3b8"}
                    for r in single_picks:
                        result    = r[6]
                        r_icon    = "🟢" if result == "WIN" else "🔴" if result == "LOSS" else "⏳"
                        r_color   = "#10b981" if result == "WIN" else "#ef4444" if result == "LOSS" else "#94a3b8"
                        side_     = html_lib.escape(str(r[2]).title()[:20]) if r[2] else "?"
                        mkt_      = html_lib.escape(str(r[1])[:22]) if r[1] else "?"
                        date_h    = str(r[7])[:10] if r[7] else "?"
                        ev_       = f"{float(r[5])*100:+.1f}%" if r[5] else "—"
                        # Parse tier from factors
                        tier = "LEAN"
                        try:
                            for f in _json.loads(r[8] or "[]"):
                                if str(f).upper() in ("STRONG", "MEDIUM", "LEAN"):
                                    tier = str(f).upper(); break
                        except Exception:
                            pass
                        tc = tier_colors_hist.get(tier, "#94a3b8")
                        st.html(f"""
                        <div style="background:#111827;border:1px solid #1e293b;border-left:3px solid {tc};border-radius:10px;padding:12px 18px;margin-bottom:8px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
                            <span style="font-size:15px">{r_icon}</span>
                            <span style="background:{tc}22;color:{tc};font-size:10px;font-weight:700;padding:2px 8px;border-radius:999px">{tier}</span>
                            <span style="color:#f1f5f9;font-size:14px;font-weight:700;min-width:80px">{side_}</span>
                            <span style="color:#64748b;font-size:12px;min-width:110px">{mkt_}</span>
                            <span style="color:#f59e0b;font-size:12px;min-width:50px">EV {ev_}</span>
                            <span style="color:{r_color};font-size:12px;font-weight:600;margin-left:auto">{result or 'Pending'}</span>
                            <span style="color:#334155;font-size:11px">{date_h}</span>
                        </div>""")

                # --- Parlay legs grouped by parlay label ---
                if parlay_legs:
                    st.markdown("#### 🎰 Parlay History")
                    parlay_groups: dict[str, list] = {}
                    for r in parlay_legs:
                        label = "Unknown"
                        try:
                            for f in _json.loads(r[8] or "[]"):
                                if str(f).startswith("parlay:"):
                                    label = str(f).replace("parlay:", ""); break
                        except Exception:
                            pass
                        parlay_groups.setdefault(label, []).append(r)

                    for plabel, legs in parlay_groups.items():
                        date_h    = str(legs[0][7])[:10] if legs[0][7] else "?"
                        p_results = [r[6] for r in legs]
                        if all(r == "WIN" for r in p_results):  g_icon, g_color, g_label = "🟢", "#10b981", "All Win"
                        elif any(r == "LOSS" for r in p_results): g_icon, g_color, g_label = "🔴", "#ef4444", "Loss"
                        else:                                     g_icon, g_color, g_label = "⏳", "#94a3b8", "Pending"
                        short_label = plabel.split("_")[1] if "_" in plabel else plabel
                        leg_rows_html = ""
                        for r in legs:
                            icon = "🟢" if r[6] == "WIN" else ("🔴" if r[6] == "LOSS" else "⏳")
                            res_color = "#10b981" if r[6] == "WIN" else ("#ef4444" if r[6] == "LOSS" else "#94a3b8")
                            side_e = html_lib.escape(str(r[2]).title()[:18]) if r[2] else "?"
                            mkt_e  = html_lib.escape(str(r[1])[:20]) if r[1] else "?"
                            res_e  = html_lib.escape(r[6] or "Pending")
                            leg_rows_html += (
                                f'<div style="display:flex;gap:10px;padding:4px 0;border-top:1px solid #1e293b">'
                                f'<span>{icon}</span>'
                                f'<span style="color:#f1f5f9;font-size:13px;min-width:90px">{side_e}</span>'
                                f'<span style="color:#64748b;font-size:12px">{mkt_e}</span>'
                                f'<span style="color:{res_color};font-size:12px;margin-left:auto">{res_e}</span>'
                                f'</div>'
                            )
                        with st.container():
                            st.html(
                                f'<div style="background:#111827;border:1px solid #1e293b;border-left:3px solid #8b5cf6;border-radius:10px;padding:14px 18px;margin-bottom:10px">'
                                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
                                f'<span style="color:#8b5cf6;font-size:11px;font-weight:700;text-transform:uppercase">{g_icon} {short_label} · {len(legs)}-leg · {date_h}</span>'
                                f'<span style="color:{g_color};font-size:12px;font-weight:600">{g_label}</span>'
                                f'</div>'
                                f'{leg_rows_html}'
                                f'</div>'
                            )

    except Exception as e:
        st.caption(f"Bet history unavailable: {e}")


# ── Ask AI tab ────────────────────────────────────────────────────────

def _build_picks_context(picks: list, parlays: list) -> str:
    """Summarise today's picks into a compact text block for the AI system prompt."""
    lines = [f"Today's date: {datetime.now().strftime('%A, %B %d, %Y')}", ""]
    tier_order = {"STRONG": 0, "MEDIUM": 1, "LEAN": 2}
    active = sorted(
        [p for p in picks if getattr(p, "tier", "SKIP") != "SKIP"],
        key=lambda p: (tier_order.get(p.tier, 9), -(p.factor_count or 0)),
    )
    if active:
        lines.append("=== TODAY'S MODEL PICKS ===")
        for p in active:
            mkt = _format_market(getattr(p, "recommended_market", "") or getattr(p, "proposed_market", ""), p)
            ev = getattr(p, "ev_pct", 0) or 0
            factors = getattr(p, "factors", []) or []
            lines.append(
                f"[{p.tier}] {p.backing_team} — {mkt} | EV: {ev:+.1f}% | "
                f"Factors: {', '.join(str(f) for f in factors[:5])}"
            )
    else:
        lines.append("No active picks today.")

    lines.append("")
    if parlays:
        lines.append("=== TODAY'S PARLAYS ===")
        for i, par in enumerate(parlays, 1):
            legs = getattr(par, "legs", [])
            dec = getattr(par, "combined_decimal", 1.0) or 1.0
            # Convert decimal to American odds for display
            if dec >= 2.0:
                combined_american = int((dec - 1) * 100)
            else:
                combined_american = int(-100 / (dec - 1)) if dec > 1 else 0
            leg_strs = []
            for lg in legs:
                team = getattr(getattr(lg, "pick", None), "backing_team", "?") or "?"
                mkt = getattr(lg, "market", "?") or "?"
                leg_strs.append(f"{team} ({mkt})")
            lines.append(f"Parlay {i} ({getattr(par, 'label', '')}): {' + '.join(leg_strs)} | Odds: +{combined_american}")

    return "\n".join(lines)


def render_ask_ai_tab(picks: list, parlays: list):
    st.markdown("### 💬 Ask the Model")
    st.caption(
        "Ask anything about today's picks — e.g. 'Which of the STRONG picks do you like best?' "
        "or 'Should I parlay the first two picks?'"
    )

    # Get API key
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        st.warning(
            "Add `ANTHROPIC_API_KEY` to your Streamlit secrets to enable the AI chat. "
            "Get a key at console.anthropic.com."
        )
        return

    # Build system prompt once per session (picks change daily)
    if "ai_picks_context" not in st.session_state:
        st.session_state["ai_picks_context"] = _build_picks_context(picks, parlays)
    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []

    system_prompt = (
        "You are an expert MLB sports betting analyst assistant embedded in a betting model dashboard. "
        "You have access to today's model picks, their tiers (STRONG/MEDIUM/LEAN), markets, "
        "expected value percentages, and key statistical factors. "
        "Answer the user's questions concisely and analytically. "
        "When comparing picks, weigh EV%, tier, and the quality of factors. "
        "Never fabricate odds or stats not provided in the context. "
        "Always remind the user that all picks are for research purposes only.\n\n"
        f"--- PICKS CONTEXT ---\n{st.session_state['ai_picks_context']}\n--- END CONTEXT ---"
    )

    # Render previous messages
    for msg in st.session_state["ai_chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about today's picks...")
    if not user_input:
        return

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["ai_chat_history"].append({"role": "user", "content": user_input})

    # Build messages list for API
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["ai_chat_history"]
    ]

    # Stream response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            with client.messages.stream(
                model="claude-opus-4-7",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text_chunk in stream.text_stream:
                    full_response += text_chunk
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Error calling Claude API: {e}"
            response_placeholder.error(full_response)

    st.session_state["ai_chat_history"].append({"role": "assistant", "content": full_response})

    # Keep history bounded
    if len(st.session_state["ai_chat_history"]) > 40:
        st.session_state["ai_chat_history"] = st.session_state["ai_chat_history"][-40:]


# ── App entry point ───────────────────────────────────────────────────

def main():
    inject_css()
    get_db_connection()

    # Grade ALL pending picks every load (no session cache — games finish at
    # different times and we want results to appear as soon as possible).
    try:
        from sports_betting.analysis.signal_tracker import grade_pending_picks
        gs = grade_pending_picks()
        if gs.get("graded", 0):
            st.cache_data.clear()   # force header metrics to refresh
        st.session_state["grade_status"] = gs
    except Exception as e:
        st.session_state["grade_status"] = {
            "graded": 0, "errors": [str(e)], "skipped": 0, "message": str(e)
        }

    date_str, run_btn = render_sidebar()

    st.markdown("""
    <div style="margin-bottom:4px">
        <span style="color:#f59e0b;font-size:12px;text-transform:uppercase;letter-spacing:0.14em;font-weight:700">⚾ MLB Betting Model</span>
        <span style="color:#1e293b"> · </span>
        <span style="color:#334155;font-size:12px">v4.0 · 12-Layer Statistical Framework · DraftKings</span>
    </div>
    """, unsafe_allow_html=True)

    # Combined bets ROI header (manually placed + model)
    roi = cached_roi()
    render_header(roi)
    # Model-only autonomous record
    render_model_header(cached_model_roi())

    # Run model
    if run_btn:
        st.cache_data.clear()

    with st.spinner("⚡ Running 12-layer analysis..."):
        try:
            results = cached_run_model(date_str)
        except Exception as e:
            st.error(f"Model error: {e}")
            st.info("Check that your API keys are set in the Streamlit secrets panel.")
            return

    picks        = results.get("picks", [])
    parlays      = results.get("parlays", [])

    # Auto-track all model picks + parlay legs in DB for self-improvement
    try:
        from sports_betting.analysis.signal_tracker import auto_record_model_picks
        auto_record_model_picks(
            [p for p in picks if getattr(p, "tier", "SKIP") != "SKIP"],
            parlays,
            date_str,
        )
    except Exception:
        pass
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
        "🧠 Intelligence",
        "💬 Ask AI",
    ])

    with tabs[0]: render_picks_tab(picks)
    with tabs[1]: render_parlays_tab(parlays, nrfi_parlay)
    with tabs[2]: render_nrfi_tab(nrfi_ranked, nrfi_parlay)
    with tabs[3]: render_hr_parlay_tab(hr_results)
    with tabs[4]: render_intelligence_tab(all_signals, xwoba_luck, picks, sharp_plays)
    with tabs[5]: render_skipped_tab(skipped)
    with tabs[6]: render_signals_tab()
    with tabs[7]: render_record_bet_tab(picks, parlays, nrfi_ranked, hr_results)
    with tabs[8]: render_model_intelligence()
    with tabs[9]: render_ask_ai_tab(picks, parlays)

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
