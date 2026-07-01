"""
Database layer — SQLite for local dev, PostgreSQL/Supabase when DATABASE_URL is set.

All queries may be written in SQLite dialect (? placeholders, :name params,
datetime('now'), INSERT OR IGNORE, etc.). _PGConn transparently converts to
PostgreSQL syntax at runtime when DATABASE_URL is detected.
"""
from __future__ import annotations
import os
import re
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime as _dt, timedelta, timezone

logger = logging.getLogger(__name__)


# ── Backend detection ──────────────────────────────────────────────────

def _load_db_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if url:
        return url
    try:
        import streamlit as st
        return st.secrets.get("DATABASE_URL", "") or ""
    except Exception:
        return ""


DATABASE_URL: str = _load_db_url()
_USE_PG: bool = bool(DATABASE_URL)

if _USE_PG:
    logger.info("database: using PostgreSQL / Supabase")
else:
    from .config import DB_PATH
    logger.info("database: using SQLite")


# ── SQLite → PostgreSQL SQL translation ───────────────────────────────

_RE_TEXT_TS  = re.compile(r"TEXT\s+DEFAULT\s+\(datetime\('now'\)\)", re.I)
_RE_INS_IGN  = re.compile(r"INSERT\s+OR\s+IGNORE\s+INTO", re.I)
_RE_INS_REPL = re.compile(r"INSERT\s+OR\s+REPLACE\s+INTO", re.I)
_RE_DT_INTV  = re.compile(r"datetime\('now',\s*['\"](-?\d+)\s+(\w+)['\"]\)", re.I)
_RE_DT_NOW   = re.compile(r"datetime\('now'\)", re.I)
_RE_DATE_NOW = re.compile(r"date\('now'\)", re.I)
_RE_AUTOINC  = re.compile(r"INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT", re.I)
_RE_NAMED    = re.compile(r":([a-zA-Z_]\w*)")


def _to_pg(sql: str) -> str:
    """Convert SQLite-dialect SQL to PostgreSQL."""
    had_ignore  = bool(_RE_INS_IGN.search(sql))
    had_replace = bool(_RE_INS_REPL.search(sql))

    # DDL: TEXT timestamp columns → TIMESTAMPTZ
    sql = _RE_TEXT_TS.sub("TIMESTAMPTZ DEFAULT NOW()", sql)
    # INSERT OR IGNORE → INSERT (ON CONFLICT DO NOTHING appended later)
    sql = _RE_INS_IGN.sub("INSERT INTO", sql)
    # INSERT OR REPLACE → INSERT (ON CONFLICT DO NOTHING appended later)
    sql = _RE_INS_REPL.sub("INSERT INTO", sql)
    # datetime('now', '-N unit') → NOW() - INTERVAL 'N unit'
    sql = _RE_DT_INTV.sub(
        lambda m: f"NOW() - INTERVAL '{abs(int(m.group(1)))} {m.group(2)}'", sql
    )
    # datetime('now') → NOW()
    sql = _RE_DT_NOW.sub("NOW()", sql)
    # date('now') → CURRENT_DATE
    sql = _RE_DATE_NOW.sub("CURRENT_DATE", sql)
    # INTEGER PRIMARY KEY AUTOINCREMENT → SERIAL PRIMARY KEY
    sql = _RE_AUTOINC.sub("SERIAL PRIMARY KEY", sql)
    sql = sql.replace("AUTOINCREMENT", "")
    # ? → %s  (positional params)
    sql = sql.replace("?", "%s")
    # :name → %(name)s  (named params)
    sql = _RE_NAMED.sub(r"%(\1)s", sql)
    # Append ON CONFLICT DO NOTHING for INSERT OR IGNORE / INSERT OR REPLACE
    if (had_ignore or had_replace) and "ON CONFLICT" not in sql.upper():
        sql = sql.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"

    return sql


# ── psycopg2 result normalisation ─────────────────────────────────────

def _norm(val):
    """Convert psycopg2 datetime/date objects → ISO strings (matches SQLite output)."""
    import datetime as _datetime_mod
    if isinstance(val, _dt):
        return val.isoformat()
    if isinstance(val, _datetime_mod.date):
        return val.isoformat()
    return val


def _pg_params(params):
    """Normalize params for PostgreSQL: cast Python booleans to int (SQLite stores as 0/1)."""
    if params is None:
        return None
    if isinstance(params, dict):
        return {k: (int(v) if isinstance(v, bool) else v) for k, v in params.items()}
    return tuple(int(v) if isinstance(v, bool) else v for v in params)


def _norm_row(row) -> dict:
    return {k: _norm(v) for k, v in dict(row).items()}


# ── psycopg2 wrapper ──────────────────────────────────────────────────

class _PGCursor:
    def __init__(self, cur):
        self._cur = cur

    def fetchone(self) -> dict | None:
        row = self._cur.fetchone()
        return _norm_row(row) if row else None

    def fetchall(self) -> list[dict]:
        return [_norm_row(r) for r in self._cur.fetchall()]

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount


class _PGConn:
    """Wraps a psycopg2 connection to look like a sqlite3 connection."""

    def __init__(self, raw):
        self._raw = raw

    def _cur(self):
        import psycopg2.extras
        return self._raw.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def execute(self, sql: str, params=None) -> _PGCursor:
        cur = self._cur()
        cur.execute(_to_pg(sql), _pg_params(params))
        return _PGCursor(cur)

    def executemany(self, sql: str, params_list):
        cur = self._cur()
        pg = _to_pg(sql)
        for p in params_list:
            cur.execute(pg, _pg_params(p))
        return self

    def executescript(self, script: str):
        """Emulate sqlite3.executescript — splits on ; and runs each statement."""
        cur = self._cur()
        for stmt in script.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(_to_pg(stmt))
        return self

    def commit(self):
        self._raw.commit()

    def rollback(self):
        self._raw.rollback()

    def close(self):
        self._raw.close()


# ── Context manager ────────────────────────────────────────────────────

@contextmanager
def get_db():
    if _USE_PG:
        import psycopg2
        raw = psycopg2.connect(DATABASE_URL, sslmode="require", connect_timeout=10)
        conn = _PGConn(raw)
        try:
            yield conn
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.close()
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ── Schema ─────────────────────────────────────────────────────────────

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE NOT NULL,
                sport TEXT NOT NULL DEFAULT 'MLB',
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                game_date TEXT NOT NULL,
                game_time TEXT,
                venue TEXT,
                status TEXT DEFAULT 'scheduled',
                home_score INTEGER,
                away_score INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                book TEXT NOT NULL,
                market TEXT NOT NULL,
                home_line REAL,
                away_line REAL,
                home_price REAL,
                away_price REAL,
                over_price REAL,
                under_price REAL,
                total REAL,
                snapshot_time TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            );

            CREATE TABLE IF NOT EXISTS line_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                book TEXT NOT NULL,
                market TEXT NOT NULL,
                old_value REAL,
                new_value REAL,
                old_price REAL,
                new_price REAL,
                side TEXT,
                movement_time TEXT DEFAULT (datetime('now')),
                sharp_flag INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sharp_plays (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                side TEXT NOT NULL,
                market TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                signal_strength REAL,
                opening_line REAL,
                current_line REAL,
                ticket_pct REAL,
                money_pct REAL,
                notes TEXT,
                detected_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                market TEXT NOT NULL,
                book_a TEXT NOT NULL,
                book_b TEXT NOT NULL,
                side_a TEXT NOT NULL,
                side_b TEXT NOT NULL,
                price_a REAL NOT NULL,
                price_b REAL NOT NULL,
                arb_pct REAL NOT NULL,
                profit_pct REAL NOT NULL,
                detected_at TEXT DEFAULT (datetime('now')),
                expired INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS value_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                book TEXT NOT NULL,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                book_price REAL NOT NULL,
                model_probability REAL NOT NULL,
                implied_probability REAL NOT NULL,
                edge REAL NOT NULL,
                kelly_fraction REAL,
                recommended_bet REAL,
                confidence TEXT,
                factors TEXT,
                detected_at TEXT DEFAULT (datetime('now')),
                result TEXT,
                profit_loss REAL
            );

            CREATE TABLE IF NOT EXISTS team_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                season INTEGER NOT NULL,
                stat_date TEXT NOT NULL,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                run_diff INTEGER DEFAULT 0,
                batting_avg REAL,
                ops REAL,
                era REAL,
                whip REAL,
                bullpen_era REAL,
                last_10_record TEXT,
                home_record TEXT,
                away_record TEXT,
                streak_type TEXT,
                streak_count INTEGER DEFAULT 0,
                raw_stats TEXT,
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(team_name, season, stat_date)
            );

            CREATE TABLE IF NOT EXISTS pitcher_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                season INTEGER NOT NULL,
                stat_date TEXT NOT NULL,
                era REAL,
                whip REAL,
                k_per_9 REAL,
                bb_per_9 REAL,
                fip REAL,
                xfip REAL,
                last_5_era REAL,
                last_5_whip REAL,
                streak_type TEXT,
                raw_stats TEXT,
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(player_id, season, stat_date)
            );

            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                temperature REAL,
                feels_like REAL,
                humidity INTEGER,
                wind_speed REAL,
                wind_direction INTEGER,
                wind_gust REAL,
                conditions TEXT,
                precipitation_chance REAL,
                visibility REAL,
                roof_closed INTEGER DEFAULT 0,
                weather_impact_score REAL,
                fetched_at TEXT DEFAULT (datetime('now')),
                UNIQUE(game_id)
            );

            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                home_win_prob REAL,
                away_win_prob REAL,
                predicted_total REAL,
                confidence REAL,
                features TEXT,
                prediction_time TEXT DEFAULT (datetime('now')),
                UNIQUE(game_id, model_version)
            );

            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy REAL,
                roi REAL,
                brier_score REAL,
                log_loss REAL,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS model_weights (
                weight_key TEXT PRIMARY KEY,
                weight_value REAL NOT NULL,
                sample_size INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS factor_performance (
                factor_key TEXT PRIMARY KEY,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                weight REAL DEFAULT 1.0,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS market_performance (
                market TEXT PRIMARY KEY,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_snapshots(game_id, snapshot_time);
            CREATE INDEX IF NOT EXISTS idx_movements_game ON line_movements(game_id, movement_time);
            CREATE INDEX IF NOT EXISTS idx_value_bets_detected ON value_bets(detected_at);
            CREATE INDEX IF NOT EXISTS idx_arb_detected ON arbitrage_opportunities(detected_at, expired);
            CREATE INDEX IF NOT EXISTS idx_team_stats ON team_stats(team_name, season);
            CREATE INDEX IF NOT EXISTS idx_pitcher_stats ON pitcher_stats(player_id, season)
        """)
    _migrate_value_bets_columns()
    logger.info("Database initialised (%s)", "PostgreSQL" if _USE_PG else f"SQLite @ {DB_PATH}")


def _migrate_value_bets_columns():
    """
    Add CLV-tracking columns to value_bets if missing. Safe to run repeatedly:
    ALTER ... ADD COLUMN fails if the column exists, so each is tried in its own
    statement and errors are ignored. This is what makes real CLV possible — the
    old schema had no closing_price column, so CLV was permanently 0.
    """
    cols = [
        ("closing_price", "REAL"),
        ("market_novig_prob", "REAL"),
        ("clv", "REAL"),
    ]
    for name, sqltype in cols:
        try:
            with get_db() as conn:
                conn.execute(f"ALTER TABLE value_bets ADD COLUMN {name} {sqltype}")
            logger.info("value_bets: added column %s", name)
        except Exception:
            pass  # already exists


# ── DML helpers ────────────────────────────────────────────────────────

def upsert_game(game_data: dict):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO games (game_id, home_team, away_team, game_date, game_time, venue, status)
            VALUES (:game_id, :home_team, :away_team, :game_date, :game_time, :venue, :status)
            ON CONFLICT(game_id) DO UPDATE SET
                status=excluded.status,
                home_score=excluded.home_score,
                away_score=excluded.away_score,
                updated_at=datetime('now')
        """, game_data)


def insert_odds_snapshot(data: dict):
    with get_db() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO games (game_id, home_team, away_team, game_date, status)
            VALUES (:game_id,
                    COALESCE(:home_team, 'Unknown'),
                    COALESCE(:away_team, 'Unknown'),
                    COALESCE(:game_date, date('now')),
                    'scheduled')
        """, {
            "game_id":   data.get("game_id"),
            "home_team": data.get("home_team"),
            "away_team": data.get("away_team"),
            "game_date": data.get("game_date"),
        })
        conn.execute("""
            INSERT INTO odds_snapshots
            (game_id, book, market, home_line, away_line, home_price, away_price,
             over_price, under_price, total)
            VALUES (:game_id, :book, :market, :home_line, :away_line, :home_price, :away_price,
                    :over_price, :under_price, :total)
        """, data)


def get_latest_odds(game_id: str, book: str, market: str):
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM odds_snapshots
            WHERE game_id=? AND book=? AND market=?
            ORDER BY snapshot_time DESC LIMIT 1
        """, (game_id, book, market)).fetchone()
        return dict(row) if row else None


def save_value_bet(data: dict):
    with get_db() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO games (game_id, home_team, away_team, game_date, status)
            VALUES (:game_id, 'Unknown', 'Unknown', date('now'), 'scheduled')
        """, {"game_id": data.get("game_id")})
        conn.execute("""
            INSERT INTO value_bets
            (game_id, book, market, side, book_price, model_probability, implied_probability,
             edge, kelly_fraction, recommended_bet, confidence, factors)
            VALUES (:game_id, :book, :market, :side, :book_price, :model_probability,
                    :implied_probability, :edge, :kelly_fraction, :recommended_bet,
                    :confidence, :factors)
        """, data)


def record_model_pick(data: dict) -> bool:
    """
    Persist a model_v4 pick to value_bets so it can be graded and its CLV
    tracked. This is the link that was previously MISSING — model picks were
    shown to the user but never written to the DB, so grading, ROI, CLV and the
    self-learning loop all ran on a disconnected pipeline.

    Deduplicates on (game_id, side, market) per calendar day so re-running the
    model doesn't create duplicate rows; a re-run instead refreshes the closing
    line (see update_pick_closing).

    Expected keys: game_id, side, market, book_price, model_probability,
    implied_probability, edge, kelly_fraction, recommended_bet,
    market_novig_prob, factors, home_team, away_team.
    """
    with get_db() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO games (game_id, home_team, away_team, game_date, status)
            VALUES (:game_id, :home_team, :away_team, date('now'), 'scheduled')
        """, {
            "game_id":   data.get("game_id"),
            "home_team": data.get("home_team", "Unknown"),
            "away_team": data.get("away_team", "Unknown"),
        })

        existing = conn.execute("""
            SELECT id FROM value_bets
            WHERE game_id=? AND side=? AND market=?
            AND confidence='MODEL_PICK'
            AND DATE(detected_at)=date('now')
        """, (data.get("game_id"), data.get("side"), data.get("market"))).fetchone()
        if existing:
            # Refresh the latest line as we approach first pitch (closing proxy).
            # Done inline on the same connection to avoid a nested-write lock.
            if data.get("book_price") is not None:
                conn.execute("""
                    UPDATE value_bets SET closing_price=?
                    WHERE game_id=? AND side=? AND market=?
                    AND confidence='MODEL_PICK'
                    AND DATE(detected_at)=date('now')
                """, (data.get("book_price"), data.get("game_id"),
                      data.get("side"), data.get("market")))
            return False

        conn.execute("""
            INSERT INTO value_bets
            (game_id, book, market, side, book_price, model_probability,
             implied_probability, edge, kelly_fraction, recommended_bet,
             confidence, factors, market_novig_prob)
            VALUES (:game_id, :book, :market, :side, :book_price, :model_probability,
                    :implied_probability, :edge, :kelly_fraction, :recommended_bet,
                    'MODEL_PICK', :factors, :market_novig_prob)
        """, {
            "game_id":             data.get("game_id"),
            "book":                data.get("book", "model"),
            "market":              data.get("market", "full_game_ml"),
            "side":                data.get("side"),
            "book_price":          data.get("book_price"),
            "model_probability":   data.get("model_probability"),
            "implied_probability": data.get("implied_probability"),
            "edge":                data.get("edge", 0),
            "kelly_fraction":      data.get("kelly_fraction"),
            "recommended_bet":     data.get("recommended_bet"),
            "factors":             data.get("factors"),
            "market_novig_prob":   data.get("market_novig_prob"),
        })
    return True


def update_pick_closing(game_id: str, side: str, market: str,
                        closing_price, closing_novig) -> None:
    """
    Update the closing line for today's still-pending pick. Because the model may
    run multiple times a day, the last line we see before first pitch is our best
    available proxy for the closing number — the basis for CLV.
    """
    if closing_price is None:
        return
    with get_db() as conn:
        conn.execute("""
            UPDATE value_bets
            SET closing_price=?
            WHERE game_id=? AND side=? AND market=?
            AND confidence='MODEL_PICK'
            AND DATE(detected_at)=date('now')
        """, (closing_price, game_id, side, market))


def get_clv_summary(days: int = 30) -> dict:
    """
    Average closing-line value across graded model picks. CLV here = the no-vig
    closing probability minus the no-vig probability we bet at (positive = we
    consistently beat the closing line, the #1 predictor of long-term edge).
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT book_price, closing_price, market_novig_prob
            FROM value_bets
            WHERE confidence='MODEL_PICK'
            AND closing_price IS NOT NULL
            AND book_price IS NOT NULL
            AND detected_at >= datetime('now', ?)
        """, (f"-{int(days)} days",)).fetchall()

    def implied(price):
        price = float(price)
        return 100.0 / (price + 100.0) if price > 0 else abs(price) / (abs(price) + 100.0)

    clvs = []
    for r in rows:
        r = dict(r)
        # No-vig bet prob: prefer stored market_novig_prob, else raw implied.
        bet_p = r.get("market_novig_prob") or implied(r["book_price"])
        close_p = implied(r["closing_price"])
        clvs.append(close_p - bet_p)

    if not clvs:
        return {"avg_clv": 0.0, "count": 0, "beat_close_pct": 0.0,
                "assessment": "Insufficient data — CLV builds as picks are graded.",
                "is_sharp": False}

    avg = sum(clvs) / len(clvs)
    beat = sum(1 for c in clvs if c > 0) / len(clvs)
    return {
        "avg_clv": round(avg, 4),
        "count": len(clvs),
        "beat_close_pct": round(beat, 4),
        "assessment": (
            "Beating the close — genuine edge signal." if avg > 0.005
            else "Roughly at market — no demonstrated edge yet." if avg > -0.005
            else "Losing to the close — model is on the wrong side."
        ),
        "is_sharp": avg > 0.01 and beat > 0.5,
    }


def save_arb_opportunity(data: dict):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO arbitrage_opportunities
            (game_id, market, book_a, book_b, side_a, side_b, price_a, price_b, arb_pct, profit_pct)
            VALUES (:game_id, :market, :book_a, :book_b, :side_a, :side_b,
                    :price_a, :price_b, :arb_pct, :profit_pct)
        """, data)


def save_sharp_play(data: dict):
    with get_db() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO games (game_id, home_team, away_team, game_date, status)
            VALUES (:game_id, 'Unknown', 'Unknown', date('now'), 'scheduled')
        """, {"game_id": data.get("game_id")})
        conn.execute("""
            INSERT INTO sharp_plays
            (game_id, side, market, signal_type, signal_strength, opening_line,
             current_line, ticket_pct, money_pct, notes)
            VALUES (:game_id, :side, :market, :signal_type, :signal_strength,
                    :opening_line, :current_line, :ticket_pct, :money_pct, :notes)
        """, data)


def get_recent_value_bets(hours: int = 24):
    cutoff = (_dt.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with get_db() as conn:
        rows = conn.execute("""
            SELECT vb.*, g.home_team, g.away_team, g.game_date, g.game_time
            FROM value_bets vb
            JOIN games g ON vb.game_id = g.game_id
            WHERE vb.detected_at >= ?
            ORDER BY vb.edge DESC
        """, (cutoff,)).fetchall()
        return [dict(r) for r in rows]


def get_recent_arb_opportunities(hours: int = 6):
    cutoff = (_dt.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with get_db() as conn:
        rows = conn.execute("""
            SELECT ao.*, g.home_team, g.away_team, g.game_date
            FROM arbitrage_opportunities ao
            JOIN games g ON ao.game_id = g.game_id
            WHERE ao.detected_at >= ?
            AND ao.expired = 0
            ORDER BY ao.profit_pct DESC
        """, (cutoff,)).fetchall()
        return [dict(r) for r in rows]


def get_recent_sharp_plays(hours: int = 24):
    cutoff = (_dt.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with get_db() as conn:
        rows = conn.execute("""
            SELECT sp.*, g.home_team, g.away_team, g.game_date
            FROM sharp_plays sp
            JOIN games g ON sp.game_id = g.game_id
            WHERE sp.detected_at >= ?
            ORDER BY sp.signal_strength DESC
        """, (cutoff,)).fetchall()
        return [dict(r) for r in rows]
