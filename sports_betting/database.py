import sqlite3
import json
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from .config import DB_PATH

logger = logging.getLogger(__name__)


@contextmanager
def get_db():
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

            CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_snapshots(game_id, snapshot_time);
            CREATE INDEX IF NOT EXISTS idx_movements_game ON line_movements(game_id, movement_time);
            CREATE INDEX IF NOT EXISTS idx_value_bets_detected ON value_bets(detected_at);
            CREATE INDEX IF NOT EXISTS idx_arb_detected ON arbitrage_opportunities(detected_at, expired);
            CREATE INDEX IF NOT EXISTS idx_team_stats ON team_stats(team_name, season);
            CREATE INDEX IF NOT EXISTS idx_pitcher_stats ON pitcher_stats(player_id, season);
        """)
    logger.info("Database initialized at %s", DB_PATH)


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
        # Ensure a parent game row exists before inserting odds (FK constraint)
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
    with get_db() as conn:
        rows = conn.execute("""
            SELECT vb.*, g.home_team, g.away_team, g.game_date, g.game_time
            FROM value_bets vb
            JOIN games g ON vb.game_id = g.game_id
            WHERE vb.detected_at >= datetime('now', ? || ' hours')
            ORDER BY vb.edge DESC
        """, (f"-{hours}",)).fetchall()
        return [dict(r) for r in rows]


def get_recent_arb_opportunities(hours: int = 6):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT ao.*, g.home_team, g.away_team, g.game_date
            FROM arbitrage_opportunities ao
            JOIN games g ON ao.game_id = g.game_id
            WHERE ao.detected_at >= datetime('now', ? || ' hours')
            AND ao.expired = 0
            ORDER BY ao.profit_pct DESC
        """, (f"-{hours}",)).fetchall()
        return [dict(r) for r in rows]


def get_recent_sharp_plays(hours: int = 24):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT sp.*, g.home_team, g.away_team, g.game_date
            FROM sharp_plays sp
            JOIN games g ON sp.game_id = g.game_id
            WHERE sp.detected_at >= datetime('now', ? || ' hours')
            ORDER BY sp.signal_strength DESC
        """, (f"-{hours}",)).fetchall()
        return [dict(r) for r in rows]
