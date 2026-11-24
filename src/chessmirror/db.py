"""SQLite storage.

Tables
------
games             original games (full PGN text kept verbatim) + metadata + split
positions         one row per ply; ``is_player`` marks the tracked player's moves.
                  Opponent rows exist only to build the *population* prior and
                  baselines -- they are never treated as the player's moves.
analysis_cache    Stockfish results keyed by (position, settings)
models            versioned move models with the exact train/test game ids
elo_games         games used by the Elo test / rating estimator
training_items    spaced-repetition exercises generated from the player's mistakes
training_reviews  answer history for training items
weaknesses        recurring-weakness snapshots (recomputed from positions)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY,
    game_hash TEXT UNIQUE NOT NULL,
    source TEXT,
    platform TEXT,
    site TEXT,
    event TEXT,
    date TEXT,
    utc_datetime TEXT,
    white TEXT,
    black TEXT,
    white_elo INTEGER,
    black_elo INTEGER,
    player_color TEXT,           -- 'white' | 'black' | NULL (player not in game)
    opponent TEXT,
    opponent_elo INTEGER,
    player_elo INTEGER,
    result TEXT,
    player_score REAL,           -- 1 / 0.5 / 0 from the player's side
    time_control TEXT,
    tc_base INTEGER,
    tc_inc INTEGER,
    tc_class TEXT,
    eco TEXT,
    opening TEXT,
    termination TEXT,
    variant TEXT,
    rated INTEGER,
    opponent_is_engine INTEGER DEFAULT 0,
    num_plies INTEGER,
    split TEXT,                  -- 'train' | 'test'  (assigned once, never changes)
    pgn TEXT NOT NULL,
    imported_at TEXT,
    analyzed_at TEXT,
    analysis_key TEXT
);
CREATE INDEX IF NOT EXISTS games_player ON games(player_color);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY,
    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    ply INTEGER NOT NULL,
    move_number INTEGER,
    side TEXT,
    is_player INTEGER NOT NULL,
    fen TEXT NOT NULL,
    move_uci TEXT,
    move_san TEXT,
    prev_move_uci TEXT,
    clock_before REAL,
    clock_after REAL,
    time_spent REAL,
    opp_clock REAL,
    -- analysis (mover's point of view)
    eval_before_cp INTEGER,
    mate_before INTEGER,
    eval_after_cp INTEGER,
    mate_after INTEGER,
    wp_before REAL,
    wp_after REAL,
    wp_loss REAL,
    cp_loss INTEGER,
    accuracy REAL,
    best_uci TEXT,
    best_san TEXT,
    played_rank INTEGER,
    top_moves TEXT,              -- JSON [{uci, san, cp, mate, wp, pv}]
    broad_moves TEXT,            -- JSON {uci: cp} shallow eval of every legal move
    played_pv TEXT,              -- JSON [uci...] engine line after the played move
    classification TEXT,
    phase TEXT,
    features TEXT,               -- JSON position features
    motifs TEXT,                 -- JSON motif annotations
    UNIQUE(game_id, ply)
);
CREATE INDEX IF NOT EXISTS positions_player ON positions(is_player);
CREATE INDEX IF NOT EXISTS positions_game ON positions(game_id);

CREATE TABLE IF NOT EXISTS analysis_cache (
    key TEXT PRIMARY KEY,
    result TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY,
    version TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT,
    params TEXT,
    artifact_path TEXT,
    feature_version TEXT,
    train_game_ids TEXT,
    test_game_ids TEXT,
    population_game_ids TEXT,
    metrics TEXT,
    notes TEXT,
    is_active INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS elo_games (
    id INTEGER PRIMARY KEY,
    game_id INTEGER REFERENCES games(id),
    played_at TEXT,
    scale TEXT NOT NULL,         -- rating scale, e.g. 'stockfish-uci', 'lichess:blitz'
    opponent TEXT,
    opponent_rating REAL,
    result REAL,                 -- player's score; NULL while pending
    status TEXT,                 -- 'pending' | 'final'
    time_control TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS training_items (
    id INTEGER PRIMARY KEY,
    position_id INTEGER UNIQUE REFERENCES positions(id),
    game_id INTEGER,
    fen TEXT NOT NULL,
    category TEXT NOT NULL,
    solutions TEXT NOT NULL,     -- JSON [uci...] moves accepted as correct
    best_san TEXT,
    played_san TEXT,
    prompt TEXT,
    explanation TEXT,
    motif TEXT,
    created_at TEXT,
    ease REAL DEFAULT 2.5,
    interval_days REAL DEFAULT 0,
    reps INTEGER DEFAULT 0,
    lapses INTEGER DEFAULT 0,
    due_at TEXT,
    last_reviewed_at TEXT,
    ever_solved INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS training_reviews (
    id INTEGER PRIMARY KEY,
    item_id INTEGER REFERENCES training_items(id) ON DELETE CASCADE,
    reviewed_at TEXT,
    answer_uci TEXT,
    correct INTEGER,
    seconds REAL
);

CREATE TABLE IF NOT EXISTS weaknesses (
    key TEXT PRIMARY KEY,
    category TEXT,
    label TEXT,
    data TEXT,                   -- JSON evidence snapshot
    updated_at TEXT
);
"""


class DB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def q(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: tuple | dict = ()) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        return None if row is None else row[0]

    # --- analysis cache -------------------------------------------------
    def cache_get(self, key: str) -> dict | None:
        row = self.one("SELECT result FROM analysis_cache WHERE key = ?", (key,))
        return json.loads(row[0]) if row else None

    def cache_put(self, key: str, result: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO analysis_cache(key, result) VALUES (?, ?)", (key, json.dumps(result))
        )


def jload(s: str | None, default: Any = None) -> Any:
    if s is None or s == "":
        return default
    return json.loads(s)


def open_db(cfg) -> DB:
    cfg.ensure_dirs()
    return DB(cfg.db_path)
