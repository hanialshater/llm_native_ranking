"""SQLite setup and helper functions for the ranking pipeline."""

import sqlite3
from datetime import datetime


SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    source TEXT,
    external_id TEXT,
    title TEXT,
    url TEXT,
    raw_text TEXT,
    scraped_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS essays (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER REFERENCES episodes(id),
    position INTEGER,
    text TEXT,
    title TEXT,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rankings (
    id INTEGER PRIMARY KEY,
    essay_id INTEGER REFERENCES essays(id),
    session_id TEXT,
    dimension TEXT,
    rank INTEGER,
    method TEXT,
    model TEXT,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rrf_scores (
    id INTEGER PRIMARY KEY,
    essay_id INTEGER REFERENCES essays(id),
    session_id TEXT,
    use_case TEXT,
    score REAL,
    rank_within_episode INTEGER,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bt_scores (
    id INTEGER PRIMARY KEY,
    essay_id INTEGER REFERENCES essays(id),
    dimension TEXT,
    bt_score REAL,
    pass_number INTEGER,
    session_id TEXT,
    created_at TIMESTAMP
);
"""


def get_connection(db_path="ranking.db"):
    """Get a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def setup_db(conn):
    """Create all tables if they don't exist."""
    conn.executescript(SCHEMA)
    conn.commit()


def insert_episode(conn, source, data):
    """Insert an episode and return its id."""
    cur = conn.execute(
        """INSERT INTO episodes (source, external_id, title, url, raw_text, scraped_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            source,
            data.get("external_id", ""),
            data.get("title", ""),
            data.get("url", ""),
            data.get("text", ""),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_essay(conn, episode_id, essay_data):
    """Insert an essay and return its id."""
    cur = conn.execute(
        """INSERT INTO essays (episode_id, position, text, title, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            episode_id,
            essay_data.get("position", 0),
            essay_data["text"],
            essay_data.get("title", ""),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_ranking(conn, essay_id, session_id, dimension, rank, method, model):
    """Insert a ranking record."""
    conn.execute(
        """INSERT INTO rankings (essay_id, session_id, dimension, rank, method, model, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (essay_id, session_id, dimension, rank, method, model, datetime.utcnow().isoformat()),
    )
    conn.commit()


def insert_rrf_score(conn, essay_id, session_id, use_case, score, rank_within_episode):
    """Insert an RRF score record."""
    conn.execute(
        """INSERT INTO rrf_scores (essay_id, session_id, use_case, score, rank_within_episode, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (essay_id, session_id, use_case, score, rank_within_episode, datetime.utcnow().isoformat()),
    )
    conn.commit()


def insert_bt_score(conn, essay_id, dimension, bt_score, pass_number, session_id):
    """Insert a BT score record."""
    conn.execute(
        """INSERT INTO bt_scores (essay_id, dimension, bt_score, pass_number, session_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (essay_id, dimension, bt_score, pass_number, session_id, datetime.utcnow().isoformat()),
    )
    conn.commit()


def get_essays_for_episode(conn, episode_id):
    """Get all essays for an episode."""
    rows = conn.execute(
        "SELECT id, position, text, title FROM essays WHERE episode_id = ? ORDER BY position",
        (episode_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_essays(conn):
    """Get all essays across all episodes."""
    rows = conn.execute(
        "SELECT id, episode_id, position, text, title FROM essays ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_episodes(conn, source=None):
    """Get episodes, optionally filtered by source."""
    if source:
        rows = conn.execute(
            "SELECT * FROM episodes WHERE source = ? ORDER BY id", (source,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM episodes ORDER BY id").fetchall()
    return [dict(r) for r in rows]
