"""SQLite setup and helper functions for the ranking pipeline."""

import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


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

CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_source_extid
    ON episodes(source, external_id) WHERE external_id != '';

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

CREATE TABLE IF NOT EXISTS rag_scores (
    id INTEGER PRIMARY KEY,
    essay_id INTEGER REFERENCES essays(id),
    dimension TEXT,
    score REAL,
    reasoning TEXT,
    session_id TEXT,
    model TEXT,
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
            _now(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_essay(conn, episode_id, essay_data):
    """Insert an essay (caller should commit in batch)."""
    cur = conn.execute(
        """INSERT INTO essays (episode_id, position, text, title, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            episode_id,
            essay_data.get("position", 0),
            essay_data["text"],
            essay_data.get("title", ""),
            _now(),
        ),
    )
    return cur.lastrowid


def insert_ranking(conn, essay_id, session_id, dimension, rank, method, model):
    """Insert a ranking record (caller should commit in batch)."""
    conn.execute(
        """INSERT INTO rankings (essay_id, session_id, dimension, rank, method, model, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (essay_id, session_id, dimension, rank, method, model, _now()),
    )


def insert_rrf_score(conn, essay_id, session_id, use_case, score, rank_within_episode):
    """Insert an RRF score record (caller should commit in batch)."""
    conn.execute(
        """INSERT INTO rrf_scores (essay_id, session_id, use_case, score, rank_within_episode, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (essay_id, session_id, use_case, score, rank_within_episode, _now()),
    )


def insert_bt_score(conn, essay_id, dimension, bt_score, pass_number, session_id):
    """Insert a BT score record (caller should commit in batch)."""
    conn.execute(
        """INSERT INTO bt_scores (essay_id, dimension, bt_score, pass_number, session_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (essay_id, dimension, bt_score, pass_number, session_id, _now()),
    )


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


def episode_exists(conn, source, external_id):
    """Check if an episode already exists by source + external_id."""
    row = conn.execute(
        "SELECT id FROM episodes WHERE source = ? AND external_id = ?",
        (source, external_id),
    ).fetchone()
    return row is not None


def get_rankings_for_episode(conn, episode_id, session_id=None):
    """Check if rankings exist for essays in an episode."""
    if session_id:
        rows = conn.execute(
            """SELECT r.id FROM rankings r
               JOIN essays e ON r.essay_id = e.id
               WHERE e.episode_id = ? AND r.session_id = ?
               LIMIT 1""",
            (episode_id, session_id),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT r.id FROM rankings r
               JOIN essays e ON r.essay_id = e.id
               WHERE e.episode_id = ?
               LIMIT 1""",
            (episode_id,),
        ).fetchall()
    return len(rows) > 0


def delete_essays_for_episode(conn, episode_id):
    """Delete all essays and their dependent rankings/scores for an episode."""
    essay_ids = conn.execute(
        "SELECT id FROM essays WHERE episode_id = ?", (episode_id,)
    ).fetchall()
    if essay_ids:
        ids = [r["id"] for r in essay_ids]
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM rankings WHERE essay_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM rrf_scores WHERE essay_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM bt_scores WHERE essay_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM rag_scores WHERE essay_id IN ({placeholders})", ids)
    conn.execute("DELETE FROM essays WHERE episode_id = ?", (episode_id,))
    conn.commit()


def delete_rankings_for_session(conn, session_id):
    """Delete all ranking data for a session."""
    conn.execute("DELETE FROM rankings WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM rrf_scores WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM bt_scores WHERE session_id = ?", (session_id,))
    conn.commit()


def get_bt_scores(conn):
    """
    Fetch latest BT scores for all essays, grouped by essay and dimension.

    Returns:
        Dict of {essay_id: {dimension: bt_score}}.
    """
    rows = conn.execute(
        """SELECT essay_id, dimension, bt_score
           FROM bt_scores b1
           WHERE b1.id = (
               SELECT b2.id FROM bt_scores b2
               WHERE b2.essay_id = b1.essay_id AND b2.dimension = b1.dimension
               ORDER BY b2.created_at DESC LIMIT 1
           )"""
    ).fetchall()
    scores = {}
    for r in rows:
        eid = r["essay_id"]
        if eid not in scores:
            scores[eid] = {}
        scores[eid][r["dimension"]] = r["bt_score"]
    return scores


def get_rrf_scores(conn, use_case="default"):
    """
    Fetch latest RRF scores for a use case.

    Returns:
        Dict of {essay_id: score}.
    """
    rows = conn.execute(
        """SELECT essay_id, score
           FROM rrf_scores r1
           WHERE r1.use_case = ? AND r1.id = (
               SELECT r2.id FROM rrf_scores r2
               WHERE r2.essay_id = r1.essay_id AND r2.use_case = r1.use_case
               ORDER BY r2.created_at DESC LIMIT 1
           )""",
        (use_case,),
    ).fetchall()
    return {r["essay_id"]: r["score"] for r in rows}


def insert_rag_score(conn, essay_id, dimension, score, reasoning, session_id, model):
    """Insert a RAG score record (caller should commit in batch)."""
    conn.execute(
        """INSERT INTO rag_scores (essay_id, dimension, score, reasoning, session_id, model, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (essay_id, dimension, score, reasoning, session_id, model, _now()),
    )


def get_rag_scores(conn):
    """
    Fetch latest RAG scores for all essays, grouped by essay and dimension.

    Returns:
        Dict of {essay_id: {dimension: score}}.
    """
    rows = conn.execute(
        """SELECT essay_id, dimension, score
           FROM rag_scores r1
           WHERE r1.id = (
               SELECT r2.id FROM rag_scores r2
               WHERE r2.essay_id = r1.essay_id AND r2.dimension = r1.dimension
               ORDER BY r2.created_at DESC LIMIT 1
           )"""
    ).fetchall()
    scores = {}
    for r in rows:
        eid = r["essay_id"]
        if eid not in scores:
            scores[eid] = {}
        scores[eid][r["dimension"]] = r["score"]
    return scores


def get_rag_score_details(conn, essay_ids):
    """
    Fetch latest RAG scores with reasoning for given essay IDs.

    Returns:
        Dict of {essay_id: {dimension: {score, reasoning}}}.
    """
    if not essay_ids:
        return {}
    placeholders = ",".join("?" * len(essay_ids))
    rows = conn.execute(
        f"""SELECT essay_id, dimension, score, reasoning
            FROM rag_scores r1
            WHERE r1.essay_id IN ({placeholders}) AND r1.id = (
                SELECT r2.id FROM rag_scores r2
                WHERE r2.essay_id = r1.essay_id AND r2.dimension = r1.dimension
                ORDER BY r2.created_at DESC LIMIT 1
            )""",
        essay_ids,
    ).fetchall()
    result = {}
    for r in rows:
        eid = r["essay_id"]
        if eid not in result:
            result[eid] = {}
        result[eid][r["dimension"]] = {"score": r["score"], "reasoning": r["reasoning"]}
    return result


def essay_has_rag_scores(conn, essay_id):
    """Check if an essay already has RAG scores."""
    row = conn.execute(
        "SELECT id FROM rag_scores WHERE essay_id = ? LIMIT 1", (essay_id,)
    ).fetchone()
    return row is not None


def get_essay_details(conn, essay_ids):
    """
    Fetch essay text, title, and episode info for given IDs.

    Returns:
        List of dicts with id, title, text, episode_title, episode_source.
    """
    if not essay_ids:
        return []
    placeholders = ",".join("?" * len(essay_ids))
    rows = conn.execute(
        f"""SELECT e.id, e.title, e.text, ep.title as episode_title, ep.source
            FROM essays e
            JOIN episodes ep ON e.episode_id = ep.id
            WHERE e.id IN ({placeholders})""",
        essay_ids,
    ).fetchall()
    return [dict(r) for r in rows]
