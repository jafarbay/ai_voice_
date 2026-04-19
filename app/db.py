"""SQLite persistence layer for jobs and PII events.

Schema (see plan / docstring of :func:`init_schema`):

- ``jobs``: one row per anonymization job.
- ``events``: one row per detected PII entity, foreign-keyed to ``jobs``.

We deliberately use ``sqlite3`` from the standard library — the access
pattern is low-volume (one row per PII span) and we want zero extra
runtime dependencies.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings


def _now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db_path() -> Path:
    """Resolve database location from settings and make sure its parent exists."""
    settings = get_settings()
    path = Path(settings.database_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(path: Path | str | None = None) -> sqlite3.Connection:
    """Open a new SQLite connection with the project defaults.

    Rows are returned as ``sqlite3.Row`` (dict-like); foreign keys are
    enforced. Callers are responsible for ``conn.close()``.
    """
    db_path = Path(path).resolve() if path is not None else _db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection | None = None) -> None:
    """Create all tables if they don't exist. Safe to call repeatedly."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                input_filename TEXT,
                duration_sec REAL,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                pii_type TEXT NOT NULL,
                text TEXT NOT NULL,
                start_sec REAL,
                end_sec REAL,
                source TEXT,
                confidence REAL,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id);
            """
        )
        conn.commit()
    finally:
        if own_conn:
            conn.close()


# --- Jobs -------------------------------------------------------------------


def create_job(
    job_id: str,
    input_filename: str | None = None,
    *,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Insert a new job row with ``status='queued'``."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO jobs (id, created_at, status, input_filename) "
            "VALUES (?, ?, 'queued', ?)",
            (job_id, _now_iso(), input_filename),
        )
        conn.commit()
    finally:
        if own_conn:
            conn.close()


def update_job_status(
    job_id: str,
    status: str,
    *,
    error: str | None = None,
    duration_sec: float | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Update mutable job fields. Unset args leave the column unchanged."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        fields = ["status = ?"]
        values: list[Any] = [status]
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if duration_sec is not None:
            fields.append("duration_sec = ?")
            values.append(duration_sec)
        values.append(job_id)
        conn.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        conn.commit()
    finally:
        if own_conn:
            conn.close()


def get_job(job_id: str, *, conn: sqlite3.Connection | None = None) -> dict | None:
    """Return the job row as a dict, or ``None`` if not found."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        cur = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cur.fetchone()
        return dict(row) if row is not None else None
    finally:
        if own_conn:
            conn.close()


def list_jobs(
    limit: int = 50, *, conn: sqlite3.Connection | None = None
) -> list[dict]:
    """Return the latest ``limit`` jobs ordered by ``created_at`` DESC."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        if own_conn:
            conn.close()


# --- Events -----------------------------------------------------------------


def insert_event(
    job_id: str,
    pii_type: str,
    text: str,
    start_sec: float | None,
    end_sec: float | None,
    source: str | None,
    confidence: float | None,
    *,
    timestamp: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Insert one PII event; return the autoincrement ``events.id``."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO events (job_id, timestamp, pii_type, text, "
            "start_sec, end_sec, source, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                timestamp or _now_iso(),
                pii_type,
                text,
                start_sec,
                end_sec,
                source,
                confidence,
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)
    finally:
        if own_conn:
            conn.close()


def list_events(
    job_id: str, *, conn: sqlite3.Connection | None = None
) -> list[dict]:
    """Return all event rows for ``job_id`` ordered by id ASC."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT * FROM events WHERE job_id = ? ORDER BY id ASC",
            (job_id,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        if own_conn:
            conn.close()


__all__ = [
    "get_connection",
    "init_schema",
    "create_job",
    "update_job_status",
    "get_job",
    "list_jobs",
    "insert_event",
    "list_events",
]
