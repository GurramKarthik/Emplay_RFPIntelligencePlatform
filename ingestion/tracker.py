"""
ingestion/tracker.py
--------------------
SQLite-based parse job tracker.

Tracks the parse status of every file in a bid folder:
  PENDING → IN_PROGRESS → DONE
                        → FAILED  (after MAX_RETRIES attempts)

This lets us:
  - Resume failed files without re-parsing successful ones
  - Report exactly which files failed and why
  - Keep a persistent audit trail across sessions
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Optional

from config import TRACKER_DB_PATH
from constants import STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_DONE, STATUS_FAILED


def _get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite tracker database and return a connection."""
    os.makedirs(os.path.dirname(TRACKER_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(TRACKER_DB_PATH)
    conn.row_factory = sqlite3.Row   # allows dict-like row access
    return conn


def init_db() -> None:
    """Create the parse_jobs table if it does not already exist."""
    conn = _get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parse_jobs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                bid_id      TEXT NOT NULL,
                file_name   TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'PENDING',
                attempts    INTEGER NOT NULL DEFAULT 0,
                last_error  TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now')),
                UNIQUE(bid_id, file_name)
            )
        """)
    conn.close()


def register_file(bid_id: str, file_name: str, file_path: str) -> None:
    """
    Register a file for tracking.
    If the file is already registered (e.g. from a previous run), skip insertion
    so we preserve its existing status and attempt count.
    """
    conn = _get_connection()
    with conn:
        conn.execute("""
            INSERT OR IGNORE INTO parse_jobs (bid_id, file_name, file_path, status, attempts)
            VALUES (?, ?, ?, ?, 0)
        """, (bid_id, file_name, file_path, STATUS_PENDING))
    conn.close()


def mark_in_progress(bid_id: str, file_name: str) -> None:
    """Mark a file as currently being parsed."""
    _update_status(bid_id, file_name, STATUS_IN_PROGRESS)


def mark_done(bid_id: str, file_name: str) -> None:
    """Mark a file as successfully parsed."""
    _update_status(bid_id, file_name, STATUS_DONE)


def record_failure(bid_id: str, file_name: str, error: str) -> int:
    """
    Increment the attempt counter and record the error message.
    Returns the new attempt count so the caller can decide whether to retry.
    """
    conn = _get_connection()
    with conn:
        conn.execute("""
            UPDATE parse_jobs
            SET attempts   = attempts + 1,
                last_error = ?,
                status     = ?,
                updated_at = ?
            WHERE bid_id = ? AND file_name = ?
        """, (error, STATUS_IN_PROGRESS, datetime.utcnow().isoformat(), bid_id, file_name))

    # Return updated attempt count
    row = get_job(bid_id, file_name)
    conn.close()
    return row["attempts"] if row else 0


def mark_failed(bid_id: str, file_name: str) -> None:
    """Mark a file as permanently failed (attempts exhausted)."""
    _update_status(bid_id, file_name, STATUS_FAILED)


def get_job(bid_id: str, file_name: str) -> Optional[sqlite3.Row]:
    """Fetch the tracker row for a specific file."""
    conn = _get_connection()
    row = conn.execute("""
        SELECT * FROM parse_jobs WHERE bid_id = ? AND file_name = ?
    """, (bid_id, file_name)).fetchone()
    conn.close()
    return row


def get_failed_files(bid_id: str) -> List[sqlite3.Row]:
    """Return all files for a bid that are in FAILED status."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT * FROM parse_jobs WHERE bid_id = ? AND status = ?
    """, (bid_id, STATUS_FAILED)).fetchall()
    conn.close()
    return rows


def get_done_files(bid_id: str) -> List[sqlite3.Row]:
    """Return all files for a bid that parsed successfully."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT * FROM parse_jobs WHERE bid_id = ? AND status = ?
    """, (bid_id, STATUS_DONE)).fetchall()
    conn.close()
    return rows


def reset_bid(bid_id: str) -> int:
    """
    Delete all tracker entries for a given bid_id.
    Use this before re-running a bid to avoid stale entries from wrong previous runs.
    Returns the number of rows deleted.
    """
    conn = _get_connection()
    with conn:
        cursor = conn.execute(
            "DELETE FROM parse_jobs WHERE bid_id = ?", (bid_id,)
        )
        deleted = cursor.rowcount
    conn.close()
    return deleted


def _update_status(bid_id: str, file_name: str, status: str) -> None:
    """Internal helper: update the status column for a given file."""
    conn = _get_connection()
    with conn:
        conn.execute("""
            UPDATE parse_jobs
            SET status = ?, updated_at = ?
            WHERE bid_id = ? AND file_name = ?
        """, (status, datetime.utcnow().isoformat(), bid_id, file_name))
    conn.close()
