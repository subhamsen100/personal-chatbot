"""SQLite-backed persistent chat memory."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """Stores sessions and messages in a local SQLite database."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or settings.db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    summary     TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id);
            """)

    # sessions ─

    def create_session(self, session_id: str) -> None:
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at) VALUES (?,?,?)",
                (session_id, now, now),
            )

    def list_sessions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT session_id, created_at, updated_at, summary FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def get_summary(self, session_id: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT summary FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row["summary"] if row else None

    def save_summary(self, session_id: str, summary: str) -> None:
        self.create_session(session_id)
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (summary, _now(), session_id),
            )

    # messages ─

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.create_session(session_id)
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    def get_history(self, session_id: str, last_n: int = 10) -> list[dict]:
        """Return the most recent `last_n` messages, oldest-first."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content, timestamp FROM messages
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, last_n),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def message_count(self, session_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["cnt"] if row else 0


# Module-level singleton
memory_store = MemoryStore()
