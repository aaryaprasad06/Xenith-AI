from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from api.schemas import UserRecord


class UserStore:
    """Id list only. Face embeddings are a later step and are not stored here."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def list_users(self) -> list[UserRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT user_id, created_at FROM users ORDER BY created_at, user_id"
            ).fetchall()
        return [UserRecord(user_id=row["user_id"], created_at=row["created_at"]) for row in rows]

    def enroll(self, user_id: str) -> UserRecord:
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users (user_id, created_at) VALUES (?, ?)",
                (user_id, created_at),
            )
        return UserRecord(user_id=user_id, created_at=created_at)

    def delete(self, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            return cursor.rowcount > 0
