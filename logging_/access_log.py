"""Every door attempt in SQLite, including rejected spoofs (PRD 4.8).

Columns follow the PRD: `liveness_geom` is the face relief in metres and
`liveness_model` stays NULL until the learned liveness model exists. `state`
is where the state machine ended, `path` the states it walked, and `status`
the API's short answer, so a spoof and a badly framed face can be told apart.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class AccessEvent:
    ts: str
    decision: str
    status: str
    reason: str
    state: str
    path: str
    liveness_geom: float | None
    liveness_model: float | None
    match_score: float | None
    person_id: str | None
    latency_ms: float
    id: int | None = None


class AccessLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS access_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    state TEXT NOT NULL,
                    path TEXT NOT NULL,
                    liveness_geom REAL,
                    liveness_model REAL,
                    match_score REAL,
                    person_id TEXT,
                    latency_ms REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat(timespec="milliseconds")

    def record(self, event: AccessEvent) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO access_events (
                    ts, decision, status, reason, state, path, liveness_geom,
                    liveness_model, match_score, person_id, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.ts,
                    event.decision,
                    event.status,
                    event.reason,
                    event.state,
                    event.path,
                    event.liveness_geom,
                    event.liveness_model,
                    event.match_score,
                    event.person_id,
                    event.latency_ms,
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 50) -> list[AccessEvent]:
        """Newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM access_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [AccessEvent(**dict(row)) for row in rows]
