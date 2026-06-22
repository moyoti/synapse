"""SQLite store for sessions, memories, and facts."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from synapse.memory.schemas import Fact, Memory, MemoryCategory, Session


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    summary       TEXT,
    message_count INTEGER DEFAULT 0,
    tags          TEXT DEFAULT '[]',
    model_used    TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    category        TEXT NOT NULL,
    importance      REAL DEFAULT 0.5,
    source_session  TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed   TEXT NOT NULL DEFAULT (datetime('now')),
    access_count    INTEGER DEFAULT 0,
    ttl             TEXT,
    metadata        TEXT DEFAULT '{}',
    FOREIGN KEY (source_session) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS facts (
    id          TEXT PRIMARY KEY,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    namespace   TEXT DEFAULT 'global',
    confidence  REAL DEFAULT 1.0,
    source      TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(key, namespace)
);

CREATE TABLE IF NOT EXISTS memory_links (
    source_id  TEXT NOT NULL,
    target_id  TEXT NOT NULL,
    relation   TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, relation),
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_memories_ttl ON memories(ttl);
CREATE INDEX IF NOT EXISTS idx_facts_namespace ON facts(namespace);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);
"""


class MemoryStore:
    """SQLite-backed store for structured memory data."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Sessions ──

    def save_session(self, session: Session) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (id, title, created_at, updated_at, summary, message_count, tags, model_used)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session.id, session.title, session.created_at, session.updated_at,
                 session.summary, session.message_count,
                 __import__("json").dumps(session.tags), session.model_used),
            )

    def get_session(self, session_id: str) -> Session | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row:
                return Session.from_row(dict(row))
        return None

    def list_sessions(self, limit: int = 20) -> list[Session]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [Session.from_row(dict(r)) for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # ── Memories ──

    def add_memory(self, memory: Memory) -> str:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memories
                   (id, content, category, importance, source_session,
                    created_at, last_accessed, access_count, ttl, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(memory.to_row().values()),
            )
        return memory.id

    def get_memory(self, memory_id: str) -> Memory | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row:
                return Memory.from_row(dict(row))
        return None

    def list_memories(
        self,
        category: MemoryCategory | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        with self._connect() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE category = ? ORDER BY last_accessed DESC LIMIT ? OFFSET ?",
                    (category.value, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY last_accessed DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            return [Memory.from_row(dict(r)) for r in rows]

    def search_memories_keyword(self, query: str, limit: int = 20) -> list[Memory]:
        """Simple keyword search using SQLite LIKE."""
        with self._connect() as conn:
            pattern = f"%{query}%"
            rows = conn.execute(
                "SELECT * FROM memories WHERE content LIKE ? ORDER BY importance DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()
            return [Memory.from_row(dict(r)) for r in rows]

    def touch_memory(self, memory_id: str) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                (now, memory_id),
            )

    def delete_memory(self, memory_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.execute("DELETE FROM memory_links WHERE source_id = ? OR target_id = ?",
                         (memory_id, memory_id))

    def delete_memories_by_category(self, category: MemoryCategory) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE category = ?", (category.value,))
            return cursor.rowcount

    def delete_expired(self) -> int:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE ttl IS NOT NULL AND ttl < ?", (now,)
            )
            return cursor.rowcount

    def get_stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_cat = conn.execute(
                "SELECT category, COUNT(*) FROM memories GROUP BY category"
            ).fetchall()
            sessions_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            facts_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

        return {
            "total_memories": total,
            "by_category": {row[0]: row[1] for row in by_cat},
            "total_sessions": sessions_count,
            "total_facts": facts_count,
        }

    # ── Facts ──

    def upsert_fact(self, fact: Fact) -> str:
        fact.updated_at = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO facts (id, key, value, namespace, confidence, source, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(key, namespace) DO UPDATE SET
                   value = excluded.value, confidence = excluded.confidence,
                   source = excluded.source, updated_at = excluded.updated_at""",
                tuple(fact.to_row().values()),
            )
        return fact.id

    def get_facts(self, namespace: str = "global") -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM facts WHERE namespace = ?", (namespace,)
            ).fetchall()
            return {row["key"]: row["value"] for row in rows}

    def get_all_facts(self) -> list[Fact]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM facts ORDER BY namespace, key").fetchall()
            return [Fact.from_row(dict(r)) for r in rows]

    def delete_fact(self, key: str, namespace: str = "global") -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM facts WHERE key = ? AND namespace = ?", (key, namespace))

    # ── Links ──

    def link_memories(self, source_id: str, target_id: str, relation: str = "related") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO memory_links (source_id, target_id, relation) VALUES (?, ?, ?)",
                (source_id, target_id, relation),
            )
