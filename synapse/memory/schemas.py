"""Memory data schemas — data models for the memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class MemoryCategory(str, Enum):
    FACT = "fact"
    PREFERENCE = "pref"
    DECISION = "decision"
    KNOWLEDGE = "knowledge"
    EVENT = "event"
    RELATION = "relation"


@dataclass
class Memory:
    """A single memory entry."""

    id: str
    content: str
    category: MemoryCategory = MemoryCategory.FACT
    importance: float = 0.5
    source_session: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    ttl: str | None = None  # ISO timestamp or None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if not self.ttl:
            return False
        return datetime.now() > datetime.fromisoformat(self.ttl)

    def to_row(self) -> dict:
        import json
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category.value,
            "importance": self.importance,
            "source_session": self.source_session,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "ttl": self.ttl,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_row(cls, row: dict) -> Memory:
        import json
        return cls(
            id=row["id"],
            content=row["content"],
            category=MemoryCategory(row["category"]),
            importance=row["importance"],
            source_session=row["source_session"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            ttl=row["ttl"],
            metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
        )


@dataclass
class Fact:
    """A persistent key-value fact extracted from conversations."""

    id: str
    key: str
    value: str
    namespace: str = "global"
    confidence: float = 1.0
    source: str | None = None  # memory id
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "namespace": self.namespace,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: dict) -> Fact:
        return cls(
            id=row["id"],
            key=row["key"],
            value=row["value"],
            namespace=row["namespace"],
            confidence=row["confidence"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass
class Session:
    """A conversation session record."""

    id: str
    title: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    summary: str | None = None
    message_count: int = 0
    tags: list[str] = field(default_factory=list)
    model_used: str = ""

    def to_row(self) -> dict:
        import json
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "message_count": self.message_count,
            "tags": json.dumps(self.tags),
            "model_used": self.model_used,
        }

    @classmethod
    def from_row(cls, row: dict) -> Session:
        import json
        return cls(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            summary=row["summary"],
            message_count=row["message_count"],
            tags=json.loads(row["tags"]) if row.get("tags") else [],
            model_used=row.get("model_used", ""),
        )
