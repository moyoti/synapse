"""Memory Agent — the central memory management subsystem for Synapse.

If ChromaDB or sentence-transformers are not installed, the MemoryAgent
degrades gracefully — it can still be imported and used for SQLite-only
storage (facts, sessions, keyword search) but vector search is unavailable.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from synapse.config.schema import MemoryConfig, SynapseConfig
from synapse.memory.compactor import Compactor
from synapse.memory.schemas import Fact, Memory, MemoryCategory, Session
from synapse.memory.store import MemoryStore

# Optional imports for vector search
VECTOR_AVAILABLE = False
VectorStore = None
EmbeddingModel = None
Retriever = None

try:
    from synapse.memory.vector_store import VectorStore
    VECTOR_AVAILABLE = True
except ImportError:
    pass

try:
    from synapse.memory.embedding import EmbeddingModel
except ImportError:
    pass

try:
    from synapse.memory.retriever import Retriever
except ImportError:
    pass


class MemoryAgent:
    """Central memory management agent.

    Handles:
    - Storage: remember(), upsert_fact()
    - Retrieval: recall(), get_facts(), recent_sessions()
    - Compression: summarize_session(), extract_facts(), compact()
    - Maintenance: consolidate(), forget()
    - Injection: bootstrap_context()
    """

    def __init__(
        self,
        config: SynapseConfig,
        user_id: str = "default",
        embedding_model: EmbeddingModel | None = None,
        provider_factory: Callable | None = None,
    ):
        self.config = config
        self.memory_config = config.memory
        self.user_id = user_id

        # Resolve store path
        store_dir = Path(self.memory_config.store_dir).expanduser()
        self.store = MemoryStore(store_dir / "synapse.db")

        # Embedding model
        if embedding_model:
            self.embedding = embedding_model
        elif EmbeddingModel:
            self.embedding = EmbeddingModel(
                model_name=config.embedding.model
            )
        else:
            self.embedding = None

        # Vector store
        if VectorStore and self.embedding:
            self.vector_store = VectorStore(
                persist_dir=store_dir / "chroma",
                embedding_fn=self.embedding,
            )
        else:
            self.vector_store = None

        # Retriever (LLM reranker can be added later)
        if Retriever and self.vector_store:
            self.retriever = Retriever(
                store=self.store,
                vector_store=self.vector_store,
                config=self.memory_config,
            )
        else:
            self.retriever = None

        # LLM provider for compaction/reranking
        self._provider_factory = provider_factory
        self._provider = None

        # Compactor
        self.compactor = Compactor(provider=None)

    async def _ensure_initialized(self):
        if self._initialized:
            return
        # Pre-warm: load the embedding model in background
        # Don't block on this — first call will lazy-load
        self._initialized = True

    def _get_provider_fn(self):
        """Get the provider callable for compaction tasks."""
        if self.compactor._provider_fn is not None:
            return self.compactor._provider_fn

        # Default: use the first available model from config
        if self._provider_factory:
            self._provider = self._provider_factory()
        else:
            from synapse.models.registry import get_provider_for_model
            try:
                # Try orchestrator model first, then first available
                orch_role = self.config.roles.get("orchestrator")
                if orch_role and orch_role.model in self.config.models:
                    model_config = self.config.models[orch_role.model]
                else:
                    model_config = next(iter(self.config.models.values()))
                self._provider = get_provider_for_model(model_config)
            except (StopIteration, Exception):
                self._provider = None

        if self._provider:
            self.compactor._provider_fn = self._make_compact_fn(self._provider)

        return self.compactor._provider_fn

    @staticmethod
    def _make_compact_fn(provider):
        """Create an async function wrapper for the provider's chat method."""
        async def _fn(messages, temperature, max_tokens):
            return await provider.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return _fn

    # ── Storage ──

    async def remember(
        self,
        content: str,
        category: MemoryCategory = MemoryCategory.FACT,
        importance: float | None = None,
        ttl: timedelta | None = None,
        source_session: str | None = None,
    ) -> str:
        """Store a new memory in both SQLite and vector store."""
        await self._ensure_initialized()

        mem_id = f"mem_{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        if importance is None:
            importance = 0.5

        ttl_str = None
        if ttl:
            ttl_str = (now + ttl).isoformat()

        memory = Memory(
            id=mem_id,
            content=content,
            category=category,
            importance=importance,
            source_session=source_session,
            created_at=now.isoformat(),
            last_accessed=now.isoformat(),
            ttl=ttl_str,
        )

        # Store in SQLite
        self.store.add_memory(memory)

        # Store in vector DB
        if self.vector_store:
            try:
                self.vector_store.add_memories(
                    ids=[mem_id],
                    texts=[content],
                    metadatas=[{"category": category.value, "importance": importance}],
                )
            except Exception:
                pass

        return mem_id

    async def upsert_fact(
        self,
        key: str,
        value: str,
        namespace: str | None = None,
        confidence: float = 1.0,
        source: str | None = None,
    ) -> str:
        """Store or update a persistent fact."""
        ns = namespace or self.user_id
        fact_id = f"fact_{uuid.uuid4().hex[:8]}"
        fact = Fact(
            id=fact_id,
            key=key,
            value=value,
            namespace=ns,
            confidence=confidence,
            source=source,
        )
        self.store.upsert_fact(fact)
        return fact_id

    # ── Retrieval ──

    async def recall(
        self,
        query: str,
        top_k: int | None = None,
        category: MemoryCategory | None = None,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        """Retrieve the most relevant memories for a query.

        If vector store is unavailable, falls back to keyword search.
        """
        await self._ensure_initialized()

        if self.retriever:
            return await self.retriever.recall(
                query=query,
                top_k=top_k,
                category=category,
                min_importance=min_importance,
            )

        # Fallback: keyword search only
        return self.store.search_memories_keyword(
            query=query,
            limit=top_k or self.memory_config.final_top_k,
        )

    def get_facts(self, namespace: str | None = None) -> dict[str, str]:
        """Get all persistent facts for a namespace."""
        return self.store.get_facts(namespace or self.user_id)

    def recent_sessions(self, top_k: int = 3) -> list[Session]:
        """Get the most recent session summaries."""
        return self.store.list_sessions(limit=top_k)

    # ── Compression ──

    async def summarize_session(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Generate a summary of a conversation."""
        self._get_provider_fn()  # Ensure provider is ready
        return await self.compactor.summarize(messages)

    async def extract_facts(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict]:
        """Extract persistent facts from a conversation."""
        self._get_provider_fn()
        return await self.compactor.extract_facts(messages, self.user_id)

    async def compact(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        title: str = "",
    ) -> Session:
        """End-of-session compaction: summarize + extract facts + save."""
        await self._ensure_initialized()
        self._get_provider_fn()

        now = datetime.now()

        # 1. Summarize
        summary = await self.summarize_session(messages)

        # 2. Extract facts
        facts_data = await self.extract_facts(messages)
        for fd in facts_data:
            await self.upsert_fact(
                key=fd.get("key", ""),
                value=str(fd.get("value", "")),
                namespace=fd.get("namespace", self.user_id),
                confidence=fd.get("confidence", 1.0),
            )

        # 3. Save session
        session = Session(
            id=session_id,
            title=title,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            summary=summary,
            message_count=len(messages),
        )
        self.store.save_session(session)

        # 4. Index session in vector store
        if summary and self.vector_store:
            try:
                self.vector_store.add_session(
                    session_id=session_id,
                    summary=summary,
                    metadata={"title": title, "created_at": session.created_at},
                )
            except Exception:
                pass

        return session

    # ── Context Injection ──

    async def bootstrap_context(self, user_input: str) -> str:
        """Build memory context to inject into the system prompt.

        Called at the start of each new conversation.
        """
        await self._ensure_initialized()

        parts = []

        # Relevant memories
        memories = await self.recall(user_input, top_k=5)
        if memories:
            parts.append("## 相关记忆 (Relevant Memories)")
            for m in memories:
                cat_label = m.category.value
                parts.append(f"- [{m.created_at[:10]} | {cat_label}] {m.content}")

        # Persistent facts
        facts = self.get_facts()
        if facts:
            parts.append("\n## 用户事实 (User Facts)")
            for key, value in facts.items():
                parts.append(f"- {key}: {value}")

        # Recent sessions
        sessions = self.recent_sessions(top_k=3)
        if sessions:
            parts.append("\n## 最近会话 (Recent Sessions)")
            for s in sessions:
                parts.append(f"- {s.created_at[:10]}: {s.title or s.summary or '(no title)'}")

        return "\n".join(parts) if parts else ""

    # ── Maintenance ──

    async def consolidate(self):
        """Cleanup expired memories."""
        deleted = self.store.delete_expired()
        return deleted

    async def forget(
        self,
        memory_id: str | None = None,
        category: MemoryCategory | None = None,
    ) -> int:
        """Delete memories by ID or category."""
        if memory_id:
            if self.vector_store:
                self.vector_store.delete_memories([memory_id])
            self.store.delete_memory(memory_id)
            return 1
        if category:
            count = self.store.delete_memories_by_category(category)
            return count
        return 0

    def stats(self) -> dict:
        """Get memory system statistics."""
        stats = self.store.get_stats()
        stats["vector_available"] = VECTOR_AVAILABLE
        stats["vector_memories"] = self.vector_store.memory_count() if self.vector_store else 0
        return stats
