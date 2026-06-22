"""Retriever — multi-route recall with keyword, vector, and recency scoring."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from synapse.config.schema import MemoryConfig
from synapse.memory.schemas import Memory, MemoryCategory
from synapse.memory.store import MemoryStore
from synapse.memory.vector_store import VectorStore


def _recency_score(memory: Memory, halflife_days: int = 30) -> float:
    """Compute a recency score with exponential decay.

    Score ranges from 0 (ancient) to 1 (just now).
    """
    try:
        created = datetime.fromisoformat(memory.created_at)
        age_days = (datetime.now() - created).total_seconds() / 86400
        return math.exp(-math.log(2) * age_days / halflife_days)
    except (ValueError, TypeError):
        return 0.5


class Retriever:
    """Multi-route memory retrieval with optional LLM reranking."""

    def __init__(
        self,
        store: MemoryStore,
        vector_store: VectorStore,
        config: MemoryConfig,
        llm_reranker=None,  # Optional async function (query, candidates) -> ranked_ids
    ):
        self.store = store
        self.vector_store = vector_store
        self.config = config
        self.llm_reranker = llm_reranker

    async def recall(
        self,
        query: str,
        top_k: int | None = None,
        category: MemoryCategory | None = None,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        """Retrieve the most relevant memories for a query."""
        final_k = top_k or self.config.final_top_k

        # Route 1: Vector search
        vector_filter = None
        if category:
            vector_filter = {"category": category.value}

        vector_results = self.vector_store.search_memories(
            query=query,
            top_k=self.config.vector_top_k,
            filter_metadata=vector_filter,
        )

        # Route 2: Keyword search
        keyword_results = self.store.search_memories_keyword(
            query=query,
            limit=self.config.keyword_top_k,
        )

        # Merge and deduplicate
        seen_ids: set[str] = set()
        all_candidates: list[tuple[Memory, float]] = []

        # Add vector results with similarity scores
        for vr in vector_results:
            mem_id = vr["id"]
            if mem_id in seen_ids:
                continue
            seen_ids.add(mem_id)

            mem = self.store.get_memory(mem_id)
            if mem is None:
                continue
            if mem.is_expired:
                continue
            if mem.importance < min_importance:
                continue
            if category and mem.category != category:
                continue

            # Convert distance to similarity (cosine distance → 0=identical, 2=opposite)
            sim_score = 1.0 - (vr.get("distance", 0) / 2.0)
            sim_score = max(0.0, min(1.0, sim_score))
            all_candidates.append((mem, sim_score))

        # Add keyword results with lower base score
        for mem in keyword_results:
            if mem.id in seen_ids:
                continue
            seen_ids.add(mem.id)
            if mem.is_expired:
                continue
            if mem.importance < min_importance:
                continue
            if category and mem.category != category:
                continue
            all_candidates.append((mem, 0.5))

        # Score and rank
        scored = []
        for mem, sim_score in all_candidates:
            recency = _recency_score(mem, self.config.recency_halflife_days)
            combined = (
                0.6 * sim_score
                + 0.2 * recency
                + 0.2 * mem.importance
            )
            scored.append((mem, combined))

        scored.sort(key=lambda x: x[1], reverse=True)

        # LLM reranking (if enabled and more candidates than needed)
        if (
            self.config.rerank_enabled
            and self.llm_reranker
            and len(scored) > final_k
        ):
            candidates = [m for m, _ in scored[: self.config.rerank_candidate_n]]
            ranked_ids = await self.llm_reranker(query, candidates)
            # Reorder based on LLM ranking
            id_to_mem = {m.id: m for m, _ in scored}
            scored = [
                (id_to_mem[mid], 1.0 - i / len(ranked_ids))
                for i, mid in enumerate(ranked_ids)
                if mid in id_to_mem
            ]

        # Return top-k and touch them
        result = []
        for mem, _ in scored[:final_k]:
            self.store.touch_memory(mem.id)
            result.append(mem)

        return result
