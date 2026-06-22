"""Vector store — ChromaDB wrapper for semantic search."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class VectorStore:
    """ChromaDB-backed vector store for memory and session embeddings."""

    def __init__(self, persist_dir: str | Path, embedding_fn):
        if not HAS_CHROMADB:
            raise ImportError(
                "chromadb is not installed. Run: pip install synapse-agent[memory]"
            )

        self.persist_dir = Path(persist_dir).expanduser()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_fn = embedding_fn  # Function to generate embeddings

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _get_or_create(self, name: str):
        try:
            return self._client.get_collection(name)
        except Exception:
            return self._client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

    # ── Memories Collection ──

    def add_memories(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        collection = self._get_or_create("synapse_memories")
        embeddings = self.embedding_fn.embed(texts)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{}] * len(ids),
        )

    def search_memories(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        collection = self._get_or_create("synapse_memories")
        query_embedding = self.embedding_fn.embed(query)

        where = filter_metadata if filter_metadata else None
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        return [
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0,
            }
            for i in range(len(results["ids"][0]))
        ]

    def delete_memories(self, ids: list[str]) -> None:
        collection = self._get_or_create("synapse_memories")
        try:
            collection.delete(ids=ids)
        except Exception:
            pass

    def memory_count(self) -> int:
        collection = self._get_or_create("synapse_memories")
        return collection.count()

    # ── Sessions Collection ──

    def add_session(self, session_id: str, summary: str, metadata: dict | None = None) -> None:
        collection = self._get_or_create("synapse_sessions")
        embeddings = self.embedding_fn.embed(summary)
        collection.add(
            ids=[session_id],
            embeddings=embeddings,
            documents=[summary],
            metadatas=[metadata or {}],
        )

    def search_sessions(self, query: str, top_k: int = 5) -> list[dict]:
        collection = self._get_or_create("synapse_sessions")
        query_embedding = self.embedding_fn.embed(query)

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        return [
            {
                "id": results["ids"][0][i],
                "summary": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0,
            }
            for i in range(len(results["ids"][0]))
        ]

    def delete_session(self, session_id: str) -> None:
        collection = self._get_or_create("synapse_sessions")
        try:
            collection.delete(ids=[session_id])
        except Exception:
            pass
