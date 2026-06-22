"""Embedding model — local sentence-transformers with fallback."""

from __future__ import annotations

import warnings
from typing import Sequence


class EmbeddingModel:
    """Wrapper around sentence-transformers for local embeddings.

    Lazy-loads the model on first use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = SentenceTransformer(
                    self.model_name,
                    device="cpu",
                )
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install synapse-agent[memory]"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}': {e}. "
                "Try installing with: pip install sentence-transformers"
            )

    def embed(self, texts: str | Sequence[str]) -> list[list[float]]:
        """Generate embeddings for one or more texts.

        Returns a list of embedding vectors (list of floats).
        """
        self._load()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            list(texts),
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Convert numpy arrays to Python lists
        import numpy as np
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]

    @property
    def dims(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()
