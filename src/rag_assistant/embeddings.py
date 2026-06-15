"""Embedding wrapper around sentence-transformers.

The model is loaded lazily on first use so importing this module is cheap and does
not require the model weights to be present. Texts are embedded in batches and
returned as a single ``float32`` NumPy array. Vectors are L2-normalized, which makes
cosine similarity equivalent to a dot product in the vector store.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import settings


class Embedder:
    """Lazy, batched sentence-transformers embedder.

    Args:
        model_name: sentence-transformers model id. Defaults to the configured
            ``EMBEDDING_MODEL``.
        normalize: If True (default), L2-normalize embeddings so cosine == dot.
        batch_size: Number of texts encoded per forward pass.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = None  # loaded on first use

    def _ensure_model(self):
        """Load the underlying model on first call (deferred heavy import)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        """Embedding dimensionality reported by the loaded model."""
        model = self._ensure_model()
        return int(model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: Texts to embed.

        Returns:
            A ``(len(texts), dim)`` ``float32`` array. Returns an empty
            ``(0, 0)`` array if ``texts`` is empty.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string, returning a 1-D ``(dim,)`` array."""
        return self.embed_texts([query])[0]
