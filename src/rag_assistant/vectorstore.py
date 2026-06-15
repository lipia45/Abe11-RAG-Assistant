"""Qdrant vector store wrapper (local embedded mode for dev, server URL for prod).

``QdrantStore`` wraps ``qdrant-client`` with the small surface this project needs:
create a cosine-distance collection, upsert vectors with payloads, and search.

In dev it uses embedded Qdrant via ``QdrantClient(path=...)`` (no Docker). If a
server URL is configured, it connects to that instead, without any other code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .config import settings


@dataclass(frozen=True)
class SearchHit:
    """A single search result.

    Attributes:
        id: Point id of the matched chunk.
        score: Similarity score (cosine; higher is more similar).
        payload: Stored metadata for the point (e.g. text, doc_id).
    """

    id: Union[str, int]
    score: float
    payload: Dict[str, Any]


class QdrantStore:
    """Thin wrapper over qdrant-client for a single collection.

    Args:
        collection: Collection name. Defaults to the configured collection.
        dim: Vector dimensionality. Defaults to the configured embedding dim.
        path: Local storage dir for embedded mode. Defaults to ``QDRANT_PATH``.
        url: Qdrant server URL. If set, overrides ``path`` (server mode).
    """

    def __init__(
        self,
        collection: Optional[str] = None,
        dim: Optional[int] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        from qdrant_client import QdrantClient

        self.collection = collection or settings.qdrant_collection
        self.dim = dim or settings.embedding_dim
        resolved_url = url or settings.qdrant_url
        if resolved_url:
            self.client = QdrantClient(url=resolved_url)
            self._location = resolved_url
        else:
            resolved_path = path or settings.qdrant_path
            self.client = QdrantClient(path=resolved_path)
            self._location = resolved_path

    def create_collection(self, recreate: bool = False) -> None:
        """Ensure the collection exists with cosine distance.

        Args:
            recreate: If True, drop and recreate the collection (wipes data).
                If False (default), create it only when missing.
        """
        from qdrant_client.models import Distance, VectorParams

        params = VectorParams(size=self.dim, distance=Distance.COSINE)
        if recreate:
            self.client.recreate_collection(
                collection_name=self.collection, vectors_config=params
            )
            return
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection, vectors_config=params
            )

    def upsert(
        self,
        ids: Sequence[Union[str, int]],
        vectors: np.ndarray,
        payloads: Sequence[Dict[str, Any]],
        batch_size: int = 256,
    ) -> None:
        """Upsert points into the collection.

        Args:
            ids: Point ids, one per vector. Use stable ids for idempotent upserts.
            vectors: ``(n, dim)`` array of embeddings.
            payloads: Per-point metadata dicts (e.g. ``{"text": ..., "doc_id": ...}``).
            batch_size: Number of points sent per request.

        Raises:
            ValueError: If the lengths of ids/vectors/payloads disagree.
        """
        from qdrant_client.models import PointStruct

        n = len(ids)
        if not (n == len(vectors) == len(payloads)):
            raise ValueError("ids, vectors, and payloads must have equal length")
        if n == 0:
            return

        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            points = [
                PointStruct(
                    id=ids[i],
                    vector=vectors[i].tolist(),
                    payload=payloads[i],
                )
                for i in range(start, stop)
            ]
            self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        query_filter: Optional[Any] = None,
    ) -> List[SearchHit]:
        """Search for the nearest points to ``query_vector``.

        Args:
            query_vector: 1-D ``(dim,)`` query embedding.
            top_k: Number of hits to return.
            query_filter: Optional ``qdrant_client.models.Filter`` to constrain
                results by payload.

        Returns:
            A list of :class:`SearchHit`, highest score first.
        """
        results = self.client.search(
            collection_name=self.collection,
            query_vector=np.asarray(query_vector, dtype=np.float32).tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            SearchHit(id=r.id, score=float(r.score), payload=dict(r.payload or {}))
            for r in results
        ]

    def count(self) -> int:
        """Return the number of points currently in the collection."""
        return int(self.client.count(collection_name=self.collection).count)
