"""Retriever: ties the embedder and the vector store together.

``Retriever.retrieve`` embeds a query, searches the Qdrant collection, and returns
ranked :class:`Hit` objects carrying the chunk id, score, text, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import settings
from .embeddings import Embedder
from .vectorstore import QdrantStore


@dataclass(frozen=True)
class Hit:
    """A single retrieved chunk.

    Attributes:
        id: Point/chunk id.
        score: Cosine similarity (higher is better).
        text: The chunk text (from payload ``text``, empty if absent).
        metadata: Remaining payload fields (everything except ``text``).
    """

    id: Any
    score: float
    text: str
    metadata: Dict[str, Any]


class Retriever:
    """Embed a query and fetch the top-k most similar chunks.

    Args:
        embedder: An :class:`Embedder`; created with defaults if omitted.
        store: A :class:`QdrantStore`; created with defaults if omitted.
        default_top_k: Fallback ``top_k`` when not passed to :meth:`retrieve`.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[QdrantStore] = None,
        default_top_k: Optional[int] = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.store = store or QdrantStore()
        self.default_top_k = default_top_k or settings.top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        query_filter: Optional[Any] = None,
    ) -> List[Hit]:
        """Retrieve the most relevant chunks for ``query``.

        Args:
            query: The user query.
            top_k: Number of hits to return; falls back to ``default_top_k``.
            query_filter: Optional Qdrant payload filter.

        Returns:
            A list of :class:`Hit`, highest score first.
        """
        k = top_k or self.default_top_k
        query_vector = self.embedder.embed_query(query)
        results = self.store.search(query_vector, top_k=k, query_filter=query_filter)
        hits: List[Hit] = []
        for r in results:
            payload = dict(r.payload)
            text = payload.pop("text", "")
            hits.append(
                Hit(id=r.id, score=r.score, text=text, metadata=payload)
            )
        return hits
