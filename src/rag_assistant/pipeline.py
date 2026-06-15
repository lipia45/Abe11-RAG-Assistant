"""End-to-end RAG pipeline orchestration.

``RAGPipeline.query`` runs retrieval now and is wired to call generation in Phase 4.
Retrieval is fully functional; generation is gated behind ``generate=True`` and will
raise until :func:`rag_assistant.generator.answer` is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import generator
from .config import settings
from .retriever import Hit, Retriever


@dataclass
class RAGResult:
    """Result of a RAG query.

    Attributes:
        query: The original query.
        hits: Retrieved chunks, highest score first.
        answer: Generated answer (None until generation is run).
    """

    query: str
    hits: List[Hit] = field(default_factory=list)
    answer: Optional[str] = None


class RAGPipeline:
    """Orchestrate retrieval and (Phase 4) generation.

    Args:
        retriever: A :class:`Retriever`; created with defaults if omitted.
        top_k: Default number of chunks to retrieve.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.top_k = top_k or settings.top_k

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        generate: bool = False,
    ) -> RAGResult:
        """Run a RAG query.

        Args:
            query: The user question.
            top_k: Number of chunks to retrieve; falls back to the pipeline default.
            generate: If True, also produce a generated answer (Phase 4). Until
                generation is implemented this raises ``NotImplementedError``.

        Returns:
            A :class:`RAGResult` with the retrieved hits and (optionally) an answer.
        """
        k = top_k or self.top_k
        hits = self.retriever.retrieve(query, top_k=k)
        result = RAGResult(query=query, hits=hits)
        if generate:
            # Phase 4: generator.answer() will fill this in.
            result.answer = generator.answer(query, hits)
        return result
