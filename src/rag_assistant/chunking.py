"""Document chunking with a sliding character window and stable chunk ids.

The chunker splits a document's text into overlapping character windows. Each chunk
gets a deterministic id derived from the source ``doc_id`` and the chunk's character
offset, so re-running ingestion produces the same ids (idempotent upserts).

Character-based windowing is intentionally simple and dependency-free; a token-aware
chunker can replace :func:`chunk_document` later behind the same signature.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    """A single chunk of a document.

    Attributes:
        chunk_id: Stable, deterministic id ``"{doc_id}::{start:08d}"`` (hashed form
            available via :attr:`uuid` for vector-store point ids).
        doc_id: Id of the source document.
        text: The chunk text.
        start: Character offset of the chunk start within the source text.
        end: Character offset of the chunk end (exclusive) within the source text.
        index: Sequential index of the chunk within the document (0-based).
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    index: int

    @property
    def uuid(self) -> str:
        """Deterministic hex digest of ``chunk_id`` (usable as a stable point id)."""
        return hashlib.sha1(self.chunk_id.encode("utf-8")).hexdigest()


def chunk_document(
    text: str,
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[Chunk]:
    """Split ``text`` into overlapping character windows.

    Args:
        text: The document text to split.
        doc_id: Identifier of the source document; used to build stable chunk ids.
        chunk_size: Maximum number of characters per chunk. Must be > 0.
        overlap: Number of characters shared between consecutive chunks. Must be
            >= 0 and < ``chunk_size``.

    Returns:
        A list of :class:`Chunk` objects in document order. Empty/whitespace-only
        windows are skipped. A document shorter than ``chunk_size`` yields a single
        chunk (when it contains non-whitespace text).

    Raises:
        ValueError: If ``chunk_size <= 0`` or ``overlap`` is out of range.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    step = chunk_size - overlap
    chunks: List[Chunk] = []
    n = len(text)
    start = 0
    index = 0

    while start < n:
        end = min(start + chunk_size, n)
        window = text[start:end]
        if window.strip():
            chunk_id = f"{doc_id}::{start:08d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=window,
                    start=start,
                    end=end,
                    index=index,
                )
            )
            index += 1
        if end >= n:
            break
        start += step

    return chunks
