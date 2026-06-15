"""rag_assistant: a small, inspectable Retrieval-Augmented Generation system.

Modules:
    config       configuration loaded from environment / .env with defaults
    chunking     split documents into overlapping chunks with stable ids
    embeddings   sentence-transformers embedding wrapper (lazy, batched)
    vectorstore  Qdrant local-mode store (cosine distance)
    retriever    embeddings + vectorstore -> ranked hits
    generator    Claude cited-answer generation (Phase 4 stub)
    pipeline     end-to-end RAG orchestration
"""

__version__ = "0.1.0"
