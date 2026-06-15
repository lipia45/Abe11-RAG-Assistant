"""Configuration for rag_assistant.

Values come from environment variables (loaded from a local ``.env`` via
``python-dotenv`` when present), with sane defaults. The only true secret is
``ANTHROPIC_API_KEY``; it is read from the environment and never hardcoded.

Usage::

    from rag_assistant.config import settings
    print(settings.embedding_model, settings.qdrant_path)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()  # loads .env from the current working directory if it exists
except ImportError:  # python-dotenv not installed yet; fall back to plain os.environ
    pass


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_opt(key: str) -> Optional[str]:
    val = os.environ.get(key)
    return val if val else None


@dataclass(frozen=True)
class Config:
    """Immutable, fully-resolved configuration object.

    Paths are resolved relative to ``DATA_DIR`` / the current working directory.
    Construct via :func:`load_config`; the module-level :data:`settings` is a
    ready-to-use instance.
    """

    # --- Secrets / generator ---
    anthropic_api_key: Optional[str]
    anthropic_model: str

    # --- Embeddings ---
    embedding_model: str
    embedding_dim: int

    # --- Vector store ---
    qdrant_path: str
    qdrant_url: Optional[str]
    qdrant_collection: str

    # --- Retrieval ---
    top_k: int

    # --- Chunking ---
    chunk_size: int
    chunk_overlap: int

    # --- Data paths ---
    data_dir: Path
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    corpus_path: Path = field(init=False)
    eval_path: Path = field(init=False)

    def __post_init__(self) -> None:
        # frozen dataclass: set derived paths via object.__setattr__
        raw = self.data_dir / "raw"
        processed = self.data_dir / "processed"
        object.__setattr__(self, "raw_dir", raw)
        object.__setattr__(self, "processed_dir", processed)
        object.__setattr__(self, "corpus_path", processed / "corpus.parquet")
        object.__setattr__(self, "eval_path", processed / "eval.parquet")

    @property
    def use_remote_qdrant(self) -> bool:
        """True when a Qdrant server URL is configured (takes precedence over path)."""
        return self.qdrant_url is not None


def load_config() -> Config:
    """Build a :class:`Config` from the current environment."""
    data_dir = Path(_env_str("DATA_DIR", "./data")).resolve()
    return Config(
        anthropic_api_key=_env_opt("ANTHROPIC_API_KEY"),
        anthropic_model=_env_str("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        embedding_model=_env_str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_dim=_env_int("EMBEDDING_DIM", 384),
        qdrant_path=_env_str("QDRANT_PATH", "./qdrant_storage"),
        qdrant_url=_env_opt("QDRANT_URL"),
        qdrant_collection=_env_str("QDRANT_COLLECTION", "rag_chunks"),
        top_k=_env_int("TOP_K", 5),
        chunk_size=_env_int("CHUNK_SIZE", 800),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
        data_dir=data_dir,
    )


# Module-level ready-to-use instance.
settings = load_config()
