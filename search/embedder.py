"""
search/embedder.py
------------------
Generates dense vector embeddings for text chunks and queries.

Supports two providers (configurable via .env):
  - openai  : text-embedding-3-small (default, best cost/quality)
  - local   : BAAI/bge-base-en-v1.5  (fully local, no API key needed)

Usage:
    from search.embedder import get_embedder
    embedder = get_embedder()
    vector = embedder.embed_text("What is the due date?")
    vectors = embedder.embed_batch(["chunk 1 text", "chunk 2 text"])
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List

import structlog

from config import EMBEDDING_PROVIDER, EMBEDDING_MODEL, OPENAI_API_KEY

log = structlog.get_logger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Interface all embedders must implement."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single string. Returns a float vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings. Returns list of float vectors."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality (needed when creating Qdrant collection)."""
        ...


# ── OpenAI Embedder ───────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder using OpenAI's text-embedding-3-small (or any OpenAI embedding model).
    Includes retry logic with exponential backoff on API errors.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds (doubles each retry)

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._model = model
        self._dimension = 1536 if "small" in model else 3072
        log.info("embedder.openai_init", model=model, dimension=self._dimension)

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with retry + exponential backoff.
        OpenAI allows up to 2048 texts per request.
        """
        # Split into safe sub-batches of 256 to avoid timeout on large inputs
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), 256):
            sub_batch = texts[i : i + 256]
            all_vectors.extend(self._embed_with_retry(sub_batch))
        return all_vectors

    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        delay = self.RETRY_DELAY
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                log.warning(
                    "embedder.openai_retry",
                    attempt=attempt,
                    error=str(e),
                    delay=delay,
                )
                if attempt == self.MAX_RETRIES:
                    raise
                time.sleep(delay)
                delay *= 2  # exponential backoff
        return []  # unreachable

    @property
    def dimension(self) -> int:
        return self._dimension


# ── FastEmbed Local Embedder (ONNX) ──────────────────────────────────────────

class FastEmbedder(BaseEmbedder):
    """
    Lightweight, fast local embedder using Qdrant's FastEmbed (ONNX runtime).
    Default model: BAAI/bge-small-en-v1.5 (384 dimensions).
    No API key required, CPU friendly, zero GPU/PyTorch overhead.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed is required: pip install fastembed"
            )
        self._model = TextEmbedding(model_name=model_name)
        self._model_name = model_name
        self._dim = 768 if "base" in model_name.lower() or "large" in model_name.lower() else 384
        log.info("embedder.fastembed_init", model=model_name, dimension=self._dim)

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings_iter = self._model.embed(texts)
        return [arr.tolist() for arr in embeddings_iter]

    @property
    def dimension(self) -> int:
        return self._dim


# ── Local BGE Embedder (PyTorch / sentence-transformers) ──────────────────────

class LocalBGEEmbedder(BaseEmbedder):
    """
    Local embedder using sentence-transformers (PyTorch).
    """

    def __init__(self, model: str = "BAAI/bge-base-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Local embeddings require sentence-transformers: "
                "pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model)
        self._dim = self._model.get_sentence_embedding_dimension()
        log.info("embedder.local_bge_init", model=model, dimension=self._dim)

    def embed_text(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    @property
    def dimension(self) -> int:
        return self._dim


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedder() -> BaseEmbedder:
    """
    Return the configured embedder based on EMBEDDING_PROVIDER env var.
    Supports:
      - 'fastembed' or 'local': FastEmbed ONNX (BAAI/bge-small-en-v1.5)
      - 'openai': OpenAI API (text-embedding-3-small)
      - 'sentence-transformers': PyTorch BGE
    """
    provider = EMBEDDING_PROVIDER.lower()

    if provider in ("fastembed", "local"):
        model_name = EMBEDDING_MODEL if "bge" in EMBEDDING_MODEL else "BAAI/bge-small-en-v1.5"
        return FastEmbedder(model_name=model_name)

    if provider == "openai":
        return OpenAIEmbedder(model=EMBEDDING_MODEL)

    if provider == "sentence-transformers":
        return LocalBGEEmbedder(model=EMBEDDING_MODEL)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER='{provider}'. "
        "Must be 'fastembed', 'local', or 'openai'."
    )

