"""
search/vector_store.py
-----------------------
Qdrant vector store wrapper.

Responsibilities:
  - Create/verify the Qdrant collection on startup
  - Upsert chunks by chunk_id (incremental — no full re-index needed)
  - Search by dense vector with optional metadata filters
  - Health check on startup

Runs in LOCAL mode by default (no Qdrant server needed).
Switch to server mode by setting QDRANT_HOST in .env.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION
from models import Chunk

log = structlog.get_logger(__name__)

# Local Qdrant storage path (used when running without a Qdrant server)
QDRANT_LOCAL_PATH = "./qdrant_storage"


# ── Client factory ────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    """
    Return a Qdrant client.
    - If QDRANT_HOST == 'localhost' and no server is running → use local file storage
    - If a remote host is configured → connect to Qdrant server
    """
    if QDRANT_HOST in ("localhost", "127.0.0.1"):
        os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
        log.info("vector_store.local_mode", path=QDRANT_LOCAL_PATH)
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    else:
        log.info("vector_store.server_mode", host=QDRANT_HOST, port=QDRANT_PORT)
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ── Collection setup ──────────────────────────────────────────────────────────

def ensure_collection(client: QdrantClient, vector_dim: int) -> None:
    """
    Create the Qdrant collection if it doesn't exist.
    Safe to call multiple times (idempotent).
    """
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
            ),
        )
        log.info(
            "vector_store.collection_created",
            collection=QDRANT_COLLECTION,
            dim=vector_dim,
        )
    else:
        log.info("vector_store.collection_exists", collection=QDRANT_COLLECTION)


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert_chunks(
    client: QdrantClient,
    chunks: List[Chunk],
    vectors: List[List[float]],
    batch_size: int = 100,
) -> None:
    """
    Upsert chunks into Qdrant with their embeddings.

    Uses chunk_id as the stable point ID (hashed to int).
    Existing points with the same ID are updated; new ones are inserted.
    Processes in batches to avoid memory/timeout issues.
    """
    total = len(chunks)
    log.info("vector_store.upsert_start", total=total)

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]

        points = [
            PointStruct(
                id=_chunk_id_to_int(chunk.chunk_id),
                vector=vector,
                payload=_chunk_to_payload(chunk),
            )
            for chunk, vector in zip(batch_chunks, batch_vectors)
        ]

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        log.debug(
            "vector_store.batch_upserted",
            batch=i // batch_size + 1,
            count=len(points),
        )

    log.info("vector_store.upsert_done", total=total)


# ── Search ────────────────────────────────────────────────────────────────────

def search_vectors(
    client: QdrantClient,
    query_vector: List[float],
    top_k: int = 20,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search Qdrant by dense vector similarity.

    Args:
        query_vector : Embedded query vector
        top_k        : Number of results to return
        filters      : Optional dict of payload field → value to filter by
                       e.g. {"bid_id": "Bid1", "doc_type": "addendum"}

    Returns:
        List of dicts with keys: chunk_id, score, text, and all metadata fields
    """
    qdrant_filter = _build_filter(filters) if filters else None

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        hits = response.points
    else:
        hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

    return [
        {
            "chunk_id": (r.payload or {}).get("chunk_id"),
            "score": r.score,
            **(r.payload or {}),
        }
        for r in hits
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_to_payload(chunk: Chunk) -> Dict[str, Any]:
    """Convert a Chunk to a Qdrant payload dict (all metadata + text)."""
    return {
        "chunk_id":        chunk.chunk_id,
        "bid_id":          chunk.bid_id,
        "file_name":       chunk.file_name,
        "page_number":     chunk.page_number,
        "chunk_index":     chunk.chunk_index,
        "text":            chunk.text,
        "doc_type":        chunk.doc_type,
        "addendum_number": chunk.addendum_number,
        "document_date":   chunk.document_date,
        "source_format":   chunk.source_format,
        "is_table":        chunk.is_table,
    }


def _build_filter(filters: Dict[str, Any]) -> Filter:
    """
    Build a Qdrant Filter from a plain dict.

    Supports:
      - Exact match:   {"bid_id": "Bid1"}
      - Range (gte):   {"addendum_number": {"gte": 2}}
    """
    conditions = []
    for field, value in filters.items():
        if value is None:
            continue
        if isinstance(value, dict):
            # Range filter e.g. {"gte": 2}
            conditions.append(
                FieldCondition(
                    key=field,
                    range=Range(**value),
                )
            )
        else:
            # Exact match
            conditions.append(
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value),
                )
            )
    return Filter(must=conditions)


def _chunk_id_to_int(chunk_id: str) -> int:
    """
    Convert a string chunk_id to a stable integer for Qdrant point ID.
    Uses Python's built-in hash (capped to positive 64-bit int).
    """
    return abs(hash(chunk_id)) % (2**63)


def health_check(client: Optional[QdrantClient] = None) -> bool:
    """
    Check if Qdrant is accessible and functioning.
    Returns True if healthy, False otherwise.
    """
    try:
        c = client or get_qdrant_client()
        _ = c.get_collections()
        return True
    except Exception as exc:
        log.warning("vector_store.health_check_failed", error=str(exc))
        return False

