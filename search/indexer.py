"""
search/indexer.py
-----------------
Part B indexing pipeline orchestrator.

Reads chunks from Part A's JSONL checkpoint and indexes them into:
  1. Qdrant  — dense vector store (for semantic search)
  2. BM25    — keyword index (for exact match search)

Designed for incremental indexing:
  - Qdrant uses upsert by chunk_id → existing chunks are updated, new ones inserted
  - BM25 index is rebuilt per bid (fast, since it's in-memory)

Usage (CLI):
  python main.py --bid ./Bid1 --bid-id Bid1 --part b

Usage (as a tool / API call):
  from search.indexer import run_indexing_pipeline
  result = run_indexing_pipeline(bid_id="Bid1")
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import structlog

from config import CHUNKS_OUTPUT_DIR
from models import Chunk
from search.embedder import get_embedder, BaseEmbedder
from search.vector_store import get_qdrant_client, ensure_collection, upsert_chunks
from search import bm25_index

log = structlog.get_logger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_indexing_pipeline(bid_id: str) -> Dict[str, Any]:
    """
    Index all chunks for a bid into Qdrant + BM25.

    Reads from: ./data/chunks/{bid_id}_chunks.jsonl  (written by Part A)
    Writes to:
      - Qdrant collection (local file storage by default)
      - ./data/bm25_index/{bid_id}_bm25.pkl
      - ./data/bm25_index/__global___bm25.pkl  (cross-bid index)

    Returns a summary dict with counts and status.
    """
    log.info("indexer.start", bid_id=bid_id)

    # ── Step 1: Load chunks from JSONL ───────────────────────────────────────
    chunks = _load_chunks_jsonl(bid_id)
    if not chunks:
        log.error("indexer.no_chunks", bid_id=bid_id)
        return {"bid_id": bid_id, "status": "error", "reason": "No chunks found"}

    log.info("indexer.chunks_loaded", bid_id=bid_id, count=len(chunks))

    # ── Step 2: Get embedder ─────────────────────────────────────────────────
    embedder = get_embedder()

    # ── Step 3: Embed all chunks (batch for efficiency) ──────────────────────
    log.info("indexer.embedding_start", bid_id=bid_id, chunks=len(chunks))
    texts = [c.text for c in chunks]
    vectors = embedder.embed_batch(texts)
    log.info("indexer.embedding_done", bid_id=bid_id)

    # ── Step 4: Upsert into Qdrant ───────────────────────────────────────────
    client = get_qdrant_client()
    ensure_collection(client, vector_dim=embedder.dimension)
    upsert_chunks(client, chunks, vectors)
    log.info("indexer.qdrant_done", bid_id=bid_id, upserted=len(chunks))

    # ── Step 5: Build BM25 index for this bid ────────────────────────────────
    chunk_dicts = [c.model_dump() for c in chunks]
    bm25_index.build_and_save(bid_id, chunk_dicts)

    # ── Step 6: Rebuild global BM25 index (all bids) ─────────────────────────
    _rebuild_global_bm25_index()

    summary = {
        "bid_id":          bid_id,
        "status":          "success",
        "chunks_indexed":  len(chunks),
        "qdrant_collection": os.environ.get("QDRANT_COLLECTION", "rfp_chunks"),
        "bm25_index_path": f"./data/bm25_index/{bid_id}_bm25.pkl",
    }
    log.info("indexer.complete", **summary)
    return summary


# ── Private Helpers ───────────────────────────────────────────────────────────

def _load_chunks_jsonl(bid_id: str) -> List[Chunk]:
    """
    Load chunks from the JSONL checkpoint written by Part A.
    Returns a list of Chunk Pydantic objects.
    """
    jsonl_path = os.path.join(CHUNKS_OUTPUT_DIR, f"{bid_id}_chunks.jsonl")

    if not os.path.exists(jsonl_path):
        log.error("indexer.jsonl_not_found", path=jsonl_path)
        return []

    chunks: List[Chunk] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.model_validate_json(line))
            except Exception as e:
                log.warning("indexer.chunk_parse_error", error=str(e), line=line[:80])

    return chunks


def _rebuild_global_bm25_index() -> None:
    """
    Rebuild the global BM25 index by combining all per-bid JSONL files.
    Called after each bid is indexed so cross-bid Q&A always has fresh data.
    """
    all_chunks: List[Dict[str, Any]] = []

    if not os.path.exists(CHUNKS_OUTPUT_DIR):
        return

    for file_name in os.listdir(CHUNKS_OUTPUT_DIR):
        if not file_name.endswith("_chunks.jsonl"):
            continue
        jsonl_path = os.path.join(CHUNKS_OUTPUT_DIR, file_name)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if all_chunks:
        bm25_index.build_global_and_save(all_chunks)
        log.info("indexer.global_bm25_rebuilt", total_chunks=len(all_chunks))
