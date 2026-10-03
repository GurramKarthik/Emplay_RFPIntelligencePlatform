"""
search/bm25_index.py
---------------------
BM25 keyword search index using rank_bm25.

Why BM25?
  - Critical for exact matches: bid numbers, part numbers, model numbers
  - Complements dense vector search (which is better for semantic similarity)
  - Lightweight — no extra infrastructure needed

Design:
  - One BM25 index per bid (stored as a pickle file)
  - Index is loaded into memory on first use and cached
  - Supports incremental addition of new bid documents
"""

from __future__ import annotations

import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import structlog
from rank_bm25 import BM25Okapi

log = structlog.get_logger(__name__)

# Directory where serialized BM25 index files are stored
BM25_INDEX_DIR = "./data/bm25_index"

# In-memory cache: bid_id → (BM25Okapi, corpus_metadata)
_INDEX_CACHE: Dict[str, Tuple[BM25Okapi, List[Dict[str, Any]]]] = {}


# ── Build & Save ──────────────────────────────────────────────────────────────

def build_and_save(
    bid_id: str,
    chunks: List[Dict[str, Any]],
) -> None:
    """
    Build a BM25 index for a bid from its chunks and save it to disk.

    Each chunk dict must have at least:
      - "chunk_id"  : str
      - "text"      : str
      - plus all metadata fields (bid_id, file_name, page_number, etc.)

    Args:
        bid_id : Bid identifier (used as the index file name)
        chunks : List of chunk dicts (from JSONL or in-memory)
    """
    os.makedirs(BM25_INDEX_DIR, exist_ok=True)

    # Tokenise each chunk's text for BM25
    tokenised_corpus = [_tokenize(chunk["text"]) for chunk in chunks]

    # Build the BM25 model
    bm25 = BM25Okapi(tokenised_corpus)

    # Store lightweight metadata alongside the index
    # (we return this in search results so callers get full context)
    corpus_metadata = [
        {
            "chunk_id":        c.get("chunk_id"),
            "bid_id":          c.get("bid_id"),
            "file_name":       c.get("file_name"),
            "page_number":     c.get("page_number"),
            "doc_type":        c.get("doc_type"),
            "addendum_number": c.get("addendum_number"),
            "document_date":   c.get("document_date"),
            "source_format":   c.get("source_format"),
            "is_table":        c.get("is_table", False),
            "text":            c.get("text"),
        }
        for c in chunks
    ]

    # Persist to disk
    index_path = _index_path(bid_id)
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "metadata": corpus_metadata}, f)

    # Update in-memory cache
    _INDEX_CACHE[bid_id] = (bm25, corpus_metadata)

    log.info(
        "bm25.index_saved",
        bid_id=bid_id,
        path=index_path,
        docs=len(chunks),
    )


def build_global_and_save(all_chunks: List[Dict[str, Any]]) -> None:
    """
    Build a single BM25 index covering ALL bids.
    Used for cross-bid Q&A queries that don't specify a bid_id.
    """
    build_and_save("__global__", all_chunks)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_index(bid_id: str) -> Tuple[Optional[BM25Okapi], List[Dict[str, Any]]]:
    """
    Load a BM25 index from disk (or return cached version).

    Returns (None, []) if no index exists for this bid.
    """
    if bid_id in _INDEX_CACHE:
        return _INDEX_CACHE[bid_id]

    path = _index_path(bid_id)
    if not os.path.exists(path):
        log.warning("bm25.index_not_found", bid_id=bid_id, path=path)
        return None, []

    with open(path, "rb") as f:
        data = pickle.load(f)

    bm25: BM25Okapi = data["bm25"]
    metadata: List[Dict[str, Any]] = data["metadata"]

    _INDEX_CACHE[bid_id] = (bm25, metadata)
    log.info("bm25.index_loaded", bid_id=bid_id, docs=len(metadata))
    return bm25, metadata


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    query: str,
    bid_id: str = "__global__",
    top_k: int = 20,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search the BM25 index for a query string.

    Args:
        query  : Raw search query
        bid_id : Which index to search ("__global__" for all bids)
        top_k  : Number of top results to return
        filters: Optional dict to post-filter results (e.g. {"doc_type": "addendum"})

    Returns:
        List of result dicts sorted by BM25 score (descending), with score added.
    """
    bm25, metadata = load_index(bid_id)
    if bm25 is None:
        log.warning("bm25.empty_index", bid_id=bid_id)
        return []

    tokenised_query = _tokenize(query)
    scores = bm25.get_scores(tokenised_query)

    # Pair each chunk with its score
    scored = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results: List[Dict[str, Any]] = []
    for idx, score in scored:
        if len(results) >= top_k:
            break
        if score <= 0:
            continue

        chunk_meta = metadata[idx]

        # Apply optional post-filters
        if filters and not _passes_filter(chunk_meta, filters):
            continue

        results.append({**chunk_meta, "bm25_score": float(score)})

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokeniser.
    Lowercases and splits on non-alphanumeric characters.
    Preserves important tokens like "JA-207652", "Net30".
    """
    return re.findall(r"[a-zA-Z0-9]+(?:[_\-][a-zA-Z0-9]+)*", text.lower())


def _index_path(bid_id: str) -> str:
    """Return the file path for a bid's BM25 index pickle."""
    return os.path.join(BM25_INDEX_DIR, f"{bid_id}_bm25.pkl")


def _passes_filter(chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Check if a chunk passes all provided filters.
    Supports exact match and gte range for numeric fields.
    """
    for field, condition in filters.items():
        value = chunk.get(field)
        if isinstance(condition, dict):
            if "gte" in condition and (value is None or value < condition["gte"]):
                return False
            if "lte" in condition and (value is None or value > condition["lte"]):
                return False
        else:
            if value != condition:
                return False
    return True


def list_indexed_bids() -> List[str]:
    """
    Return a list of all bid IDs that have a saved BM25 index on disk.
    Excludes '__global__'.
    """
    if not os.path.exists(BM25_INDEX_DIR):
        return []
    bids = []
    for fname in os.listdir(BM25_INDEX_DIR):
        if fname.endswith("_bm25.pkl"):
            bid_id = fname[:-9]  # strip '_bm25.pkl'
            if bid_id != "__global__":
                bids.append(bid_id)
    return sorted(bids)

