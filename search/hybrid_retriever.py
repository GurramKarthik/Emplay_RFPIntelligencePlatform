from __future__ import annotations
from typing import Any, Dict, List, Optional
import structlog
from search.embedder import get_embedder, BaseEmbedder
from search.vector_store import get_qdrant_client, search_vectors
from search import bm25_index

log = structlog.get_logger(__name__)



_RRF_K: int = 60
_DEFAULT_RETRIEVER_TOP_K: int = 20


def hybrid_search(
    query: str,
    bid_id: str = "__global__",
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    mode: str = "hybrid",
    retriever_top_k: int = _DEFAULT_RETRIEVER_TOP_K,
    embedder: Optional[BaseEmbedder] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval with Reciprocal Rank Fusion.

    RRF score = sum( 1 / (60 + rank_i) ) across retrievers.
    Chunks appearing high in both dense and sparse lists rank highest.

    Args:
        query          : Raw search query
        bid_id         : Bid to search; '__global__' for cross-bid
        top_k          : Final results to return after fusion
        filters        : Metadata filters e.g. {'doc_type': 'addendum'}
        mode           : 'hybrid' (RRF) | 'semantic' (Qdrant) | 'keyword' (BM25)
        retriever_top_k: Candidates pulled from each retriever before fusion
        embedder       : Pre-loaded embedder instance (optional)

    Returns:
        List of dicts sorted by rrf_score desc. Each dict includes:
          rrf_score, retriever_hits, chunk_id, text, file_name,
          page_number, bid_id, doc_type, addendum_number
    """
    log.info("hybrid_retriever.search", query=query[:80], bid_id=bid_id, mode=mode, top_k=top_k)

    dense_results: List[Dict[str, Any]] = []
    bm25_results: List[Dict[str, Any]] = []

    # Dense vector search (Qdrant)
    if mode in ("hybrid", "semantic"):
        emb = embedder or get_embedder()
        client = get_qdrant_client()
        qfilts = _build_qdrant_filters(bid_id, filters)
        vec = emb.embed_text(query)
        dense_results = search_vectors(client, query_vector=vec, top_k=retriever_top_k, filters=qfilts)
        log.debug("hybrid_retriever.dense_hits", count=len(dense_results))

    # BM25 keyword search
    if mode in ("hybrid", "keyword"):
        bm25_results = bm25_index.search(query=query, bid_id=bid_id, top_k=retriever_top_k, filters=filters)
        for r in bm25_results:
            r["score"] = r.pop("bm25_score", 0.0)
        log.debug("hybrid_retriever.bm25_hits", count=len(bm25_results))

    # RRF fusion
    if mode == "hybrid":
        fused = _rrf_merge(dense_results, bm25_results, k=_RRF_K)
    elif mode == "semantic":
        fused = _add_rrf_single(dense_results, source="dense", k=_RRF_K)
    else:
        fused = _add_rrf_single(bm25_results, source="bm25", k=_RRF_K)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    final = fused[:top_k]
    log.info("hybrid_retriever.done", returned=len(final),
             top_rrf=round(final[0]["rrf_score"], 5) if final else 0)
    return final


def _rrf_merge(
    dense: List[Dict[str, Any]],
    sparse: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Merge two ranked lists. score = sum(1 / (k + rank))."""
    rrf: Dict[str, Dict[str, Any]] = {}

    def _accumulate(results: List[Dict[str, Any]], source: str) -> None:
        for rank, chunk in enumerate(results, start=1):
            cid = chunk.get("chunk_id") or chunk.get("text", "")[:40]
            if cid not in rrf:
                rrf[cid] = {**chunk, "rrf_score": 0.0, "retriever_hits": set()}
            rrf[cid]["rrf_score"] += 1.0 / (k + rank)
            rrf[cid]["retriever_hits"].add(source)

    _accumulate(dense, "dense")
    _accumulate(sparse, "bm25")

    out = []
    for entry in rrf.values():
        entry["retriever_hits"] = ",".join(sorted(entry["retriever_hits"]))
        out.append(entry)
    return out


def _add_rrf_single(
    results: List[Dict[str, Any]],
    source: str,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Assign RRF scores for single-retriever mode (no merging)."""
    return [
        {**chunk, "rrf_score": 1.0 / (k + rank), "retriever_hits": source}
        for rank, chunk in enumerate(results, start=1)
    ]


def _build_qdrant_filters(
    bid_id: str,
    extra: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build Qdrant filter dict, scoping by bid unless global."""
    merged: Dict[str, Any] = {}
    if bid_id != "__global__":
        merged["bid_id"] = bid_id
    if extra:
        merged.update(extra)
    return merged or None
