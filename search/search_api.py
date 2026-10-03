"""
search/search_api.py
--------------------
FastAPI Search API and Agent Tool Interface for Part B.

Responsibilities:
  - Exposes the complete RAG search pipeline via REST API:
      Query Understanding -> Hybrid Retrieval (Qdrant + BM25) -> RRF Fusion -> Cross-Encoder Reranker -> Top-k Results with Citations
  - Endpoints:
      POST /api/v1/search    : Full query search with configurable reranking & filters
      GET  /api/v1/search    : Convenience GET query search
      GET  /api/v1/health    : Health-check for vector store, BM25, and reranker
      GET  /api/v1/bids      : List indexed bids and document stats
  - Direct Python interface for Part C agents:
      search_rfp(...)        : Python callable returning structured results
      RFPSearchTool          : Agent-ready tool wrapper (LangChain / LangGraph compatible)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import structlog

from config import (
    RERANKER_ENABLED,
    RERANKER_TOP_K,
    RETRIEVER_TOP_K,
)
from search.query_understander import understand_query, QueryUnderstanding
from search.hybrid_retriever import hybrid_search
from search.reranker import rerank, get_reranker, CrossEncoderReranker
from search.vector_store import get_qdrant_client, health_check as qdrant_health_check
from search import bm25_index

log = structlog.get_logger(__name__)


# ── Pydantic Request / Response Models ────────────────────────────────────────

class Citation(BaseModel):
    """Source reference for evidence citation."""
    file: str
    page: int
    bid_id: Optional[str] = None
    doc_type: Optional[str] = None
    addendum_number: Optional[int] = None


class SearchResultItem(BaseModel):
    """A single retrieved chunk with citation and scoring."""
    chunk_id: str
    text: str
    score: float
    rerank_score: Optional[float] = None
    rrf_score: Optional[float] = None
    sources: List[Citation] = Field(default_factory=list)
    file_name: Optional[str] = None
    page_number: Optional[int] = None
    bid_id: Optional[str] = None
    doc_type: Optional[str] = None
    addendum_number: Optional[int] = None
    retriever_hits: Optional[str] = None


class SearchRequest(BaseModel):
    """Search request payload."""
    query: str = Field(..., description="User search query or question")
    bid_id: str = Field(default="__global__", description="Target bid ID or '__global__' for all bids")
    top_k: int = Field(default=RERANKER_TOP_K, ge=1, le=50, description="Number of results to return")
    use_reranker: bool = Field(default=True, description="Whether to apply cross-encoder reranking")
    use_query_understanding: bool = Field(default=True, description="Whether to run LLM query understanding")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters e.g. {'doc_type': 'addendum'}")
    mode: Optional[str] = Field(default=None, description="'hybrid' | 'semantic' | 'keyword' (auto-detected if None)")


class SearchResponse(BaseModel):
    """Search response payload."""
    query: str
    rewritten_query: Optional[str] = None
    mode_used: str
    applied_filters: Dict[str, Any]
    total_results: int
    results: List[SearchResultItem]
    latency_ms: float
    query_understanding: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """System health check response."""
    status: str
    qdrant: Dict[str, Any]
    bm25: Dict[str, Any]
    reranker: Dict[str, Any]


class BidInfo(BaseModel):
    """Metadata about an indexed bid."""
    bid_id: str
    bm25_indexed: bool
    bm25_doc_count: int


# ── Core Search Pipeline (Pure Python) ────────────────────────────────────────

def search_rfp(
    query: str,
    bid_id: str = "__global__",
    top_k: int = RERANKER_TOP_K,
    use_reranker: bool = True,
    use_query_understanding: bool = True,
    filters: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    candidate_k: int = RETRIEVER_TOP_K,
) -> Dict[str, Any]:
    """
    Execute the full end-to-end RAG search pipeline:
      1. Pre-retrieval Query Understanding (rewrites query, extracts filters, picks mode)
      2. Hybrid Retrieval (Qdrant ANN + BM25 keyword search) + RRF fusion
      3. Cross-Encoder Reranking (scores candidates, applies sigmoid, standardizes citations)

    Returns:
      Dictionary matching SearchResponse schema.
    """
    t0 = time.perf_counter()
    log.info("search_api.search_rfp.start", query=query[:80], bid_id=bid_id)

    qu_info: Optional[QueryUnderstanding] = None
    search_query = query
    retrieval_mode = mode or "hybrid"
    merged_filters: Dict[str, Any] = dict(filters or {})

    # Step 1: Pre-retrieval Query Understanding
    if use_query_understanding:
        try:
            qu_info = understand_query(query)
            if qu_info.rewritten_query:
                search_query = qu_info.rewritten_query
            if not mode and qu_info.mode:
                retrieval_mode = qu_info.mode
            # Merge extracted filters with explicit filters (explicit takes precedence)
            for k, v in qu_info.filters.items():
                if k not in merged_filters and v is not None:
                    # If query understanding extracted bid_id and caller had __global__, adopt it
                    if k == "bid_id" and bid_id == "__global__":
                        bid_id = v
                    else:
                        merged_filters[k] = v
        except Exception as exc:
            log.warning("search_api.query_understanding_failed", error=str(exc))

    # Step 2: Hybrid Retrieval (Dense Qdrant + Sparse BM25 + RRF)
    # Pull candidate_k chunks before reranking
    candidates = hybrid_search(
        query=search_query,
        bid_id=bid_id,
        top_k=candidate_k,
        filters=merged_filters if merged_filters else None,
        mode=retrieval_mode,
        retriever_top_k=candidate_k,
    )

    # Step 3: Reranker
    if use_reranker and RERANKER_ENABLED:
        final_results = rerank(
            query=query,  # Use original user query for cross-encoder reranking
            candidates=candidates,
            top_k=top_k,
        )
    else:
        # Standardize citations and default score to RRF
        final_results = CrossEncoderReranker._format_results(candidates, default_score_key="rrf_score")[:top_k]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info(
        "search_api.search_rfp.done",
        returned=len(final_results),
        latency_ms=latency_ms,
        mode=retrieval_mode,
    )

    return {
        "query": query,
        "rewritten_query": search_query if search_query != query else None,
        "mode_used": retrieval_mode,
        "applied_filters": merged_filters,
        "total_results": len(final_results),
        "results": final_results,
        "latency_ms": latency_ms,
        "query_understanding": {
            "rewritten_query": qu_info.rewritten_query,
            "filters": qu_info.filters,
            "mode": qu_info.mode,
            "reasoning": qu_info.reasoning,
        } if qu_info else None,
    }


# ── Agent Tool Interface (Part C integration) ─────────────────────────────────

class RFPSearchTool:
    """
    Agent tool wrapper for Part C agents (Retrieval Agent, Q&A Agent).
    Can be used directly as a Python callable or registered as a LangChain / LangGraph tool.
    """

    name: str = "rfp_search"
    description: str = (
        "Search indexed RFP bid documents using hybrid vector + keyword search and cross-encoder reranking. "
        "Returns relevant chunks with file and page citations."
    )

    def __call__(
        self,
        query: str,
        bid_id: str = "__global__",
        top_k: int = 5,
        doc_type: Optional[str] = None,
        addendum_number: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {}
        if doc_type:
            filters["doc_type"] = doc_type
        if addendum_number is not None:
            filters["addendum_number"] = addendum_number

        resp = search_rfp(
            query=query,
            bid_id=bid_id,
            top_k=top_k,
            filters=filters or None,
        )
        return resp.get("results", [])

    def run(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return self(*args, **kwargs)


# ── FastAPI Application ───────────────────────────────────────────────────────

def create_app():
    """Factory creating FastAPI application instance."""
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="RFP Intelligence Platform — RAG Search API",
        description="Hybrid Vector (Qdrant) + BM25 Search Engine with Query Understanding, Cross-Encoder Reranking, and Citations.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/v1/search", response_model=SearchResponse, tags=["Search"])
    def post_search(req: SearchRequest):
        """Execute hybrid search with query understanding and cross-encoder reranker."""
        try:
            result = search_rfp(
                query=req.query,
                bid_id=req.bid_id,
                top_k=req.top_k,
                use_reranker=req.use_reranker,
                use_query_understanding=req.use_query_understanding,
                filters=req.filters,
                mode=req.mode,
            )
            return SearchResponse(**result)
        except Exception as exc:
            log.error("search_api.endpoint_error", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/v1/search", response_model=SearchResponse, tags=["Search"])
    def get_search(
        q: str = Query(..., description="Search query"),
        bid_id: str = Query("__global__", description="Bid ID or '__global__'"),
        top_k: int = Query(5, ge=1, le=50),
        use_reranker: bool = Query(True),
        use_query_understanding: bool = Query(True),
        doc_type: Optional[str] = Query(None),
        addendum_number: Optional[int] = Query(None),
        mode: Optional[str] = Query(None),
    ):
        """Convenience GET endpoint for browser testing and simple queries."""
        filters: Dict[str, Any] = {}
        if doc_type:
            filters["doc_type"] = doc_type
        if addendum_number is not None:
            filters["addendum_number"] = addendum_number

        result = search_rfp(
            query=q,
            bid_id=bid_id,
            top_k=top_k,
            use_reranker=use_reranker,
            use_query_understanding=use_query_understanding,
            filters=filters or None,
            mode=mode,
        )
        return SearchResponse(**result)

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
    def get_health():
        """Check the operational health of Qdrant, BM25, and Reranker."""
        client = get_qdrant_client()
        qd_ok = qdrant_health_check(client)

        bids = bm25_index.list_indexed_bids()
        reranker_instance = get_reranker()

        is_healthy = qd_ok and (len(bids) > 0)
        return HealthResponse(
            status="healthy" if is_healthy else "degraded",
            qdrant={"status": "ok" if qd_ok else "failed"},
            bm25={"indexed_bids": bids, "count": len(bids)},
            reranker={
                "enabled": RERANKER_ENABLED,
                "model": reranker_instance.model_name,
                "available": reranker_instance.is_available,
            },
            query_understander={"status": "ready"},
        )

    @app.get("/api/v1/bids", response_model=List[BidInfo], tags=["Bids"])
    def get_bids():
        """List indexed bids and their document counts."""
        bids = bm25_index.list_indexed_bids()
        items: List[BidInfo] = []
        for bid in bids:
            bm25_model, corpus_metadata = bm25_index.load_index(bid)
            items.append(BidInfo(
                bid_id=bid,
                bm25_indexed=bm25_model is not None,
                bm25_doc_count=len(corpus_metadata) if corpus_metadata else 0,
            ))
        return items

    return app


# Module-level app instance for uvicorn
app = None
def get_app():
    global app
    if app is None:
        app = create_app()
    return app

# If imported by uvicorn search.search_api:app
try:
    app = create_app()
except Exception:
    pass
