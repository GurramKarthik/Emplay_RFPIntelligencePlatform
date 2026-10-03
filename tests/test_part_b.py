"""
tests/test_part_b.py
--------------------
Unit tests for Part B — RAG Search Engine:
  - Query Understanding (Pre-Retrieval)
  - BM25 Sparse Index
  - Hybrid Retriever & RRF Fusion
  - Cross-Encoder Reranker & Citations
  - FastAPI Search API & Agent Tool Interface

Run with:
  pytest tests/test_part_b.py -v
"""

from __future__ import annotations

import tempfile
import pytest
from unittest.mock import MagicMock, patch

from search.query_understander import (
    understand_query,
    QueryUnderstanding,
    _fallback,
    _parse_llm_output,
)
from search import bm25_index
from search.bm25_index import _tokenize
from search.hybrid_retriever import _rrf_merge, _add_rrf_single, _build_qdrant_filters
from search.reranker import CrossEncoderReranker, rerank
from search.search_api import (
    search_rfp,
    RFPSearchTool,
    create_app,
    SearchRequest,
)


# ── 1. Query Understander Tests ──────────────────────────────────────────────

class TestQueryUnderstander:
    def test_fallback(self):
        res = _fallback("What is the submission deadline?")
        assert isinstance(res, QueryUnderstanding)
        assert res.rewritten_query == "What is the submission deadline?"
        assert res.filters == {}
        assert res.mode == "hybrid"

    def test_parse_llm_output_valid(self):
        raw = """{
            "rewritten_query": "proposal submission deadline due date",
            "filters": {"bid_id": "Bid1", "doc_type": "rfp"},
            "mode": "keyword",
            "reasoning": "Looking for specific submission deadline."
        }"""
        res = _parse_llm_output(raw, "deadline for Bid1", known_bid_ids=["Bid1", "Bid2"])
        assert res.rewritten_query == "proposal submission deadline due date"
        assert res.filters["bid_id"] == "Bid1"
        assert res.filters["doc_type"] == "rfp"
        assert res.mode == "keyword"

    def test_parse_llm_output_markdown_fenced(self):
        raw = """```json
        {
            "rewritten_query": "addendum changes",
            "filters": {"addendum_number": 2, "doc_type": "addendum"},
            "mode": "hybrid",
            "reasoning": "Addendum 2 changes"
        }
        ```"""
        res = _parse_llm_output(raw, "what changed in addendum 2", known_bid_ids=None)
        assert res.filters["addendum_number"] == 2
        assert res.filters["doc_type"] == "addendum"



from search import bm25_index
from search.bm25_index import _tokenize


# ── 2. BM25 Index Tests ───────────────────────────────────────────────────────

class TestBM25Index:
    def test_tokenize(self):
        tokens = _tokenize("Addendum #2: Due Date is 2024-04-15!")
        assert "addendum" in tokens
        assert "2" in tokens
        assert "date" in tokens

    def test_bm25_search_exact_match(self):
        chunks = [
            {
                "chunk_id": "c1",
                "text": "The contract payment terms are Net 30 days after invoice.",
                "doc_type": "rfp",
                "page_number": 5,
                "file_name": "rfp.pdf",
                "bid_id": "TestBid",
            },
            {
                "chunk_id": "c2",
                "text": "Bid bond of 5% is required with proposal submission.",
                "doc_type": "rfp",
                "page_number": 8,
                "file_name": "rfp.pdf",
                "bid_id": "TestBid",
            },
            {
                "chunk_id": "c3",
                "text": "Addendum 1 changes the proposal due date to April 25.",
                "doc_type": "addendum",
                "addendum_number": 1,
                "page_number": 1,
                "file_name": "add1.pdf",
                "bid_id": "TestBid",
            },
        ]
        bm25_index.build_and_save("TestBid", chunks)

        # Exact keyword search
        results = bm25_index.search("bid bond 5%", bid_id="TestBid", top_k=2)
        assert len(results) > 0
        assert results[0]["chunk_id"] == "c2"

        # Search with doc_type filter
        filtered = bm25_index.search("date", bid_id="TestBid", top_k=5, filters={"doc_type": "addendum"})
        assert len(filtered) == 1
        assert filtered[0]["chunk_id"] == "c3"



# ── 3. Hybrid Retriever & RRF Fusion Tests ────────────────────────────────────

class TestHybridRetrieverRRF:
    def test_rrf_merge_formula(self):
        dense = [
            {"chunk_id": "c1", "text": "chunk 1 text", "score": 0.9},
            {"chunk_id": "c2", "text": "chunk 2 text", "score": 0.8},
        ]
        sparse = [
            {"chunk_id": "c2", "text": "chunk 2 text", "bm25_score": 12.5},
            {"chunk_id": "c3", "text": "chunk 3 text", "bm25_score": 9.1},
        ]

        fused = _rrf_merge(dense, sparse, k=60)
        by_id = {c["chunk_id"]: c for c in fused}

        # c2 appears at rank 2 in dense and rank 1 in sparse:
        # score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        expected_c2 = (1.0 / 62) + (1.0 / 61)
        assert abs(by_id["c2"]["rrf_score"] - expected_c2) < 1e-6
        assert "dense" in by_id["c2"]["retriever_hits"]
        assert "bm25" in by_id["c2"]["retriever_hits"]

        # c2 should have higher RRF score than c1 (rank 1 dense only)
        # c1 score = 1/61 ~ 0.01639; c2 score = 1/62 + 1/61 ~ 0.03252
        assert by_id["c2"]["rrf_score"] > by_id["c1"]["rrf_score"]

    def test_add_rrf_single(self):
        single = [{"chunk_id": "c1", "text": "sample"}]
        res = _add_rrf_single(single, source="dense", k=60)
        assert res[0]["rrf_score"] == 1.0 / 61
        assert res[0]["retriever_hits"] == "dense"

    def test_build_qdrant_filters(self):
        f = _build_qdrant_filters("Bid1", {"doc_type": "rfp"})
        assert f == {"bid_id": "Bid1", "doc_type": "rfp"}

        f_global = _build_qdrant_filters("__global__", None)
        assert f_global is None


# ── 4. Cross-Encoder Reranker & Citations Tests ───────────────────────────────

class TestReranker:
    def test_empty_candidates(self):
        assert rerank("query", []) == []

    def test_citations_standardization(self):
        candidates = [
            {
                "chunk_id": "Bid1__doc.pdf__3__1",
                "text": "Submission deadline is April 15, 2024.",
                "file_name": "doc.pdf",
                "page_number": 3,
                "bid_id": "Bid1",
                "doc_type": "rfp",
                "rrf_score": 0.03,
            }
        ]
        formatted = CrossEncoderReranker._format_results(candidates)
        assert len(formatted) == 1
        item = formatted[0]
        assert "sources" in item
        assert len(item["sources"]) == 1
        assert item["sources"][0]["file"] == "doc.pdf"
        assert item["sources"][0]["page"] == 3
        assert item["sources"][0]["bid_id"] == "Bid1"

    def test_reranker_with_mock_model(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "test-model"
        reranker._available = True
        mock_model = MagicMock()
        # Mock logits: candidate 2 is more relevant than candidate 1
        mock_model.predict.return_value = [-2.5, 4.2]
        reranker._model = mock_model

        candidates = [
            {"chunk_id": "c1", "text": "Irrelevant background text", "file_name": "f1.pdf", "page_number": 1},
            {"chunk_id": "c2", "text": "Exact matching submission date", "file_name": "f2.pdf", "page_number": 2},
        ]

        scored = reranker.rerank("What is the submission date?", candidates, top_k=2)
        assert len(scored) == 2
        # c2 should rank first because of higher model score
        assert scored[0]["chunk_id"] == "c2"
        assert scored[0]["rerank_score"] > scored[1]["rerank_score"]
        assert 0.0 <= scored[0]["rerank_score"] <= 1.0


# ── 5. Search API & Agent Tool Interface Tests ────────────────────────────────

class TestSearchAPIAndTool:
    @patch("search.search_api.hybrid_search")
    @patch("search.search_api.understand_query")
    def test_search_rfp_pipeline(self, mock_qu, mock_hybrid):
        mock_qu.return_value = QueryUnderstanding(
            rewritten_query="proposal submission deadline",
            filters={"bid_id": "Bid1"},
            mode="hybrid",
        )
        mock_hybrid.return_value = [
            {
                "chunk_id": "c1",
                "text": "The proposal deadline is May 1, 2024 at 2:00 PM.",
                "file_name": "rfp.pdf",
                "page_number": 4,
                "bid_id": "Bid1",
                "doc_type": "rfp",
                "rrf_score": 0.032,
            }
        ]

        res = search_rfp(
            query="When is the bid due?",
            bid_id="Bid1",
            top_k=1,
            use_reranker=False,
        )

        assert res["total_results"] == 1
        assert res["results"][0]["chunk_id"] == "c1"
        assert res["results"][0]["sources"][0]["file"] == "rfp.pdf"
        assert res["mode_used"] == "hybrid"
        assert res["latency_ms"] >= 0

    @patch("search.search_api.search_rfp")
    def test_rfp_search_tool_callable(self, mock_search_rfp):
        mock_search_rfp.return_value = {
            "results": [
                {
                    "chunk_id": "c1",
                    "text": "Liquidated damages are $500 per day.",
                    "sources": [{"file": "specs.pdf", "page": 10, "bid_id": "Bid1"}],
                }
            ]
        }

        tool = RFPSearchTool()
        results = tool(query="liquidated damages", bid_id="Bid1")
        assert len(results) == 1
        assert "Liquidated damages" in results[0]["text"]
        mock_search_rfp.assert_called_once()

    def test_fastapi_endpoints(self):
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)

        # Health endpoint
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "reranker" in data
        assert "bm25" in data

        # Bids endpoint
        resp_bids = client.get("/api/v1/bids")
        assert resp_bids.status_code == 200
        assert isinstance(resp_bids.json(), list)
