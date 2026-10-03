"""
search/reranker.py
------------------
Cross-Encoder Reranker for RFP Intelligence Platform.

Role in pipeline:
  Takes Top-N candidates from hybrid retrieval (Qdrant dense + BM25 sparse + RRF)
  and scores each (query, chunk_text) pair using a deep cross-encoder model
  (default: cross-encoder/ms-marco-MiniLM-L-6-v2).

Features:
  - Lazy-loaded singleton model (loads once into memory, reused across requests)
  - Scores (query, text) pairs directly with full bidirectional cross-attention
  - Appends rerank_score and standardized citations (sources)
  - Graceful fallback: If sentence-transformers is missing or model download fails,
    cleanly falls back to RRF ranking without breaking the retrieval pipeline
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import structlog

from config import RERANKER_MODEL, RERANKER_ENABLED, RERANKER_TOP_K

log = structlog.get_logger(__name__)

_RERANKER_INSTANCE: Optional["CrossEncoderReranker"] = None


class CrossEncoderReranker:
    """
    Wrapper around HuggingFace CrossEncoder model.
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._available = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            log.info("reranker.loading_model", model_name=self.model_name)
            self._model = CrossEncoder(self.model_name)
            self._available = True
            log.info("reranker.model_loaded", model_name=self.model_name)
        except Exception as exc:
            log.warning(
                "reranker.model_load_failed",
                error=str(exc),
                fallback="Will fall back to RRF scores",
            )
            self._model = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = RERANKER_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of retrieved chunk dictionaries against the query.

        Args:
            query      : Search query
            candidates : List of chunk dicts from retriever
            top_k      : Max results to return after reranking

        Returns:
            Reranked list of dicts with 'rerank_score' and standardized 'sources'
        """
        if not candidates:
            return []

        # If reranker disabled or unavailable, format citations and return top_k by RRF
        if not RERANKER_ENABLED or not self._available or self._model is None:
            log.info(
                "reranker.skipped",
                reason="disabled" if not RERANKER_ENABLED else "model_unavailable",
                candidate_count=len(candidates),
            )
            return self._format_results(candidates, default_score_key="rrf_score")[:top_k]

        pairs = [[query, c.get("text", "")] for c in candidates]

        try:
            # Predict cross-encoder scores
            raw_scores = self._model.predict(pairs)

            # Assign sigmoid-normalized or raw scores
            scored_candidates: List[Dict[str, Any]] = []
            for candidate, raw_score in zip(candidates, raw_scores):
                # Sigmoid normalization: 1 / (1 + exp(-x))
                score_val = float(raw_score)
                norm_score = 1.0 / (1.0 + math.exp(-score_val)) if -50 <= score_val <= 50 else (1.0 if score_val > 50 else 0.0)

                updated = dict(candidate)
                updated["rerank_score"] = round(norm_score, 4)
                updated["raw_rerank_score"] = round(score_val, 4)
                updated["score"] = updated["rerank_score"]

                # Ensure citations passthrough
                if "sources" not in updated or not updated["sources"]:
                    updated["sources"] = [
                        {
                            "file": updated.get("file_name", "unknown"),
                            "page": updated.get("page_number", 1),
                            "bid_id": updated.get("bid_id", "unknown"),
                            "doc_type": updated.get("doc_type"),
                            "addendum_number": updated.get("addendum_number"),
                        }
                    ]
                scored_candidates.append(updated)

            # Sort by rerank score descending
            scored_candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            log.info(
                "reranker.done",
                candidate_count=len(candidates),
                returned_count=min(len(scored_candidates), top_k),
                top_score=scored_candidates[0]["rerank_score"] if scored_candidates else 0.0,
            )
            return scored_candidates[:top_k]

        except Exception as exc:
            log.error("reranker.prediction_failed", error=str(exc), fallback="using RRF rank")
            return self._format_results(candidates, default_score_key="rrf_score")[:top_k]

    @staticmethod
    def _format_results(
        candidates: List[Dict[str, Any]],
        default_score_key: str = "rrf_score",
    ) -> List[Dict[str, Any]]:
        """Ensure all candidates have standardized citations and a 'score' field."""
        results: List[Dict[str, Any]] = []
        for c in candidates:
            item = dict(c)
            if "score" not in item:
                item["score"] = item.get(default_score_key, 0.0)
            if "sources" not in item or not item["sources"]:
                item["sources"] = [
                    {
                        "file": item.get("file_name", "unknown"),
                        "page": item.get("page_number", 1),
                        "bid_id": item.get("bid_id", "unknown"),
                        "doc_type": item.get("doc_type"),
                        "addendum_number": item.get("addendum_number"),
                    }
                ]
            results.append(item)
        return results


def get_reranker(model_name: Optional[str] = None) -> CrossEncoderReranker:
    """Return singleton CrossEncoderReranker instance."""
    global _RERANKER_INSTANCE
    target_model = model_name or RERANKER_MODEL
    if _RERANKER_INSTANCE is None or _RERANKER_INSTANCE.model_name != target_model:
        _RERANKER_INSTANCE = CrossEncoderReranker(model_name=target_model)
    return _RERANKER_INSTANCE


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = RERANKER_TOP_K,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience helper to rerank candidates."""
    reranker_obj = get_reranker(model_name=model_name)
    return reranker_obj.rerank(query=query, candidates=candidates, top_k=top_k)
