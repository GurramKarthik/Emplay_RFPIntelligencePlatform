"""
search/__init__.py
------------------
Exposes the public indexing, retrieval, reranking, and search API for Part B.
"""

from search.indexer import run_indexing_pipeline
from search.hybrid_retriever import hybrid_search
from search.query_understander import understand_query, QueryUnderstanding
from search.reranker import rerank, get_reranker, CrossEncoderReranker
from search.search_api import search_rfp, RFPSearchTool, get_app

__all__ = [
    "run_indexing_pipeline",
    "hybrid_search",
    "understand_query",
    "QueryUnderstanding",
    "rerank",
    "get_reranker",
    "CrossEncoderReranker",
    "search_rfp",
    "RFPSearchTool",
    "get_app",
]
