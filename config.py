"""
config.py
---------
Loads all configuration from environment variables (.env file).
All tuneable parameters live here — nothing is hardcoded in logic files.
"""

import os
from dotenv import load_dotenv

# Load .env file (if present); silently skip if not found
load_dotenv()


# ── Part A — Ingestion & Parsing ──────────────────────────────────────────────

# Max parse attempts per file before marking as FAILED
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 3))

# SQLite tracker database path
TRACKER_DB_PATH: str = os.getenv("TRACKER_DB_PATH", "./data/parse_jobs.db")

# Directory where per-bid chunk JSONL files are written
CHUNKS_OUTPUT_DIR: str = os.getenv("CHUNKS_OUTPUT_DIR", "./data/chunks")


# ── Chunking ──────────────────────────────────────────────────────────────────

# Max tokens per chunk (sliding window fallback)
CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", 512))

# Overlap between consecutive sliding-window chunks
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", 64))

# Context tokens added before/after a table chunk
TABLE_CONTEXT_TOKENS: int = int(os.getenv("TABLE_CONTEXT_TOKENS", 50))


# ── Part B — Vector Store ─────────────────────────────────────────────────────

QDRANT_HOST: str       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "rfp_chunks")


# ── LLM ───────────────────────────────────────────────────────────────────────

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL: str    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
OPENAI_API_KEY: str     = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY: str     = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY: str       = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY: str  = os.getenv("ANTHROPIC_API_KEY", "")


# ── Embeddings ────────────────────────────────────────────────────────────────

EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL: str    = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# ── Part B — Reranker & Search API ───────────────────────────────────────────

RERANKER_MODEL: str      = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ENABLED: bool   = os.getenv("RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")
RERANKER_TOP_K: int      = int(os.getenv("RERANKER_TOP_K", 5))
RETRIEVER_TOP_K: int     = int(os.getenv("RETRIEVER_TOP_K", 20))
SEARCH_API_HOST: str     = os.getenv("SEARCH_API_HOST", "0.0.0.0")
SEARCH_API_PORT: int     = int(os.getenv("SEARCH_API_PORT", 8000))


# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR: str   = os.getenv("LOG_DIR", "./logs")

