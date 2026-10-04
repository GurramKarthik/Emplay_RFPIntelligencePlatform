# Part B — RAG Search Engine


> **Scope**: Indexing, hybrid retrieval, RRF merging, reranking, query understanding, citations

---

## 1. Overview

```
Part A chunks.jsonl
        │
        ▼
  ┌──────────────┐
  │   Embedder   │ ── generates dense vectors (text-embedding-3-small)
  └──────────────┘
        │
        ├──▶ Qdrant (vector store + metadata payload)
        └──▶ BM25 Index (rank_bm25)

                        ┌─────────────────────────────────┐
  User Query ──▶ Query  │  Understander (LLM pre-step)    │
                        │  - rewrites query               │
                        │  - extracts filters             │
                        │  - detects retrieval mode       │
                        └────────────┬────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
             Qdrant ANN Search               BM25 Keyword Search
               (Top-20 dense)                 (Top-20 sparse)
                    │                                 │
                    └──────────┬──────────────────────┘
                               ▼
                    Reciprocal Rank Fusion (RRF)
                               │ Top-30 merged
                               ▼
                    Metadata Filter (bid_id, doc_type, addendum_number)
                               │
                               ▼
                    Cross-Encoder Reranker
                               │ Top-5 to Top-10
                               ▼
                    Results: chunks + metadata + citations + scores
```

---

## 2. Components

| Component | Role | Tech |
|---|---|---|
| **Embedder** | Generates dense vectors for chunks + queries | `text-embedding-3-small` (OpenAI) or `BAAI/bge-base-en-v1.5` (local) |
| **Vector Store** | Stores dense vectors + metadata payload | `Qdrant` (local mode) |
| **BM25 Index** | Keyword / exact match search | `rank_bm25` |
| **Query Understander** | Rewrites query, extracts filters, detects mode | LLM call (lightweight) |
| **Hybrid Merger** | Fuses dense + sparse ranked lists | Reciprocal Rank Fusion (RRF) |
| **Metadata Filter** | Narrows results by `bid_id`, `doc_type`, `addendum_number` | Qdrant payload filters |
| **Reranker** | Re-scores top-k for precision | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Search API** | Exposes search as REST endpoint + agent tool | `FastAPI` |

---

## 3. Indexing

**Input:** `./data/chunks/{bid_id}_chunks.jsonl` (from Part A)

**Steps:**
1. Read each chunk from JSONL
2. Generate embedding via Embedder
3. Upsert into Qdrant with `chunk_id` as point ID + full metadata as payload
4. Add chunk text to BM25 index (in-memory, serialized to disk)

**Incremental indexing:**
```python
chunk_id = f"{bid_id}__{file_name}__{page_num}__{chunk_index}"
# Qdrant upsert: existing chunk_id → update; new → insert
# No full re-index when adding Bid3, Bid4, etc.
```

**Qdrant payload per point:**
```json
{
  "chunk_id": "Bid1__RFP_FINAL.pdf__3__1",
  "bid_id": "Bid1",
  "file_name": "RFP_FINAL.pdf",
  "doc_type": "rfp",
  "addendum_number": null,
  "page_number": 3,
  "text": "..."
}
```

---

## 4. Query Understanding (Pre-Retrieval Step)

**LLM prompt (lightweight):**
```
Given the user query, return:
1. rewritten_query: cleaner version for retrieval
2. filters: {bid_id, doc_type, addendum_number} — null if not implied
3. mode: "keyword" | "semantic" | "hybrid"

Query: "{user_query}"
```

**Example:**
```
Input:  "What changed in Addendum 2 for Bid1?"
Output: {
  "rewritten_query": "changes modifications Addendum 2 Bid1",
  "filters": {"bid_id": "Bid1", "addendum_number": 2},
  "mode": "keyword"
}
```

---

## 5. Retrieval & Merging

### RRF Formula
```
RRF_score(chunk) = Σ 1 / (k + rank_i)
  where k = 60 (standard constant), rank_i = rank in each result list
```

**Why RRF over Weighted Score:**
- No score normalization needed (dense vs sparse scores are on different scales)
- Parameter-free — no calibration required
- Robust out-of-the-box for mixed retrieval signals

### Metadata Filter Examples
```python
# Only addendums for Bid1
filter = {"bid_id": "Bid1", "doc_type": "addendum"}

# Latest addendum (number >= 2)
filter = {"bid_id": "Bid1", "addendum_number": {"gte": 2}}

# Specs for Bid2
filter = {"bid_id": "Bid2", "doc_type": "specs"}
```

---

## 6. Citations

Every returned chunk carries its source metadata — no extra processing needed:

```json
{
  "chunk_id": "Bid1__Addendum2.pdf__1__0",
  "text": "The due date is extended to April 15, 2024...",
  "score": 0.91,
  "sources": [
    {"file": "Addendum2.pdf", "page": 1, "bid_id": "Bid1"}
  ]
}
```

Citations are a **passthrough** from chunk metadata → no post-processing logic required.

---

## 7. Failure Points & Mitigations

| Failure | Mitigation |
|---|---|
| Empty retrieval | Return `"Not found in documents"`; never hallucinate |
| Embedding API timeout | Retry with exponential backoff; switch to local BGE model |
| Qdrant not running | Health-check on startup; fail fast with clear error message |
| BM25 misses exact match | Supplement with character n-gram index |
| Query Understander returns bad JSON | Fallback to raw query + no filters |

---

## 8. Project File Structure (Part B)

```
search/
├── embedder.py             # generate embeddings (OpenAI or local BGE)
├── vector_store.py         # Qdrant upsert, search, health-check
├── bm25_index.py           # rank_bm25 index build + search + disk serialize
├── query_understander.py   # LLM pre-step: rewrite + filter extraction
├── hybrid_retriever.py     # calls dense + sparse, applies RRF merge
├── reranker.py             # cross-encoder reranking
├── metadata_filter.py      # Qdrant payload filter builder
└── search_api.py           # FastAPI endpoints + agent tool interface

data/
└── bm25_index/
    └── {bid_id}_bm25.pkl   # serialized BM25 index per bid
```

---

## 9. Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Vector store | Qdrant (local) | Native payload filters; upsert by chunk_id; no infra needed |
| Keyword search | `rank_bm25` | Lightweight; critical for exact bid/part/model numbers |
| Merge strategy | RRF | Parameter-free; robust; no score normalization needed |
| Reranker | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Precision boost with minimal latency |
| Query pre-step | LLM query understander | Extracts filters + improves retrieval before hitting index |
| Citations | Chunk metadata passthrough | Zero extra logic; always present |
| Incremental indexing | Qdrant upsert by `chunk_id` | New bids → upsert only; no full re-index |
