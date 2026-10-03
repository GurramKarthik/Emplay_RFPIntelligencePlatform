# RFP Intelligence Platform — Architecture Design

> **Status**: Draft v1 — Under Discussion  
> **Scope**: All four parts (A, B, C, D)  
> **Note**: No implementation yet. This document drives the design conversation.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RFP Intelligence Platform                        │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Part A  │───▶│   Part B     │◀───│       Part C             │  │
│  │ Ingestion│    │ RAG Search   │    │  Multi-Agent System      │  │
│  │ & Parsing│    │   Engine     │    │  (Orchestrator + Agents) │  │
│  └──────────┘    └──────────────┘    └──────────┬───────────────┘  │
│                                                 │                   │
│                                      ┌──────────▼───────────────┐  │
│                                      │       Part D             │  │
│                                      │  Structured JSON Output  │  │
│                                      └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part A — Document Ingestion & Parsing

### Components

| Component | Role | Tech |
|---|---|---|
| **HTML Parser** | Extracts text + tables from BidNet HTML pages | `trafilatura` + `BeautifulSoup4` |
| **PDF Parser** | Extracts text + tables from RFP/addendum PDFs | `pdfplumber` (primary), `PyMuPDF` (fallback) |
| **OCR Fallback** *(bonus)* | Handles scanned pages | `pytesseract` + `pdf2image` |
| **Table Extractor** | Preserves tabular structure as markdown | `pdfplumber` table API |
| **Text Cleaner** | Removes headers/footers, fixes hyphenation, normalizes whitespace | Custom regex pipeline |
| **Metadata Tagger** | Attaches structured metadata to each document chunk | Custom |
| **Failure Logger** | Logs unreadable/empty pages without crashing | `structlog` |

### Metadata Schema (per document)

```python
{
  "bid_id": "Bid1",
  "file_name": "Addendum 2.pdf",
  "doc_type": "addendum",          # bid_page | rfp | addendum | specs | affidavit
  "addendum_number": 2,            # None for non-addenda
  "page_number": 1,
  "document_date": "2024-03-15",   # extracted or None
  "source_format": "pdf"           # pdf | html
}
```

### Data Flow

```
Bid Folder (HTML + PDFs)
        │
        ▼
  ┌─────────────┐
  │ File Router │ ──── .html ──▶ HTML Parser ──▶ Text + Tables
  │             │ ──── .pdf  ──▶ PDF Parser  ──▶ Text + Tables
  └─────────────┘                    │
                                     ▼
                              Text Cleaner
                                     │
                                     ▼
                              Metadata Tagger
                                     │
                                     ▼
                        ParsedDocument objects (list)
                        (passed to Part B for indexing)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary PDF parser | `pdfplumber` | Best table extraction; handles complex layouts |
| HTML parser | `trafilatura` first, `BS4` fallback | `trafilatura` handles boilerplate removal well |
| Failure handling | Log + skip; never crash pipeline | Assignment explicitly requires graceful degradation |
| Table format | Convert to Markdown | Preserves structure for both embedding and LLM context |
| Date extraction | Regex + dateparser library | Flexible; handles "March 15, 2024", "03/15/2024", etc. |

### Failure Points & Mitigations

| Failure | Mitigation |
|---|---|
| Scanned PDF (no text layer) | Detect via empty text → fallback to OCR |
| Corrupt/password-protected PDF | Try/except → log and skip |
| Footer/header repetition in chunks | Strip via frequency analysis across pages |
| Multi-column PDFs | Use `PyMuPDF` word-sort mode as fallback |

---

## Part B — RAG Search Engine

### Components

| Component | Role | Tech |
|---|---|---|
| **Chunker** | Splits parsed docs into retrieval units | Custom (section-aware + sliding window) |
| **Embedding Model** | Encodes chunks into dense vectors | `text-embedding-3-small` (OpenAI) or `BAAI/bge-base-en-v1.5` (local) |
| **Vector Store** | Stores + queries dense embeddings | `Qdrant` (recommended) |
| **BM25 Index** | Keyword/exact match search | `rank_bm25` or `Elasticsearch` lightweight |
| **Reranker** | Re-scores top-k candidates from both retrievers | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Hybrid Merger** | Fuses dense + sparse results | Reciprocal Rank Fusion (RRF) |
| **Search API** | Exposes search as REST endpoint + tool interface | `FastAPI` |
| **Metadata Filter** | Filters by bid_id, doc_type, addendum_number | Qdrant payload filters |

### Chunking Strategy

```
Strategy: Section/Heading-aware chunking with sliding overlap fallback

1. PRIMARY: Detect section headings (regex + font-size heuristics from PyMuPDF)
   → Create one chunk per section (heading + body)
   → Tables: always kept as atomic units (never split mid-table)

2. FALLBACK: Sliding window
   → chunk_size = 512 tokens, overlap = 64 tokens

3. TABLE CHUNKS: Stored separately with doc_type=table
   → Markdown table + surrounding context (50 tokens before/after)

Justification:
  - RFPs are structured docs; section boundaries are natural retrieval units
  - Tables contain key bid fields → must not be split
  - 512 tokens balances context richness vs. embedding quality
```

### Retrieval Flow

```
Query (string)
     │
     ├──▶ Embedding Model ──▶ Dense Vector ──▶ Qdrant ANN Search ──▶ Top-20
     │
     └──▶ BM25 Index ──────────────────────────────────────────────▶ Top-20
                                        │
                                        ▼
                            Reciprocal Rank Fusion (RRF)
                                        │
                                        ▼ Top-30 merged
                            Metadata Filter (bid_id, doc_type)
                                        │
                                        ▼ Filtered
                            Cross-Encoder Reranker
                                        │
                                        ▼ Top-5 to Top-10
                            Return: chunks + metadata + scores
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Vector store | **Qdrant** | Native payload filtering; supports incremental upsert; no re-index needed |
| Hybrid merge | **RRF** | Parameter-free; robust; no score normalization needed |
| Reranker | **Cross-encoder** | Significantly improves precision at cost of small latency |
| Embedding model | **text-embedding-3-small** | Best cost/quality ratio; or BGE for fully local setup |
| Keyword search | **BM25** (rank_bm25) | Critical for exact bid numbers, part numbers, model numbers |
| Incremental indexing | **Qdrant upsert by chunk_id** | New bid folders → new chunks; existing untouched |

### Incremental Indexing Design

```python
chunk_id = f"{bid_id}__{file_name}__{page_num}__{chunk_index}"
# Qdrant upsert: existing chunk_id → update; new → insert
# No full re-index required when adding Bid3, Bid4, etc.
```

### Metadata Filtering Examples

```
# Get only addendums for Bid1
filter = {bid_id: "Bid1", doc_type: "addendum"}

# Get specs documents for Bid2
filter = {bid_id: "Bid2", doc_type: "specs"}

# Get latest addendum (highest number)
filter = {bid_id: "Bid1", addendum_number: {gte: 2}}
```

### Failure Points & Mitigations

| Failure | Mitigation |
|---|---|
| Empty retrieval | Return "Not found in documents"; never hallucinate |
| Embedding API timeout | Retry with exponential backoff; local fallback model |
| BM25 misses exact match | Supplement with character n-gram index |
| Qdrant down | Health-check on startup; fail fast with clear error |

---

## Part C — Multi-Agent System

### Agent Roster

| Agent | Responsibility | Output |
|---|---|---|
| **Orchestrator** | Plans task, routes to agents, manages retries, aggregates final output | Task plan + final JSON |
| **Retrieval Agent** | Calls search engine with field-specific queries; returns evidence chunks with citations | `Evidence[]` with citations |
| **Extraction Agent(s)** | Fills structured fields using only retrieved evidence (no hallucination) | `DraftField[]` values |
| **Addendum Reconciliation Agent** | Compares base RFP vs addendums; applies latest valid change to each field | Change log + updated fields |
| **Validator / Critic Agent** | Checks evidence support, format validity, consistency; flags failures | Validation report + confidence |
| **Q&A / Report Agent** | Answers free-form NL questions with cited evidence | Cited answer + JSON report |

### Shared State Object

```python
class BidExtractionState(BaseModel):
    bid_id: str
    task_mode: Literal["extraction", "qa"]
    query: Optional[str]                     # for QA mode
    
    # Planning
    field_plan: List[str]                    # fields to extract
    
    # Evidence
    retrieved_evidence: Dict[str, List[Evidence]]  # field → chunks
    
    # Extraction
    draft_fields: Dict[str, DraftField]     # field → {value, sources, confidence}
    
    # Reconciliation
    addendum_changes: List[AddendumChange]
    
    # Validation
    validation_results: Dict[str, ValidationResult]  # passed/failed/not_found
    retry_counts: Dict[str, int]            # field → retry count (max 3)
    
    # Final
    final_output: Optional[BidRecord]
    trace: List[AgentStep]                  # observability log
```

### Orchestration Flow

```
User Request (bid_folder, mode)
         │
         ▼
   [Orchestrator Agent]
         │
         ├─── Extraction Mode ──────────────────────────────────────┐
         │                                                          │
         │    ┌─────────────────────────────────────────────────┐  │
         │    │  PARALLEL: Field Group Extraction               │  │
         │    │                                                 │  │
         │    │  Group 1 (Dates/Logistics)  ──▶ Extraction A   │  │
         │    │  Group 2 (Commercial/Legal) ──▶ Extraction B   │  │
         │    │  Group 3 (Product/Specs)    ──▶ Extraction C   │  │
         │    │                                                 │  │
         │    │  Each calls: Retrieval Agent → Extraction Agent │  │
         │    └──────────────────┬──────────────────────────────┘  │
         │                       │                                  │
         │                       ▼                                  │
         │         [Addendum Reconciliation Agent]                  │
         │                       │                                  │
         │                       ▼                                  │
         │          [Validator / Critic Agent]                      │
         │                       │                                  │
         │             ┌─────────┴──────────┐                      │
         │             │ FAILED fields      │ PASSED fields        │
         │             ▼                    ▼                      │
         │       Re-run Retrieval     Finalize output              │
         │       + Extraction                                      │
         │       (max 3 retries)                                   │
         │                                          ◀─────────────┘
         │
         └─── QA Mode ──▶ Retrieval Agent ──▶ Q&A/Report Agent ──▶ Cited Answer
```

### Agent Framework Choice

**Recommended: LangGraph**

| Criterion | LangGraph | CrewAI | AutoGen |
|---|---|---|---|
| Explicit state graph | ✅ Native | ❌ Implicit | ❌ Implicit |
| Feedback loops / cycles | ✅ Native | ⚠️ Limited | ⚠️ Complex |
| Parallel node execution | ✅ Fan-out | ⚠️ Limited | ⚠️ Complex |
| Structured outputs | ✅ Easy | ✅ Yes | ✅ Yes |
| Observability | ✅ LangSmith | ⚠️ Basic | ⚠️ Basic |
| Complexity | Medium | Low | High |

**LangGraph wins** because the feedback loop (Validator → retry Orchestrator) is a graph cycle — LangGraph models this natively. CrewAI would require workarounds.

### Field Grouping for Parallelism

```
Group 1 — Dates & Logistics:
  Due Date, Pre Bid Meeting, Delivery Date, Term of Bid

Group 2 — Commercial & Legal:
  Bid Bond, Payment Terms, Bid Submission Type, Additional Documentation, 
  Contract/Cooperative, MFG Registration

Group 3 — Product & Specs:
  Product, Model_no, Part_no, Product Specification, Installation

Group 4 — Identity & Summary:
  Bid Number, Title, Contact Info, Company Name, Bid Summary
```

### Guardrails & Observability

- Every field value **must** have `sources[]` — if empty → value = null, notes = "Not found"
- Max retry count = 3 per field (prevents infinite loops)
- LangSmith tracing OR structured JSON log per run (input, tool call, output, tokens, latency)
- All agent messages use **Pydantic models** (no raw dicts)
- LLM errors → caught at agent boundary; Orchestrator marks field as "extraction_error" and continues

---

## Part D — Structured Information Extraction

### Output Schema

```python
class FieldValue(BaseModel):
    value: Optional[str]
    sources: List[Source]       # [{file, page}]
    confidence: float           # 0.0–1.0
    notes: Optional[str]        # "Extended by Addendum 2", "Not found", etc.

class BidRecord(BaseModel):
    bid_id: str
    fields: Dict[str, FieldValue]   # all 20 fields
    addendum_changes: List[AddendumChange]
    validation: ValidationSummary   # {passed, failed, not_found}
```

### Extraction Prompt Pattern

```
System: You are an extraction agent. Extract ONLY from the evidence below.
        If the value is not in the evidence, return null with a reason.
        Never guess or infer values not explicitly stated.

Evidence:
  [chunk 1 — file: X, page: N]
  ...
  [chunk k — file: Y, page: M]

Task: Extract the field "{field_name}" ({description}).
      Return JSON: {value, sources, confidence, notes}
```

### Addendum Reconciliation Logic

```
For each field:
  1. Extract value from base RFP → v_base
  2. For each addendum (sorted by number ASC):
       Check if addendum modifies this field → v_addendum
  3. Final value = last non-null v_addendum OR v_base
  4. Record change log: {field, original, updated, addendum_number, source}
```

### Confidence Score Heuristics

| Score Range | Meaning |
|---|---|
| 0.90–1.00 | Exact value found, single unambiguous source |
| 0.70–0.89 | Value found but needs interpretation (date format, implicit) |
| 0.50–0.69 | Value inferred from context; validator flags for review |
| 0.00–0.49 | Weak or no evidence; likely "Not found" |

---

## Cross-Part Integration Map

```
Part A ──[ParsedDocument]──▶ Part B ──[Chunks + Metadata]──▶ Qdrant/BM25 Index
                                  ◀──[Evidence + Citations]── Part C Retrieval Agent
Part C Extraction Agent ──[DraftFields]──▶ Part C Validator
Part C Validator ──[Validated Fields]──▶ Part D Output Schema
Part D ──[BidRecord JSON]──▶ File Output + API Response
```

---

## Technology Stack Summary

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.11 | Assignment requirement |
| PDF Parsing | pdfplumber + PyMuPDF | Complementary strengths |
| HTML Parsing | trafilatura + BS4 | Robustness |
| Vector Store | Qdrant | Filtering + incremental upsert |
| BM25 | rank_bm25 | Lightweight; no extra infra |
| Embeddings | text-embedding-3-small (or BGE) | Cost/quality balance |
| Reranker | Cross-encoder (HuggingFace) | Precision boost |
| LLM | Gemini 1.5 Pro / GPT-4o / Claude 3.5 | Configurable via env var |
| Agent Framework | LangGraph | Native cycles + state graph |
| API | FastAPI | Clean; async-native |
| Config | python-dotenv + config.py | No secrets in code |
| Observability | LangSmith / structlog | Assignment requirement |
| UI (bonus) | Streamlit | Fast; +1 bonus point |
| Testing | pytest | Parsing + search unit tests |
| Containerization | Docker + docker-compose | Bonus + reproducibility |

---

## Review Criteria Alignment

| Criterion (Weight) | How Architecture Satisfies It |
|---|---|
| RAG Search Engine (25%) | Hybrid BM25 + vector + reranker + metadata filter + citations + FastAPI |
| Multi-Agent System (25%) | LangGraph with 6 agents, shared state, feedback loop, parallel groups, observability |
| Extraction Accuracy (20%) | 20-field schema, addendum reconciliation, citation-backed values |
| Robustness (10%) | Graceful failure handling at every layer; works on unseen bids |
| Code Quality (10%) | Modular structure, Pydantic types, env config, unit tests |
| Documentation (10%) | README, architecture diagram, eval report, Q&A log, agent trace |

---

## Proposed Project Structure

```
rfp-intelligence/
├── ingestion/          # Part A
│   ├── html_parser.py
│   ├── pdf_parser.py
│   ├── text_cleaner.py
│   ├── metadata_tagger.py
│   └── ocr.py
├── search/             # Part B
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── bm25_index.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   └── search_api.py
├── agents/             # Part C
│   ├── orchestrator.py
│   ├── retrieval_agent.py
│   ├── extraction_agent.py
│   ├── reconciliation_agent.py
│   ├── validator_agent.py
│   ├── qa_agent.py
│   └── state.py
├── extraction/         # Part D
│   ├── schema.py
│   ├── prompts.py
│   └── output_writer.py
├── api/                # FastAPI app
├── eval/               # Retrieval evaluation
├── tests/              # Unit tests
├── config.py
├── constants.py
├── main.py
├── .env.example
├── requirements.txt
└── docker-compose.yml
```

---

*Open questions / decisions to discuss with team:*
1. Local vs. API-based embeddings? (cost vs. latency vs. privacy)
2. Qdrant local vs. Qdrant Cloud?
3. LLM provider preference? (OpenAI / Gemini / Anthropic / Local)
4. Is Streamlit UI a priority for the bonus?
