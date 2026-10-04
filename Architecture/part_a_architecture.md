# Part A — Document Ingestion & Parsing


> **Scope**: File parsing, failure tracking, chunking, local storage before vector DB injection

---

## 1. Overview

```
Bid Folder (HTML + PDFs)
        │
        ▼
  ┌─────────────┐
  │ File Router │ ── .html ──▶ HTML Parser
  │             │ ── .pdf  ──▶ PDF Parser (pdfplumber → PyMuPDF fallback)
  └─────────────┘
        │
        ▼
  ParseJobTracker (SQLite) ── tracks status, attempts, errors per file
        │
        ▼
  Text Cleaner → Metadata Tagger → ParsedDocument
        │
        ▼
  Chunker → chunks.jsonl (local cache: ./data/chunks/{bid_id}_chunks.jsonl)
        │
        ▼
  ──── Hand-off to Part B (Embedder + Vector DB) ────
```

---

## 2. Components

| Component | Role | Tech |
|---|---|---|
| **File Router** | Detects file type, routes to correct parser | `pathlib` |
| **HTML Parser** | Extracts text + tables from BidNet HTML pages | `trafilatura` (primary), `BeautifulSoup4` (fallback) |
| **PDF Parser** | Extracts text + tables from RFP/addendum PDFs | `pdfplumber` (primary), `PyMuPDF` (fallback) |
| **OCR Fallback** *(bonus)* | Handles scanned/image-only pages | `pytesseract` + `pdf2image` |
| **Text Cleaner** | Strips headers/footers, fixes hyphenation, normalizes whitespace | Custom regex pipeline |
| **Metadata Tagger** | Attaches structured metadata to each parsed document | Custom |
| **ParseJobTracker** | SQLite-based per-file status tracker with retry logic | `sqlite3` |
| **Chunker** | Splits ParsedDocuments into retrieval units | Custom (section-aware + sliding window fallback) |
| **Failure Notifier** | Collects all FAILED files at end of run and returns summary | Custom |

---

## 3. ParseJobTracker — SQLite Schema

```sql
CREATE TABLE parse_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    bid_id          TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING | IN_PROGRESS | DONE | FAILED
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(bid_id, file_name)
);
```

**Status transitions:**
```
PENDING → IN_PROGRESS → DONE
                      → FAILED (if attempts >= MAX_RETRIES)
                      → PENDING (re-queued if attempts < MAX_RETRIES)
```

**Config (`config.py`):**
```python
MAX_RETRIES = 3          # max parse attempts per file
TRACKER_DB_PATH = "./data/parse_jobs.db"
CHUNKS_OUTPUT_DIR = "./data/chunks/"
```

---

## 4. Retry & Failure Flow

```
for each file in bid_folder:
    tracker.register(bid_id, file_name)   # status = PENDING

    while attempts < MAX_RETRIES:
        try:
            result = parse(file)
            tracker.mark_done(bid_id, file_name)
            break
        except Exception as e:
            attempts += 1
            tracker.update(bid_id, file_name, attempts, error=str(e))
            if attempts >= MAX_RETRIES:
                tracker.mark_failed(bid_id, file_name)

# After all files processed:
failed_files = tracker.get_failed(bid_id)
notify_user(failed_files)
```

**End-of-run notification payload:**
```json
{
  "bid_id": "Bid1",
  "processed": ["RFP_FINAL.pdf", "Addendum1.pdf", "bid_page.html"],
  "failed": [
    {
      "file": "Addendum2.pdf",
      "attempts": 3,
      "reason": "No text layer detected — possible scanned PDF"
    }
  ]
}
```

---

## 5. Metadata Schema (per ParsedDocument)

```python
{
  "bid_id": "Bid1",
  "file_name": "Addendum 2.pdf",
  "doc_type": "addendum",          # bid_page | rfp | addendum | specs | affidavit
  "addendum_number": 2,            # None for non-addenda
  "page_number": 1,
  "document_date": "2024-03-15",   # extracted via regex + dateparser; None if not found
  "source_format": "pdf"           # pdf | html
}
```

---

## 6. Chunking Strategy

```
Strategy: Section/Heading-aware → Sliding Window fallback

1. PRIMARY — Heading-aware:
   - Detect section headings via regex + font-size heuristics (PyMuPDF)
   - One chunk = heading + body text
   - Tables: always atomic (never split mid-table)
   - Table chunk gets doc_type = "table" + 50-token surrounding context

2. FALLBACK — Sliding Window:
   - chunk_size  = 512 tokens
   - overlap     = 64 tokens

chunk_id format:
  "{bid_id}__{file_name}__{page_num}__{chunk_index}"
```

---

## 7. Local Chunk Storage (Checkpoint Before Part B)

**Why local first:**
- If embedding API fails mid-run → resume from JSONL, no re-parsing needed
- Allows chunk inspection/debugging before DB insertion
- Clean separation between Part A (parse) and Part B (index)

**Storage path:** `./data/chunks/{bid_id}_chunks.jsonl`  
**Format:** One JSON object per line

```jsonl
{"chunk_id": "Bid1__RFP_FINAL.pdf__1__0", "text": "...", "metadata": {...}}
{"chunk_id": "Bid1__RFP_FINAL.pdf__1__1", "text": "...", "metadata": {...}}
```

---

## 8. Failure Points & Mitigations

| Failure | Mitigation |
|---|---|
| Scanned PDF (no text layer) | Detect via empty text → OCR fallback (bonus) → log if still empty |
| Corrupt / password-protected PDF | try/except → log error → tracker marks FAILED after MAX_RETRIES |
| HTML with heavy JavaScript | `trafilatura` handles boilerplate; `BS4` fallback for structure |
| Multi-column PDF layout | `PyMuPDF` word-sort mode as fallback after `pdfplumber` |
| Repeated headers/footers in chunks | Strip via frequency analysis across pages |
| Chunking yields empty chunk | Skip + log; never insert empty chunks into JSONL |

---

## 9. Project File Structure (Part A)

```
ingestion/
├── file_router.py          # detects file type, dispatches to correct parser
├── html_parser.py          # trafilatura + BS4
├── pdf_parser.py           # pdfplumber + PyMuPDF fallback
├── ocr.py                  # pytesseract + pdf2image (bonus)
├── text_cleaner.py         # regex pipeline
├── metadata_tagger.py      # attaches metadata dict to ParsedDocument
├── chunker.py              # section-aware + sliding window
├── tracker.py              # SQLite ParseJobTracker
└── notifier.py             # builds and returns failure summary

data/
├── parse_jobs.db           # SQLite tracker DB
└── chunks/
    ├── Bid1_chunks.jsonl
    └── Bid2_chunks.jsonl
```

---

## 10. Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Failure tracking | SQLite (`parse_jobs.db`) | Persistent across sessions; queryable; lightweight |
| Max retries | 3 (configurable) | Balances resilience vs. time waste |
| Re-parse scope | Only failed files, not full bid folder | Efficiency |
| Chunk storage | Local JSONL before vector DB | Resume-safe checkpoint; debuggable |
| Primary PDF parser | `pdfplumber` | Best table extraction |
| Table format | Markdown | Preserves structure for embedding + LLM context |
