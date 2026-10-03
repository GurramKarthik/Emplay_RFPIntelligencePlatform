# Part C — Multi-Agent System

> **Status**: Finalized  
> **Scope**: LangGraph agents, shared state, orchestration, error handling, observability

---

## 1. Overview

```
User Request (bid_folder, mode)
        │
        ▼
  [Orchestrator Agent]  ◀─────────────────────────────────────┐
        │                                                      │ retry (max 3)
        ├── Extraction Mode                                    │
        │       │                                             │
        │   ┌───┴──────────────────────────────────────┐     │
        │   │  PARALLEL Field Groups                    │     │
        │   │  Group 1: Dates & Logistics               │     │
        │   │  Group 2: Commercial & Legal              │     │
        │   │  Group 3: Product & Specs                 │     │
        │   │  Group 4: Identity & Summary              │     │
        │   │  Each: Retrieval Agent → Extraction Agent │     │
        │   └───────────────────┬──────────────────────┘     │
        │                       ▼                             │
        │       [Addendum Reconciliation Agent]               │
        │                       ▼                             │
        │         [Validator / Critic Agent]                  │
        │                ┌──────┴──────────┐                  │
        │           FAILED fields      PASSED fields          │
        │                └──────────────────────────────────▶─┘
        │                                  ▼
        │                          [Part D Output]
        │
        └── QA Mode
                ▼
        Retrieval Agent → Q&A / Report Agent → Cited Answer
```

---

## 2. Agent Roster

| Agent | Responsibility | Output |
|---|---|---|
| **Orchestrator** | Plans task, routes to agents, manages retries, aggregates final output | Task plan + final JSON |
| **Retrieval Agent** | Calls Part B search engine with field-specific queries; returns evidence + citations | `Evidence[]` |
| **Extraction Agent** | Fills structured fields using only retrieved evidence; no hallucination | `DraftField[]` |
| **Addendum Reconciliation Agent** | Compares base RFP vs addendums; applies latest valid change per field | Change log + updated fields |
| **Validator / Critic Agent** | Checks evidence support, format, consistency; flags failures; sends back for retry | Validation report + confidence |
| **Q&A / Report Agent** | Answers free-form NL questions with cited evidence | Cited answer + JSON report |

---

## 3. Shared State (`BidExtractionState`)

```python
class BidExtractionState(BaseModel):
    bid_id: str
    task_mode: Literal["extraction", "qa"]
    query: Optional[str]                          # QA mode only

    # Planning
    field_plan: List[str]                         # fields to extract

    # Evidence
    retrieved_evidence: Dict[str, List[Evidence]] # field → chunks

    # Extraction
    draft_fields: Dict[str, DraftField]           # field → {value, sources, confidence}

    # Reconciliation
    addendum_changes: List[AddendumChange]

    # Validation
    validation_results: Dict[str, ValidationResult]
    retry_counts: Dict[str, int]                  # field → count (max 3)
    field_status: Dict[str, FieldStatus]          # DONE | NOT_FOUND | EXTRACTION_ERROR | VALIDATION_FAILED | RETRY

    # Final
    final_output: Optional[BidRecord]
    trace: List[AgentStep]                        # observability log
```

---

## 4. Field Groups (Parallel Execution)

```
Group 1 — Dates & Logistics:
  Due Date, Pre Bid Meeting, Delivery Date, Term of Bid

Group 2 — Commercial & Legal:
  Bid Bond, Payment Terms, Bid Submission Type,
  Additional Documentation, Contract/Cooperative, MFG Registration

Group 3 — Product & Specs:
  Product, Model_no, Part_no, Product Specification, Installation

Group 4 — Identity & Summary:
  Bid Number, Title, Contact Info, Company Name, Bid Summary
```

All 4 groups run in parallel via LangGraph fan-out. Each group runs its own Retrieval → Extraction pipeline.

---

## 5. Error Handling (No Pipeline Crashes)

**Rule:** Every agent wraps its logic in try/except. The Orchestrator only reads field status — never propagates exceptions.

```python
# At every agent boundary:
try:
    result = agent.run(input)
except LLMTimeoutError:
    state.field_status[field] = "EXTRACTION_ERROR"
    log.error("LLM timeout", field=field)
    # Orchestrator sees EXTRACTION_ERROR → triggers retry if count < 3

except json.JSONDecodeError:
    # Retry once with stricter output prompt
    # If still fails → EXTRACTION_ERROR

except EmptyRetrievalError:
    state.field_status[field] = "NOT_FOUND"
    state.draft_fields[field] = DraftField(
        value=None, sources=[], confidence=0.0, notes="Not found in documents"
    )
```

**Field Status Flow:**
```
PENDING → RETRY (attempt 1,2,3) → DONE
                                 → NOT_FOUND
                                 → EXTRACTION_ERROR (after max retries)
```

Final JSON always has an entry per field — pipeline never crashes.

---

## 6. Feedback / Retry Loop

```
Validator flags field as VALIDATION_FAILED
        │
        ▼
Orchestrator checks retry_counts[field]
        ├── count < 3 → increment count → re-run Retrieval + Extraction for that field only
        └── count >= 3 → mark EXTRACTION_ERROR → move on
```

---

## 7. Observability (`structlog` JSON logs)

**Every agent step logs:**
```json
{
  "timestamp": "2024-03-15T10:23:01Z",
  "agent": "ExtractionAgent",
  "field": "Due Date",
  "input_tokens": 812,
  "output_tokens": 64,
  "latency_ms": 1340,
  "status": "DONE",
  "value": "2024-04-15 17:00 EST",
  "sources": [{"file": "Addendum2.pdf", "page": 1}]
}
```

**Log file:** `./logs/{bid_id}_trace.jsonl`  
**Streamlit:** reads and renders the trace log as a live step-by-step panel.

---

## 8. Agent Framework — LangGraph

| Criterion | LangGraph | Why chosen |
|---|---|---|
| Explicit state graph | ✅ Native | State is `BidExtractionState` |
| Feedback loops / cycles | ✅ Native | Validator → retry is a graph cycle |
| Parallel fan-out | ✅ Native | 4 field groups run concurrently |
| Structured outputs | ✅ Easy | All messages are Pydantic models |
| Observability | structlog | No external service needed |

---

## 9. Guardrails

- Every field value **must** have `sources[]` — if empty → `value = null`, `notes = "Not found"`
- Max retries = **3** per field (prevents infinite loops)
- All agent messages use **Pydantic models** — no raw dicts
- Agents must respond `"Not found in documents"` — never guess

---

## 10. Project File Structure (Part C)

```
agents/
├── state.py                  # BidExtractionState + all sub-models
├── orchestrator.py           # LangGraph graph definition + routing logic
├── retrieval_agent.py        # calls Part B search API with field-specific queries
├── extraction_agent.py       # LLM extraction from retrieved evidence
├── reconciliation_agent.py   # addendum vs base RFP comparison
├── validator_agent.py        # evidence check + format + consistency
├── qa_agent.py               # free-form NL Q&A with citations
└── error_handler.py          # shared try/except wrappers + field status updater

logs/
└── {bid_id}_trace.jsonl      # structlog agent trace per run
```

---

## 11. Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Agent framework | LangGraph | Native cycles, fan-out, explicit state graph |
| Observability | `structlog` JSON logs | No infra; Streamlit-friendly; assignment-compliant |
| Error scope | Per-agent; Orchestrator absorbs | Pipeline never crashes |
| Retry scope | Per-field only (not full bid) | Efficiency |
| Agent messages | Pydantic models | Type safety; structured outputs |
| Parallelism | LangGraph fan-out (4 groups) | Speed; independent field groups |
