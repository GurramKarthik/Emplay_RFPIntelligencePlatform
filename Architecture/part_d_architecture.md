# Part D — Structured Information Extraction

> **Status**: Finalized  
> **Scope**: Output schema, extraction prompts, addendum reconciliation, confidence scoring, output writer

---

## 1. Overview

```
Part C (Validated DraftFields)
        │
        ▼
  [Output Writer]
        │
        ├── Applies addendum reconciliation (final value = latest non-null)
        ├── Builds BidRecord (Pydantic)
        ├── Writes → ./output/{bid_id}_output.json
        └── Returns BidRecord to API / Streamlit
```

---

## 2. Output Schema (Pydantic)

```python
class Source(BaseModel):
    file: str
    page: int

class FieldValue(BaseModel):
    value: Optional[str]
    sources: List[Source]           # always present; empty list if not found
    confidence: float               # 0.0 – 1.0
    notes: Optional[str]           # "Extended by Addendum 2", "Not found", etc.

class AddendumChange(BaseModel):
    field: str
    original_value: Optional[str]
    updated_value: Optional[str]
    addendum_number: int
    source: Source

class ValidationSummary(BaseModel):
    passed: int
    failed: int
    not_found: int

class BidRecord(BaseModel):
    bid_id: str
    fields: Dict[str, FieldValue]   # all 20 fields
    addendum_changes: List[AddendumChange]
    validation: ValidationSummary
```

---

## 3. 20 Fields to Extract

| Field | Description |
|---|---|
| `Bid Number` | Official solicitation / RFP identifier |
| `Title` | Full title of the bid |
| `Due Date` | Final submission deadline (with time + timezone); reflects addendum changes |
| `Bid Submission Type` | Electronic portal / sealed paper / email etc. |
| `Term of Bid` | Contract duration + renewal options |
| `Pre Bid Meeting` | Date, time, location, mandatory or not; "None" if absent |
| `Installation` | Whether installation/imaging services are required |
| `Bid Bond` | Required bond/security amount or %; "Not required" if absent |
| `Delivery Date` | Required delivery date or window after award |
| `Payment Terms` | Net 30, invoicing conditions etc. |
| `Additional Documentation` | Affidavits, forms, certificates, W-9 etc. |
| `MFG for Registration` | Manufacturer(s) bidder must be registered with |
| `Contract or Cooperative` | Referenced cooperative/state contract vehicle |
| `Model_no` | Model number(s) of requested product(s) |
| `Part_no` | Part / SKU number(s) |
| `Product` | Product name(s) and quantity |
| `contact_info` | Procurement contact: name, email, phone |
| `company_name` | Issuing organization / agency name |
| `Bid Summary` | 3–6 sentence system-generated summary |
| `Product Specification` | Key technical specs (CPU, RAM, storage, display, warranty) |

---

## 4. Extraction Prompt Pattern

```
System:
  You are an extraction agent. Extract ONLY from the evidence below.
  If the value is not in the evidence, return null with a reason.
  Never guess or infer values not explicitly stated.
  Return valid JSON only.

Evidence:
  [chunk 1 — file: Addendum2.pdf, page: 1]
  "The due date has been extended to April 15, 2024 at 5:00 PM EST..."

  [chunk 2 — file: RFP_FINAL.pdf, page: 4]
  "..."

Task:
  Extract the field "Due Date" (Final submission deadline, including time and timezone).
  Return JSON:
  {
    "value": "<extracted value or null>",
    "sources": [{"file": "<filename>", "page": <page_number>}],
    "confidence": <0.0–1.0>,
    "notes": "<reason if null, or addendum note if changed>"
  }
```

---

## 5. Addendum Reconciliation Logic

```
For each field:
  1. Extract value from base RFP        → v_base
  2. For each addendum (sorted ASC by addendum_number):
       Check if addendum modifies this field → v_addendum
  3. Final value = last non-null v_addendum  OR  v_base
  4. Log change: {field, original, updated, addendum_number, source}
```

**Example:**
```
Due Date in RFP_FINAL.pdf   → "March 30, 2024"
Due Date in Addendum 1      → null (not mentioned)
Due Date in Addendum 2      → "April 15, 2024 17:00 EST"  ← WINS

Final value: "2024-04-15 17:00 EST"
notes: "Extended by Addendum 2 (original: 2024-03-30)"
```

---

## 6. Confidence Score Heuristics

| Range | Meaning |
|---|---|
| `0.90 – 1.00` | Exact value found; single unambiguous source |
| `0.70 – 0.89` | Value found but needs interpretation (date format, implicit) |
| `0.50 – 0.69` | Inferred from context; validator flags for review |
| `0.00 – 0.49` | Weak or no evidence → likely `null` + "Not found" |

---

## 7. Null / Not Found Handling

```json
"Bid Bond": {
  "value": null,
  "sources": [],
  "confidence": 0.80,
  "notes": "Not found in documents"
}
```

- `value = null` is valid and expected
- `sources = []` when nothing retrieved
- `notes` always explains why

---

## 8. Sample Output JSON

```json
{
  "bid_id": "Bid1",
  "fields": {
    "Due Date": {
      "value": "2024-04-15 17:00 EST",
      "sources": [{"file": "Addendum2.pdf", "page": 1}],
      "confidence": 0.95,
      "notes": "Extended by Addendum 2 (original: 2024-03-30)"
    },
    "Bid Bond": {
      "value": null,
      "sources": [],
      "confidence": 0.80,
      "notes": "Not found in documents"
    }
  },
  "addendum_changes": [
    {
      "field": "Due Date",
      "original_value": "2024-03-30",
      "updated_value": "2024-04-15 17:00 EST",
      "addendum_number": 2,
      "source": {"file": "Addendum2.pdf", "page": 1}
    }
  ],
  "validation": {
    "passed": 18,
    "failed": 0,
    "not_found": 2
  }
}
```

---

## 9. Project File Structure (Part D)

```
extraction/
├── schema.py           # Pydantic models: FieldValue, BidRecord, AddendumChange, etc.
├── prompts.py          # Extraction prompt templates per field (or field group)
└── output_writer.py    # Applies reconciliation → builds BidRecord → writes JSON

output/
├── Bid1_output.json
└── Bid2_output.json
```

---

## 10. Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Output format | Pydantic → JSON | Type-safe; validated; serializable |
| Null handling | `value=null` + `notes` | Assignment requirement; never silently skip |
| Addendum precedence | Latest addendum wins (ASC sort) | Correct override semantics |
| Confidence scoring | LLM-assigned + heuristic thresholds | Transparent; reviewable |
| Prompt scope | One prompt per field (not all 20 at once) | Better precision; smaller context |
