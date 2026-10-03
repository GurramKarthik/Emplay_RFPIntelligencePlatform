"""
models.py
---------
Shared Pydantic models used across all parts of the pipeline.
All inter-module data exchange uses these typed models — no raw dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Part A Models ─────────────────────────────────────────────────────────────

class ParsedPage(BaseModel):
    """Represents one page extracted from a document."""
    page_number: int
    text: str                        # cleaned plain text
    tables: List[str] = Field(default_factory=list)  # markdown-formatted tables


class ParsedDocument(BaseModel):
    """
    Fully parsed document — output of Part A ingestion.
    Passed to the Chunker, then handed off to Part B.
    """
    bid_id: str
    file_name: str
    file_path: str
    doc_type: str                    # bid_page | rfp | addendum | specs | affidavit
    addendum_number: Optional[int]   # only set for addenda
    document_date: Optional[str]     # ISO date string or None
    source_format: str               # pdf | html
    pages: List[ParsedPage]


class Chunk(BaseModel):
    """
    A single retrieval unit — output of the Chunker.
    Written to {bid_id}_chunks.jsonl and later indexed into Qdrant + BM25.
    """
    chunk_id: str                    # "{bid_id}__{file_name}__{page_num}__{chunk_index}"
    bid_id: str
    file_name: str
    page_number: int
    chunk_index: int
    text: str                        # chunk content (plain text or markdown table)
    doc_type: str
    addendum_number: Optional[int]
    document_date: Optional[str]
    source_format: str
    is_table: bool = False           # True for atomic table chunks


# ── Parse Job / Failure Models ────────────────────────────────────────────────

class FailedFile(BaseModel):
    """Represents a file that exhausted all parse retries."""
    file: str
    attempts: int
    reason: str


class IngestionSummary(BaseModel):
    """End-of-run summary returned to the caller (API / Streamlit / CLI)."""
    bid_id: str
    processed: List[str]             # file names successfully parsed
    failed: List[FailedFile]         # files that could not be parsed


# ── Part D Models (defined here so Part C agents can import them) ─────────────

class Source(BaseModel):
    """Citation pointing to the exact file and page a value was extracted from."""
    file: str
    page: int


class FieldValue(BaseModel):
    """Structured value for one extracted field."""
    value: Optional[str]
    sources: List[Source] = Field(default_factory=list)
    confidence: float = 0.0          # 0.0 – 1.0
    notes: Optional[str] = None      # "Extended by Addendum 2", "Not found", etc.


class AddendumChange(BaseModel):
    """Records a field value that was overridden by an addendum."""
    field: str
    original_value: Optional[str]
    updated_value: Optional[str]
    addendum_number: int
    source: Source


class ValidationSummary(BaseModel):
    passed: int = 0
    failed: int = 0
    not_found: int = 0


class BidRecord(BaseModel):
    """
    Final structured output for one bid — written to output/{bid_id}_output.json.
    """
    bid_id: str
    fields: Dict[str, FieldValue]
    addendum_changes: List[AddendumChange] = Field(default_factory=list)
    validation: ValidationSummary = Field(default_factory=ValidationSummary)
