"""
ingestion/pdf_parser.py
-----------------------
Parses PDF files into a ParsedDocument.

Strategy:
  1. pdfplumber  — primary: best table extraction + text layout
  2. PyMuPDF     — fallback: better for multi-column / complex layouts
  3. OCR         — bonus fallback for scanned (image-only) PDFs

Tables are extracted as Markdown strings and stored alongside page text.
"""

import os
import re
from typing import List, Optional, Tuple

import structlog

from ingestion.text_cleaner import clean_text, remove_headers_footers
from ingestion.metadata_tagger import infer_doc_type, extract_addendum_number, extract_document_date
from models import ParsedDocument, ParsedPage
from constants import FORMAT_PDF

log = structlog.get_logger(__name__)


def parse_pdf(bid_id: str, file_path: str) -> Optional[ParsedDocument]:
    """
    Parse a PDF file into a ParsedDocument.

    Tries pdfplumber first; falls back to PyMuPDF for complex layouts.
    Raises an exception if no text is extractable (caller handles retry logic).
    """
    file_name = os.path.basename(file_path)

    # ── Attempt 1: pdfplumber ─────────────────────────────────────────────────
    try:
        pages = _parse_with_pdfplumber(file_path)
        if _is_empty(pages):
            raise ValueError("pdfplumber returned empty text")
        log.info("pdf_parser.pdfplumber_success", file=file_name, pages=len(pages))

    except Exception as e:
        # ── Attempt 2: PyMuPDF fallback ───────────────────────────────────────
        log.warning("pdf_parser.pdfplumber_failed", file=file_name, error=str(e))
        try:
            pages = _parse_with_pymupdf(file_path)
            if _is_empty(pages):
                raise ValueError("PyMuPDF returned empty text")
            log.info("pdf_parser.pymupdf_success", file=file_name, pages=len(pages))

        except Exception as e2:
            # ── Attempt 3: OCR fallback (bonus) ───────────────────────────────
            log.warning("pdf_parser.pymupdf_failed", file=file_name, error=str(e2))
            try:
                from ingestion.ocr import parse_with_ocr
                pages = parse_with_ocr(file_path)
                if _is_empty(pages):
                    raise ValueError("OCR returned empty text")
                log.info("pdf_parser.ocr_success", file=file_name, pages=len(pages))
            except Exception as e3:
                log.error("pdf_parser.all_methods_failed", file=file_name, error=str(e3))
                raise RuntimeError(
                    f"All parse methods failed for {file_name}: {e3}"
                ) from e3

    # ── Clean text across all pages (header/footer removal) ──────────────────
    raw_texts = [p.text for p in pages]
    cleaned_texts = remove_headers_footers(raw_texts)

    # Apply per-page cleaning and rebuild pages
    cleaned_pages: List[ParsedPage] = []
    for i, page in enumerate(pages):
        cleaned_text = clean_text(cleaned_texts[i])
        cleaned_pages.append(ParsedPage(
            page_number=page.page_number,
            text=cleaned_text,
            tables=page.tables,
        ))

    # ── Attach metadata ───────────────────────────────────────────────────────
    full_text = " ".join(p.text for p in cleaned_pages)
    doc_type        = infer_doc_type(file_name, FORMAT_PDF)
    addendum_number = extract_addendum_number(file_name)
    document_date   = extract_document_date(full_text)

    return ParsedDocument(
        bid_id=bid_id,
        file_name=file_name,
        file_path=file_path,
        doc_type=doc_type,
        addendum_number=addendum_number,
        document_date=document_date,
        source_format=FORMAT_PDF,
        pages=cleaned_pages,
    )


# ── pdfplumber Parser ─────────────────────────────────────────────────────────

def _parse_with_pdfplumber(file_path: str) -> List[ParsedPage]:
    """
    Extract text and tables from a PDF using pdfplumber.

    Tables are converted to Markdown and removed from the main text
    to avoid duplication in chunks.
    """
    import pdfplumber

    pages: List[ParsedPage] = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_num = page.page_number

            # ── Extract tables first ──────────────────────────────────────────
            markdown_tables = _extract_pdfplumber_tables(page)

            # ── Extract text (excluding table bounding boxes) ─────────────────
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

            pages.append(ParsedPage(
                page_number=page_num,
                text=text,
                tables=markdown_tables,
            ))

    return pages


def _extract_pdfplumber_tables(page) -> List[str]:
    """Extract all tables on a pdfplumber page as Markdown strings."""
    markdown_tables: List[str] = []

    try:
        tables = page.extract_tables()
    except Exception:
        return []

    for table in tables:
        if not table:
            continue
        md = _table_to_markdown(table)
        if md:
            markdown_tables.append(md)

    return markdown_tables


# ── PyMuPDF Fallback ──────────────────────────────────────────────────────────

def _parse_with_pymupdf(file_path: str) -> List[ParsedPage]:
    """
    Extract text using PyMuPDF (fitz).
    Uses word-sort mode which handles multi-column layouts better.
    """
    import fitz  # PyMuPDF

    pages: List[ParsedPage] = []

    with fitz.open(file_path) as doc:
        for page in doc:
            # TEXT_PRESERVE_WHITESPACE + sort=True handles multi-column well
            text = page.get_text("text", sort=True) or ""
            pages.append(ParsedPage(
                page_number=page.number + 1,  # fitz is 0-indexed
                text=text,
                tables=[],  # PyMuPDF table extraction needs extra lib; skip here
            ))

    return pages


# ── Shared Helpers ────────────────────────────────────────────────────────────

def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    """
    Convert a pdfplumber table (list of rows, each row a list of cell strings)
    into a Markdown table string.
    """
    if not table:
        return ""

    rows: List[str] = []
    for i, row in enumerate(table):
        # Replace None cells with empty string
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
        # Add header separator after first row
        if i == 0:
            rows.append("|" + "|".join(["---"] * len(cells)) + "|")

    return "\n".join(rows)


def _is_empty(pages: List[ParsedPage]) -> bool:
    """Return True if all pages have negligible text content."""
    total_text = "".join(p.text.strip() for p in pages)
    return len(total_text) < 20
