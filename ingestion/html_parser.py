"""
ingestion/html_parser.py
------------------------
Parses HTML bid pages into a ParsedDocument.

Strategy:
  1. Try trafilatura  — handles boilerplate/navigation removal well
  2. Fallback to BeautifulSoup4 — for pages trafilatura cannot parse

Tables in HTML are extracted and converted to Markdown format.
"""

import re
from typing import List, Optional

import structlog
from bs4 import BeautifulSoup

from ingestion.text_cleaner import clean_text, remove_headers_footers
from ingestion.metadata_tagger import infer_doc_type, extract_addendum_number, extract_document_date
from models import ParsedDocument, ParsedPage
from constants import FORMAT_HTML

log = structlog.get_logger(__name__)


def parse_html(bid_id: str, file_path: str) -> Optional[ParsedDocument]:
    """
    Parse an HTML file into a ParsedDocument.

    Returns None if the file produces no usable text (logged as a failure).
    """
    file_name = _get_file_name(file_path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_html = f.read()
    except OSError as e:
        log.error("html_parser.read_error", file=file_name, error=str(e))
        raise

    # ── Step 1: Extract main text via trafilatura ──────────────────────────────
    text = _extract_with_trafilatura(raw_html)

    # ── Step 2: Fallback to BeautifulSoup if trafilatura yields nothing ────────
    if not text or len(text.strip()) < 50:
        log.warning("html_parser.trafilatura_fallback", file=file_name)
        text = _extract_with_bs4(raw_html)

    if not text or len(text.strip()) < 20:
        log.error("html_parser.empty_content", file=file_name)
        raise ValueError(f"No usable text extracted from HTML: {file_name}")

    # ── Step 3: Extract tables ─────────────────────────────────────────────────
    tables = _extract_tables_as_markdown(raw_html)

    # ── Step 4: Clean text ────────────────────────────────────────────────────
    cleaned_text = clean_text(text)

    # ── Step 5: Build ParsedPage (HTML treated as single page) ────────────────
    page = ParsedPage(page_number=1, text=cleaned_text, tables=tables)

    # ── Step 6: Attach metadata ───────────────────────────────────────────────
    doc_type = infer_doc_type(file_name, FORMAT_HTML)
    addendum_number = extract_addendum_number(file_name)
    document_date = extract_document_date(cleaned_text)

    log.info("html_parser.success", file=file_name, tables_found=len(tables))

    return ParsedDocument(
        bid_id=bid_id,
        file_name=file_name,
        file_path=file_path,
        doc_type=doc_type,
        addendum_number=addendum_number,
        document_date=document_date,
        source_format=FORMAT_HTML,
        pages=[page],
    )


# ── Private Helpers ───────────────────────────────────────────────────────────

def _extract_with_trafilatura(raw_html: str) -> str:
    """Use trafilatura to strip boilerplate and extract main content."""
    try:
        import trafilatura
        return trafilatura.extract(raw_html, include_tables=False) or ""
    except Exception as e:
        log.warning("html_parser.trafilatura_error", error=str(e))
        return ""


def _extract_with_bs4(raw_html: str) -> str:
    """
    BeautifulSoup fallback — removes script/style tags and returns visible text.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove script, style, nav, footer, header elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def _extract_tables_as_markdown(raw_html: str) -> List[str]:
    """
    Find all <table> elements and convert each to a Markdown table string.
    Returns a list of markdown table strings (one per table).
    """
    soup = BeautifulSoup(raw_html, "lxml")
    markdown_tables: List[str] = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue

        md_rows: List[str] = []
        for i, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            cell_texts = [_clean_cell(cell.get_text()) for cell in cells]
            if not any(cell_texts):
                continue
            md_rows.append("| " + " | ".join(cell_texts) + " |")

            # Add separator after the header row
            if i == 0:
                md_rows.append("|" + "|".join(["---"] * len(cell_texts)) + "|")

        if md_rows:
            markdown_tables.append("\n".join(md_rows))

    return markdown_tables


def _clean_cell(text: str) -> str:
    """Strip extra whitespace from a table cell."""
    return re.sub(r"\s+", " ", text).strip()


def _get_file_name(file_path: str) -> str:
    """Extract just the filename from a full path."""
    import os
    return os.path.basename(file_path)
