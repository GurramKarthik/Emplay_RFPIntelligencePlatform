"""
tests/test_part_a.py
--------------------
Unit tests for Part A — ingestion, parsing, chunking.

Run with:
  pytest tests/test_part_a.py -v
"""

import os
import json
import tempfile
import pytest

from ingestion.text_cleaner import clean_text, remove_headers_footers
from ingestion.metadata_tagger import infer_doc_type, extract_addendum_number, extract_document_date
from ingestion.chunker import chunk_document
from models import ParsedDocument, ParsedPage
from constants import DOC_TYPE_ADDENDUM, DOC_TYPE_RFP, DOC_TYPE_BID_PAGE, FORMAT_PDF, FORMAT_HTML


# ── text_cleaner ──────────────────────────────────────────────────────────────

class TestTextCleaner:
    def test_fix_hyphenation(self):
        text = "specifi-\ncation is important"
        result = clean_text(text)
        assert "specification" in result

    def test_normalize_whitespace(self):
        text = "hello    world\n\n\n\nextra"
        result = clean_text(text)
        assert "  " not in result            # no double spaces
        assert result.count("\n\n\n") == 0   # max 2 consecutive newlines

    def test_remove_repeated_headers(self):
        pages = [
            "Page Header\nContent of page 1",
            "Page Header\nContent of page 2",
            "Page Header\nContent of page 3",
        ]
        result = remove_headers_footers(pages)
        for page in result:
            assert "Page Header" not in page

    def test_single_page_not_affected(self):
        pages = ["Only one page here"]
        result = remove_headers_footers(pages)
        assert result == pages


# ── metadata_tagger ───────────────────────────────────────────────────────────

class TestMetadataTagger:
    def test_doc_type_html(self):
        assert infer_doc_type("bid_page.html", FORMAT_HTML) == DOC_TYPE_BID_PAGE

    def test_doc_type_addendum(self):
        assert infer_doc_type("Addendum 2.pdf", FORMAT_PDF) == DOC_TYPE_ADDENDUM

    def test_doc_type_rfp_default(self):
        assert infer_doc_type("RFP_FINAL.pdf", FORMAT_PDF) == DOC_TYPE_RFP

    def test_extract_addendum_number(self):
        assert extract_addendum_number("Addendum 2.pdf") == 2
        assert extract_addendum_number("Amendment_3.pdf") == 3
        assert extract_addendum_number("RFP_FINAL.pdf") is None

    def test_extract_document_date_iso(self):
        text = "Issued on 2024-03-15. All bids must be submitted."
        date = extract_document_date(text)
        assert date == "2024-03-15"

    def test_extract_document_date_written(self):
        text = "Date: March 15, 2024. Responses due by April 1."
        date = extract_document_date(text)
        assert date is not None
        assert "2024" in date

    def test_extract_document_date_none(self):
        text = "No date information here at all."
        date = extract_document_date(text)
        assert date is None


# ── chunker ───────────────────────────────────────────────────────────────────

class TestChunker:
    def _make_doc(self, text: str, tables=None) -> ParsedDocument:
        return ParsedDocument(
            bid_id="TestBid",
            file_name="test.pdf",
            file_path="/tmp/test.pdf",
            doc_type=DOC_TYPE_RFP,
            addendum_number=None,
            document_date=None,
            source_format=FORMAT_PDF,
            pages=[ParsedPage(page_number=1, text=text, tables=tables or [])],
        )

    def test_chunks_not_empty(self):
        doc = self._make_doc("This is a test document with some content about the bid.")
        chunks = chunk_document(doc)
        assert len(chunks) > 0

    def test_chunk_ids_are_unique(self):
        text = "Section 1\nContent here.\n\nSection 2\nMore content here."
        doc = self._make_doc(text)
        chunks = chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # all unique

    def test_table_chunk_marked(self):
        table_md = "| Col1 | Col2 |\n|---|---|\n| A | B |"
        doc = self._make_doc("Some text before the table.", tables=[table_md])
        chunks = chunk_document(doc)
        table_chunks = [c for c in chunks if c.is_table]
        assert len(table_chunks) == 1
        assert table_md in table_chunks[0].text

    def test_chunk_inherits_metadata(self):
        doc = self._make_doc("Addendum content with updated dates.")
        doc.addendum_number = 2
        doc.doc_type = DOC_TYPE_ADDENDUM
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.bid_id == "TestBid"
            assert chunk.addendum_number == 2

    def test_chunk_text_not_empty(self):
        doc = self._make_doc("Non-empty content for a real document.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.text.strip() != ""
