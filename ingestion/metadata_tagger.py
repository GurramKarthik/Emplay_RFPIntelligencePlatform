"""
ingestion/metadata_tagger.py
-----------------------------
Infers and attaches structured metadata to each parsed document.

Metadata attached:
  - doc_type     : inferred from filename keywords
  - addendum_number : extracted from filename (e.g. "Addendum 2.pdf" → 2)
  - document_date   : extracted from text using regex + dateparser
"""

import re
from typing import Optional

import dateparser

from constants import (
    ADDENDUM_KEYWORDS,
    AFFIDAVIT_KEYWORDS,
    SPECS_KEYWORDS,
    DOC_TYPE_BID_PAGE,
    DOC_TYPE_RFP,
    DOC_TYPE_ADDENDUM,
    DOC_TYPE_SPECS,
    DOC_TYPE_AFFIDAVIT,
    FORMAT_HTML,
)


def infer_doc_type(file_name: str, source_format: str) -> str:
    """
    Infer the document type from the filename.
    Falls back to 'rfp' for unrecognised PDF files.
    """
    name_lower = file_name.lower()

    if source_format == FORMAT_HTML:
        return DOC_TYPE_BID_PAGE

    if any(kw in name_lower for kw in ADDENDUM_KEYWORDS):
        return DOC_TYPE_ADDENDUM

    if any(kw in name_lower for kw in AFFIDAVIT_KEYWORDS):
        return DOC_TYPE_AFFIDAVIT

    if any(kw in name_lower for kw in SPECS_KEYWORDS):
        return DOC_TYPE_SPECS

    # Default for unrecognised PDFs
    return DOC_TYPE_RFP


def extract_addendum_number(file_name: str) -> Optional[int]:
    """
    Extract the addendum number from a filename.

    Handles patterns like:
      "Addendum 2.pdf", "Addendum_3.pdf", "Amendment2.pdf"
    Returns None if no number found.
    """
    match = re.search(r"(?:addendum|amendment|corrigendum)[_\s-]*(\d+)", file_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_document_date(text: str) -> Optional[str]:
    """
    Extract the first recognisable date from the document text.

    Uses dateparser which handles formats like:
      "March 15, 2024", "03/15/2024", "2024-03-15", "April 5th, 2024"

    Returns an ISO date string (YYYY-MM-DD) or None.
    """
    # Look for common date patterns in the first 2000 characters (header area)
    snippet = text[:2000]

    # Common date patterns to try
    date_patterns = [
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]

    for pattern in date_patterns:
        match = re.search(pattern, snippet, re.IGNORECASE)
        if match:
            parsed = dateparser.parse(match.group(), settings={"RETURN_AS_TIMEZONE_AWARE": False})
            if parsed:
                return parsed.strftime("%Y-%m-%d")

    return None
