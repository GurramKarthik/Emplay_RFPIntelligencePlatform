"""
ingestion/text_cleaner.py
-------------------------
Cleans raw extracted text before chunking:
  - Removes repeated headers/footers (frequency-based)
  - Fixes broken hyphenation across lines
  - Normalises whitespace and control characters
  - Collapses excessive blank lines
"""

import re
from collections import Counter
from typing import List


# ── Public API ────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to a single text string.
    Designed to be called on the raw text of each page before chunking.
    """
    text = _remove_control_characters(text)
    text = _fix_hyphenation(text)
    text = _normalize_whitespace(text)
    return text.strip()


def remove_headers_footers(pages_text: List[str]) -> List[str]:
    """
    Detect and strip repeated lines that appear across many pages
    (typical headers and footers).

    Strategy: a line that appears in > 40% of pages is considered a header/footer.
    Returns a new list with those lines removed from each page.
    """
    if len(pages_text) < 2:
        return pages_text  # nothing to compare with one page

    # Count how many pages each line appears in
    line_counts: Counter = Counter()
    for page in pages_text:
        # Use a set so we only count a line once per page
        unique_lines = {line.strip() for line in page.splitlines() if line.strip()}
        line_counts.update(unique_lines)

    threshold = max(2, int(len(pages_text) * 0.4))
    repeated_lines = {line for line, count in line_counts.items() if count >= threshold}

    # Remove the repeated lines from every page
    cleaned_pages = []
    for page in pages_text:
        filtered = [
            line for line in page.splitlines()
            if line.strip() not in repeated_lines
        ]
        cleaned_pages.append("\n".join(filtered))

    return cleaned_pages


# ── Private Helpers ───────────────────────────────────────────────────────────

def _remove_control_characters(text: str) -> str:
    """Strip non-printable control characters except newlines and tabs."""
    return re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", "", text)


def _fix_hyphenation(text: str) -> str:
    """
    Rejoin words broken with a hyphen at end of line.
    Example: "specifi-\ncation" → "specification"
    """
    return re.sub(r"-\n(\S)", r"\1", text)


def _normalize_whitespace(text: str) -> str:
    """
    - Replace tabs with a space
    - Collapse multiple spaces into one
    - Collapse 3+ consecutive blank lines into 2
    """
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)         # collapse multiple spaces
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse excessive blank lines
    return text
