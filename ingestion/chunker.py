"""
ingestion/chunker.py
--------------------
Splits a ParsedDocument into retrieval-ready Chunk objects.

Strategy (in order of preference):
  1. Section/Heading-aware chunking
     - Detects headings via regex (numbered sections, ALL-CAPS lines, underlined lines)
     - One chunk = heading + its body text
     - Advantage: natural retrieval unit for structured RFP documents

  2. Sliding Window fallback
     - Used when no headings are detected on a page
     - chunk_size = CHUNK_SIZE_TOKENS, overlap = CHUNK_OVERLAP_TOKENS

Tables are always treated as atomic chunks — never split mid-table.
Each table chunk gets a 50-token context window from surrounding text.
"""

import re
from typing import List, Tuple

import tiktoken
import structlog

from models import ParsedDocument, Chunk, ParsedPage
from constants import DOC_TYPE_TABLE
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TABLE_CONTEXT_TOKENS

log = structlog.get_logger(__name__)

# Use cl100k_base tokenizer (compatible with OpenAI models + BGE)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Regex patterns for detecting section headings
_HEADING_PATTERNS = [
    re.compile(r"^\d+(\.\d+)*[\s\.\)]+\S"),          # "1.", "1.2.", "1.2.3 Title"
    re.compile(r"^[A-Z][A-Z\s]{4,}$"),                # "SECTION TITLE" (all caps)
    re.compile(r"^(?:SECTION|PART|ARTICLE)\s+\w+", re.IGNORECASE),
]


def chunk_document(doc: ParsedDocument) -> List[Chunk]:
    """
    Chunk a ParsedDocument into a flat list of Chunk objects.
    Processes all pages and returns chunks in document order.
    """
    all_chunks: List[Chunk] = []
    global_chunk_index = 0

    for page in doc.pages:
        # ── Table chunks (always atomic) ──────────────────────────────────────
        table_chunks, global_chunk_index = _chunk_tables(
            doc, page, global_chunk_index
        )
        all_chunks.extend(table_chunks)

        # ── Text chunks (heading-aware or sliding window) ─────────────────────
        text_chunks, global_chunk_index = _chunk_text(
            doc, page, global_chunk_index
        )
        all_chunks.extend(text_chunks)

    log.info(
        "chunker.done",
        bid_id=doc.bid_id,
        file=doc.file_name,
        total_chunks=len(all_chunks),
    )
    return all_chunks


# ── Table Chunking ────────────────────────────────────────────────────────────

def _split_markdown_table(table_md: str, max_tokens: int = CHUNK_SIZE_TOKENS) -> List[str]:
    """Splits a large markdown table into smaller tables, preserving the header."""
    lines = table_md.strip().split('\n')
    if len(lines) <= 2:
        return [table_md] # Not a real table or just headers
        
    header = lines[0]
    separator = lines[1]
    rows = lines[2:]
    
    result = []
    current_chunk_lines = [header, separator]
    
    for row in rows:
        current_chunk_lines.append(row)
        if _count_tokens("\n".join(current_chunk_lines)) >= max_tokens:
            result.append("\n".join(current_chunk_lines))
            current_chunk_lines = [header, separator]
            
    if len(current_chunk_lines) > 2:
        result.append("\n".join(current_chunk_lines))
        
    return result if result else [table_md]


def _chunk_tables(
    doc: ParsedDocument, page: ParsedPage, start_index: int
) -> Tuple[List[Chunk], int]:
    """
    Create chunk(s) per table on this page.
    Adds TABLE_CONTEXT_TOKENS of surrounding plain text as context.
    If a table is too large, it is split row-by-row.
    """
    chunks: List[Chunk] = []
    idx = start_index
    context = _truncate_tokens(page.text, TABLE_CONTEXT_TOKENS)

    for table_md in page.tables:
        sub_tables = _split_markdown_table(table_md, CHUNK_SIZE_TOKENS)
        for sub_table in sub_tables:
            chunk_text = f"{context}\n\n{sub_table}" if context else sub_table
            chunks.append(_make_chunk(doc, page.page_number, idx, chunk_text, is_table=True))
            idx += 1

    return chunks, idx


# ── Text Chunking ─────────────────────────────────────────────────────────────

def _chunk_text(
    doc: ParsedDocument, page: ParsedPage, start_index: int
) -> Tuple[List[Chunk], int]:
    """
    Chunk the plain text of a page.
    Uses heading-aware chunking first; falls back to sliding window.
    """
    text = page.text.strip()
    if not text:
        return [], start_index

    sections = _split_by_headings(text)

    chunks: List[Chunk] = []
    idx = start_index

    if len(sections) > 1:
        # Heading-aware: one chunk per section
        for section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue
            # If a section is too large, sub-chunk it with sliding window
            if _count_tokens(section_text) > CHUNK_SIZE_TOKENS:
                sub_chunks = _sliding_window(section_text)
                for sub in sub_chunks:
                    chunks.append(_make_chunk(doc, page.page_number, idx, sub))
                    idx += 1
            else:
                chunks.append(_make_chunk(doc, page.page_number, idx, section_text))
                idx += 1
    else:
        # No headings found — use sliding window
        windows = _sliding_window(text)
        for window in windows:
            chunks.append(_make_chunk(doc, page.page_number, idx, window))
            idx += 1

    return chunks, idx


def _split_by_headings(text: str) -> List[str]:
    """
    Split text at detected heading lines.
    Returns a list of sections (heading + body).
    If no headings are found, returns the original text as a single-item list.
    """
    lines = text.splitlines()
    sections: List[str] = []
    current: List[str] = []

    for line in lines:
        if _is_heading(line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current))

    return sections if len(sections) > 1 else [text]


def _is_heading(line: str) -> bool:
    """Return True if a line matches any heading pattern."""
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    return any(pattern.match(stripped) for pattern in _HEADING_PATTERNS)


def _sliding_window(text: str) -> List[str]:
    """
    Split text into overlapping token windows.
    chunk_size = CHUNK_SIZE_TOKENS, overlap = CHUNK_OVERLAP_TOKENS.
    """
    tokens = _TOKENIZER.encode(text)
    windows: List[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        window_tokens = tokens[start:end]
        windows.append(_TOKENIZER.decode(window_tokens))
        if end == len(tokens):
            break
        start += CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS

    return windows


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(
    doc: ParsedDocument,
    page_number: int,
    chunk_index: int,
    text: str,
    is_table: bool = False,
) -> Chunk:
    """Build a Chunk object from document metadata + text."""
    chunk_id = f"{doc.bid_id}__{doc.file_name}__{page_number}__{chunk_index}"
    doc_type = DOC_TYPE_TABLE if is_table else doc.doc_type

    return Chunk(
        chunk_id=chunk_id,
        bid_id=doc.bid_id,
        file_name=doc.file_name,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        doc_type=doc_type,
        addendum_number=doc.addendum_number,
        document_date=doc.document_date,
        source_format=doc.source_format,
        is_table=is_table,
    )


def _count_tokens(text: str) -> int:
    """Return the number of tokens in a text string."""
    return len(_TOKENIZER.encode(text))


def _truncate_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens."""
    tokens = _TOKENIZER.encode(text)
    return _TOKENIZER.decode(tokens[:max_tokens])
