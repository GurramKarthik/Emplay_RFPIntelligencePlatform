"""
ingestion/file_router.py
------------------------
Routes each file in a bid folder to the correct parser based on extension.

Supported:
  .html / .htm  → html_parser
  .pdf          → pdf_parser

Unsupported extensions are logged and skipped — pipeline never crashes.
"""

import os
from typing import Optional

import structlog

from models import ParsedDocument
from constants import SUPPORTED_EXTENSIONS

log = structlog.get_logger(__name__)


def route_and_parse(bid_id: str, file_path: str) -> ParsedDocument:
    """
    Detect file type and dispatch to the appropriate parser.

    Returns a ParsedDocument on success.
    Raises an exception on failure — the caller (pipeline.py) handles retries.
    """
    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)

    if ext not in SUPPORTED_EXTENSIONS:
        log.warning("file_router.unsupported_extension", file=file_name, ext=ext)
        raise ValueError(f"Unsupported file extension '{ext}' for file: {file_name}")

    if ext in {".html", ".htm"}:
        log.debug("file_router.routing_to_html_parser", file=file_name)
        from ingestion.html_parser import parse_html
        return parse_html(bid_id, file_path)

    if ext == ".pdf":
        log.debug("file_router.routing_to_pdf_parser", file=file_name)
        from ingestion.pdf_parser import parse_pdf
        return parse_pdf(bid_id, file_path)

    # Should never reach here given the extension check above
    raise ValueError(f"Unhandled file type: {file_name}")
