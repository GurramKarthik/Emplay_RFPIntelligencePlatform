"""
ingestion/pipeline.py
---------------------
Top-level Part A orchestrator.

For a given bid folder:
  1. Registers all files in the SQLite tracker
  2. Parses each file (with retry logic up to MAX_RETRIES)
  3. Chunks each parsed document
  4. Writes chunks to ./data/chunks/{bid_id}_chunks.jsonl
  5. Returns an IngestionSummary (processed + failed files)

This is the only function that external code (main.py, API, Streamlit) calls
for Part A.
"""

import json
import os
from pathlib import Path
from typing import List

import structlog

from config import MAX_RETRIES, CHUNKS_OUTPUT_DIR
from constants import SUPPORTED_EXTENSIONS
from ingestion import tracker, notifier
from ingestion.file_router import route_and_parse
from ingestion.chunker import chunk_document
from models import IngestionSummary, Chunk

log = structlog.get_logger(__name__)


def run_ingestion_pipeline(bid_id: str, bid_folder: str) -> IngestionSummary:
    """
    Run the full Part A pipeline for a single bid folder.

    Args:
        bid_id     : Unique identifier for this bid (e.g. "Bid1")
        bid_folder : Path to the folder containing the bid's HTML and PDF files

    Returns:
        IngestionSummary with lists of processed and failed files.
    """
    log.info("pipeline.start", bid_id=bid_id, folder=bid_folder)

    # ── Initialise tracker DB ─────────────────────────────────────────────────
    tracker.init_db()

    # ── Discover files ────────────────────────────────────────────────────────
    files = _discover_files(bid_folder)
    if not files:
        log.error("pipeline.no_files_found", bid_id=bid_id, folder=bid_folder)
        return IngestionSummary(bid_id=bid_id, processed=[], failed=[])

    log.info("pipeline.files_discovered", bid_id=bid_id, count=len(files))

    # ── Register all files in tracker ─────────────────────────────────────────
    for file_path in files:
        file_name = os.path.basename(file_path)
        tracker.register_file(bid_id, file_name, file_path)

    # ── Parse each file (with retry) ─────────────────────────────────────────
    all_chunks: List[Chunk] = []

    for file_path in files:
        file_name = os.path.basename(file_path)
        chunks = _parse_with_retry(bid_id, file_name, file_path)
        all_chunks.extend(chunks)

    # ── Write chunks to local JSONL (checkpoint before Part B) ────────────────
    if all_chunks:
        _write_chunks_jsonl(bid_id, all_chunks)

    # ── Build and return run summary ──────────────────────────────────────────
    summary = notifier.build_summary(bid_id)
    _write_summary_json(bid_id, summary, total_chunks=len(all_chunks))
    log.info(
        "pipeline.complete",
        bid_id=bid_id,
        total_chunks=len(all_chunks),
        processed=len(summary.processed),
        failed=len(summary.failed),
    )
    return summary


# ── Private Helpers ───────────────────────────────────────────────────────────

def _discover_files(bid_folder: str) -> List[str]:
    """
    Return a sorted list of supported file paths found in the bid folder.
    Only processes files at the top level of the folder (not recursive).
    """
    folder = Path(bid_folder)
    if not folder.exists() or not folder.is_dir():
        log.error("pipeline.folder_not_found", folder=bid_folder)
        return []

    files = [
        str(f) for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def _parse_with_retry(bid_id: str, file_name: str, file_path: str) -> List[Chunk]:
    """
    Attempt to parse a file up to MAX_RETRIES times.

    On success: marks file DONE in tracker, returns chunk list.
    On exhausted retries: marks file FAILED, returns empty list.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        log.info(
            "pipeline.parse_attempt",
            bid_id=bid_id,
            file=file_name,
            attempt=attempt,
            max=MAX_RETRIES,
        )
        tracker.mark_in_progress(bid_id, file_name)

        try:
            # Parse file → ParsedDocument
            parsed_doc = route_and_parse(bid_id, file_path)

            # Chunk document → List[Chunk]
            chunks = chunk_document(parsed_doc)

            tracker.mark_done(bid_id, file_name)
            log.info(
                "pipeline.parse_success",
                bid_id=bid_id,
                file=file_name,
                chunks=len(chunks),
            )
            return chunks

        except Exception as e:
            new_attempt_count = tracker.record_failure(bid_id, file_name, str(e))
            log.warning(
                "pipeline.parse_failed",
                bid_id=bid_id,
                file=file_name,
                attempt=attempt,
                error=str(e),
            )

            if new_attempt_count >= MAX_RETRIES:
                tracker.mark_failed(bid_id, file_name)
                log.error(
                    "pipeline.file_permanently_failed",
                    bid_id=bid_id,
                    file=file_name,
                    attempts=new_attempt_count,
                )
                return []

    return []  # Should not reach here


def _write_chunks_jsonl(bid_id: str, chunks: List[Chunk]) -> None:
    """
    Write all chunks to ./data/chunks/{bid_id}_chunks.jsonl.
    One JSON object per line (newline-delimited JSON).
    This acts as a checkpoint — Part B reads from here.
    """
    os.makedirs(CHUNKS_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(CHUNKS_OUTPUT_DIR, f"{bid_id}_chunks.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

    log.info(
        "pipeline.chunks_written",
        bid_id=bid_id,
        path=output_path,
        total=len(chunks),
    )


def _write_summary_json(bid_id: str, summary: "IngestionSummary", total_chunks: int) -> None:
    """
    Persist the ingestion summary to ./data/chunks/{bid_id}_summary.json.
    Includes processed files, failed files, and total chunk count.
    """
    from models import IngestionSummary  # avoid circular import at module level

    os.makedirs(CHUNKS_OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(CHUNKS_OUTPUT_DIR, f"{bid_id}_summary.json")

    payload = {
        "bid_id": bid_id,
        "total_chunks": total_chunks,
        "processed_count": len(summary.processed),
        "failed_count": len(summary.failed),
        "processed": summary.processed,
        "failed": [f.model_dump() for f in summary.failed],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log.info("pipeline.summary_written", bid_id=bid_id, path=summary_path)
