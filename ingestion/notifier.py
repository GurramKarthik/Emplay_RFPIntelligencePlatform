"""
ingestion/notifier.py
---------------------
Builds the end-of-run ingestion summary.

After all files are processed, this module queries the tracker
and returns a structured IngestionSummary with:
  - processed : list of successfully parsed file names
  - failed    : list of FailedFile objects (file, attempts, reason)
"""

import structlog

from ingestion import tracker
from models import IngestionSummary, FailedFile

log = structlog.get_logger(__name__)


def build_summary(bid_id: str) -> IngestionSummary:
    """
    Query the SQLite tracker and construct the IngestionSummary for a bid run.
    Logs a warning for each failed file.
    """
    done_rows  = tracker.get_done_files(bid_id)
    failed_rows = tracker.get_failed_files(bid_id)

    processed = [row["file_name"] for row in done_rows]

    failed: list[FailedFile] = []
    for row in failed_rows:
        log.warning(
            "notifier.file_failed",
            bid_id=bid_id,
            file=row["file_name"],
            attempts=row["attempts"],
            reason=row["last_error"],
        )
        failed.append(FailedFile(
            file=row["file_name"],
            attempts=row["attempts"],
            reason=row["last_error"] or "Unknown error",
        ))

    summary = IngestionSummary(
        bid_id=bid_id,
        processed=processed,
        failed=failed,
    )

    # Log a concise top-level summary
    log.info(
        "notifier.run_complete",
        bid_id=bid_id,
        processed=len(processed),
        failed=len(failed),
    )

    return summary
