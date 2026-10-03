# -*- coding: utf-8 -*-
"""
main.py
-------
CLI entry point for the RFP Intelligence Platform.

Usage:
  python main.py --bid ./Bid1 --bid-id Bid1
  python main.py --bid ./Bid2 --bid-id Bid2

Currently runs Part A (ingestion + parsing + chunking).
Parts B, C, D will be added incrementally.
"""

import argparse
import json
import sys
import os

import structlog
import logging

from config import LOG_LEVEL, LOG_DIR


def _setup_logging() -> None:
    """Configure structlog for human-readable console + JSON file output."""
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def run_part_a(bid_id: str, bid_folder: str) -> None:
    """Run Part A — ingestion, parsing, chunking."""
    from ingestion.pipeline import run_ingestion_pipeline

    log = structlog.get_logger("main")
    log.info("=== Part A: Ingestion & Parsing ===", bid_id=bid_id, folder=bid_folder)

    summary = run_ingestion_pipeline(bid_id=bid_id, bid_folder=bid_folder)

    # Pretty-print the summary (ASCII-safe for Windows terminal)
    print("\n" + "=" * 60)
    print(f"  Ingestion Summary -- {bid_id}")
    print("=" * 60)
    print(f"  [OK]  Processed : {len(summary.processed)} file(s)")
    for f in summary.processed:
        print(f"        - {f}")

    if summary.failed:
        print(f"\n  [FAIL] Failed : {len(summary.failed)} file(s)")
        for ff in summary.failed:
            print(f"        - {ff.file}  (attempts: {ff.attempts})")
            print(f"          Reason: {ff.reason}")
    else:
        print("\n  [OK] No failures -- all files parsed successfully!")

    print("=" * 60 + "\n")

def run_part_b(bid_id: str) -> None:
    """Run Part B — embed chunks and index into Qdrant + BM25."""
    from search.indexer import run_indexing_pipeline

    log = structlog.get_logger("main")
    log.info("=== Part B: Indexing ===", bid_id=bid_id)

    result = run_indexing_pipeline(bid_id=bid_id)

    print("\n" + "=" * 60)
    print(f"  Indexing Summary -- {bid_id}")
    print("=" * 60)
    if result["status"] == "success":
        print(f"  [OK] Chunks indexed : {result['chunks_indexed']}")
        print(f"       Qdrant         : {result['qdrant_collection']}")
        print(f"       BM25 index     : {result['bm25_index_path']}")
    else:
        print(f"  [FAIL] {result.get('reason')}")
    print("=" * 60 + "\n")



def run_part_c(bid_id: str, fields: list = None, mode: str = "extraction", query: str = None) -> None:
    """Run Part C & D — Agents, Orchestration, and Output generation."""
    from agents.orchestrator import app
    from agents.state import BidExtractionState
    import json
    
    log = structlog.get_logger("main")
    log.info("=== Part C & D: AI Pipeline ===", bid_id=bid_id, mode=mode)
    
    if mode == "extraction":
        if fields is None:
            fields = [
                "Bid Number", "Title", "Due Date", "Bid Submission Type",
                "Term of Bid", "Pre Bid Meeting", "Installation",
                "Bid Bond Requirement", "Delivery Date", "Payment Terms",
                "Any Additional Documentation Required", "MFG for Registration",
                "Contract or Cooperative to use", "Model_no", "Part_no",
                "Product", "contact_info", "company_name",
                "Bid Summary", "Product Specification"
            ]
        initial_state = {
            "bid_id": bid_id,
            "task_mode": "extraction",
            "field_plan": fields
        }
    else:
        # QA / Query Mode
        initial_state = {
            "bid_id": bid_id,
            "task_mode": "qa",
            "query": query or "What is this bid about?"
        }
    
    final_state = app.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print(f"  Extraction Summary -- {bid_id}")
    print("=" * 60)
    
    if final_state and "final_output" in final_state:
        out = final_state["final_output"]
        if mode == "extraction":
            print(f"  [OK] Extraction completed. Extracted {len(out.extracted_fields)} fields.")
            print(f"  [OK] Saved to: ./output/{bid_id}_output.json")
        else:
            print(f"  [QA ANSWER]: {out.extracted_fields.get('answer')}")
    else:
        print("  [FAIL] Orchestrator did not return final output.")
        
    print("=" * 60 + "\n")
def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="RFP Intelligence Platform -- CLI"
    )
    parser.add_argument(
        "--bid",
        required=False,
        default=None,
        help="Path to the bid folder (e.g. ./Bid1) — required for Part A",
    )
    parser.add_argument(
        "--bid-id",
        required=False,
        help="Unique bid identifier (e.g. Bid1). Use 'all' or omit for global QA queries.",
        default="all"
    )
    parser.add_argument(
        "--part",
        choices=["a", "b", "c", "all"],
        default="a",
        help="Which part to run (default: a)",
    )
    parser.add_argument(
        "--mode",
        choices=["extraction", "qa"],
        default="extraction",
        help="Task mode for Part C (default: extraction)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="The question to ask when in QA mode",
        default=None,
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Specific fields to extract in Part C (e.g. --fields 'Due Date' 'Bid Number')",
        default=None,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear stale tracker entries for this bid before running",
    )

    args = parser.parse_args()

    if args.part in ("a", "all"):
        if args.reset:
            from ingestion.tracker import init_db, reset_bid
            init_db()
            deleted = reset_bid(args.bid_id)
            print(f"[reset] Cleared {deleted} stale tracker entries for {args.bid_id}")
        run_part_a(bid_id=args.bid_id, bid_folder=args.bid)

    if args.part in ("b", "all"):
        run_part_b(bid_id=args.bid_id)
        
    if args.part in ("c", "all"):
        run_part_c(bid_id=args.bid_id, fields=args.fields, mode=args.mode, query=args.query)


if __name__ == "__main__":
    main()
