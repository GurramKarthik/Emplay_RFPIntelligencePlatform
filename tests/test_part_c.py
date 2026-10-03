import os
import pytest
from agents.orchestrator import app
from agents.state import BidExtractionState
from search import bm25_index

def test_orchestrator_extraction():
    """
    Integration test for the LangGraph orchestrator.
    Tests extracting a single field to ensure the graph successfully runs
    Retrieval -> Extraction -> Recon -> Validation -> Output.
    """
    
    # Check if there's any bid indexed, otherwise skip
    bids = bm25_index.list_indexed_bids()
    if not bids:
        pytest.skip("No bids indexed. Please run indexing first.")
        
    test_bid = bids[0]
    
    # We will test extraction of just two fields to keep the test fast
    initial_state = {
        "bid_id": test_bid,
        "task_mode": "extraction",
        "field_plan": ["Due Date", "Bid Number"]
    }
    
    # Run the graph
    print(f"\n--- Running Extraction Graph for {test_bid} ---")
    final_state = app.invoke(initial_state)
    
    assert final_state is not None
    assert "final_output" in final_state
    
    record = final_state["final_output"]
    assert record.bid_id == test_bid
    
    extracted = record.extracted_fields
    assert "Due Date" in extracted
    assert "Bid Number" in extracted
    
    # Check the logs directory for the trace file
    trace_file = f"./logs/{test_bid}_trace.jsonl"
    assert os.path.exists(trace_file), f"Observability trace file {trace_file} was not created"
    
    print("\n--- Final Extracted Output ---")
    import json
    # Use standard dict conversion since record is a pydantic model
    print(json.dumps(record.model_dump(), indent=2))
    
    print("\n--- Trace Logs Written ---")
    with open(trace_file, "r") as f:
        lines = f.readlines()
        print(f"Total steps logged: {len(lines)}")

def test_orchestrator_qa():
    """
    Integration test for the QA mode of the orchestrator.
    """
    bids = bm25_index.list_indexed_bids()
    if not bids:
        pytest.skip("No bids indexed.")
        
    test_bid = bids[0]
    
    initial_state = {
        "bid_id": test_bid,
        "task_mode": "qa",
        "query": "What is the deadline for this bid?"
    }
    
    final_state = app.invoke(initial_state)
    
    record = final_state["final_output"]
    extracted = record.extracted_fields
    
    assert "answer" in extracted
    assert len(extracted["answer"]) > 0
    assert "sources" in extracted
    
    print("\n--- QA Final Output ---")
    print(extracted["answer"])
