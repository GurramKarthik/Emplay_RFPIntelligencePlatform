import structlog
from typing import Dict, Any, List, Literal
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, START, END

from .state import BidExtractionState, BidRecord, AgentStep
from .retrieval_agent import retrieval_node
from .extraction_agent import extraction_node
from .reconciliation_agent import reconciliation_node
from .validator_agent import validator_node
from .qa_agent import qa_node

logger = structlog.get_logger(__name__)

# Field Groups for Parallel Execution
FIELD_GROUPS = [
    ["Due Date", "Pre Bid Meeting", "Delivery Date", "Term of Bid"],
    ["Bid Bond", "Payment Terms", "Bid Submission Type", "Additional Documentation", "Contract/Cooperative", "MFG Registration"],
    ["Product", "Model_no", "Part_no", "Product Specification", "Installation"],
    ["Bid Number", "Title", "Contact Info", "Company Name", "Bid Summary"]
]

def init_plan_node(state: BidExtractionState) -> Dict[str, Any]:
    """Initializes the field plan and status for extraction, with smart caching."""
    import os
    import json
    from .state import DraftField, Evidence
    
    logger.info("Orchestrator.init_plan_node")
    if state["task_mode"] == "qa":
        return {}
        
    field_plan = state.get("field_plan", [])
    if not field_plan:
        field_plan = [field for group in FIELD_GROUPS for field in group]
        
    initial_status = {f: "PENDING" for f in field_plan}
    initial_retries = {f: 0 for f in field_plan}
    draft_fields = state.get("draft_fields", {})
    
    # Check for existing cached output to prevent re-running completed fields
    out_path = f"./output/{state['bid_id']}_output.json"
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                extracted = cached_data.get("extracted_fields", {})
                for field in field_plan:
                    if field in extracted and extracted[field] is not None:
                        cached = extracted[field]
                        
                        # Only cache successfully extracted fields. Re-run any field that was null.
                        if cached.get("value") is None:
                            continue 
                            
                        # Reconstruct the DraftField so it bypasses processing but gets saved again
                        sources = [
                            Evidence(
                                content="", score=1.0, 
                                metadata={"chunk_id": s.get("chunk_id")}, 
                                source_file=s.get("file", "Unknown"), 
                                page_num=s.get("page")
                            ) for s in cached.get("sources", [])
                        ]
                        draft_fields[field] = DraftField(
                            value=cached.get("value"),
                            confidence=cached.get("confidence", 1.0),
                            notes=cached.get("notes", "Loaded from cache"),
                            sources=sources
                        )
                        initial_status[field] = "DONE"
                        logger.info("Orchestrator.cache_hit", field=field)
        except Exception as e:
            logger.warning("Orchestrator.cache_read_error", error=str(e))
    
    return {
        "field_plan": field_plan,
        "field_status": initial_status,
        "retry_counts": initial_retries,
        "draft_fields": draft_fields
    }

def process_single_field(state: BidExtractionState, field: str) -> Dict[str, Any]:
    """Runs the full Retrieval -> Extraction -> Recon -> Validate pipeline for one field."""
    status = state.get("field_status", {}).get(field, "PENDING")
    
    if status not in ["PENDING", "RETRY"]:
        return {} # Skip if already done or failed hard
        
    logger.info("Orchestrator.processing_field", field=field, status=status)
    
    updates = {}
    
    def merge_updates(new_updates: Dict[str, Any]):
        for k, v in new_updates.items():
            if k not in updates:
                if isinstance(v, dict):
                    updates[k] = {}
                elif isinstance(v, list):
                    updates[k] = []
                else:
                    updates[k] = v
                    continue
                    
            if isinstance(v, dict):
                updates[k].update(v)
            elif isinstance(v, list):
                updates[k].extend(v)
            else:
                updates[k] = v
                
    # 1. Retrieval
    r_up = retrieval_node(state, field)
    merge_updates(r_up)
    # Temporarily apply retrieval updates so extraction sees them
    temp_state = dict(state)
    temp_state["retrieved_evidence"] = {**temp_state.get("retrieved_evidence", {}), **updates.get("retrieved_evidence", {})}
    
    # 2. Extraction
    e_up = extraction_node(temp_state, field)
    merge_updates(e_up)
    temp_state["draft_fields"] = {**temp_state.get("draft_fields", {}), **updates.get("draft_fields", {})}
    temp_state["field_status"] = {**temp_state.get("field_status", {}), **updates.get("field_status", {})}
    
    # 3. Addendum Reconciliation
    recon_up = reconciliation_node(temp_state, field)
    if recon_up:
        merge_updates(recon_up)
        temp_state["draft_fields"] = {**temp_state.get("draft_fields", {}), **updates.get("draft_fields", {})}
        
    # 4. Validation
    v_up = validator_node(temp_state, field)
    merge_updates(v_up)
    
    return updates

def run_extraction_cycle_node(state: BidExtractionState) -> Dict[str, Any]:
    """Runs all pending/retry fields in parallel (simulating the fan-out)."""
    logger.info("Orchestrator.run_extraction_cycle")
    fields_to_process = [
        f for f in state["field_plan"]
        if state.get("field_status", {}).get(f) in ["PENDING", "RETRY"]
    ]
    
    if not fields_to_process:
        return {}
        
    overall_updates = {}
    
    # Run fields in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_single_field, state, f): f for f in fields_to_process}
        
        for future in futures:
            try:
                result = future.result()
                # Merge dictionaries carefully
                for k, v in result.items():
                    if k not in overall_updates:
                        if isinstance(v, dict):
                            overall_updates[k] = {}
                        elif isinstance(v, list):
                            overall_updates[k] = []
                        else:
                            overall_updates[k] = v
                            continue
                            
                    if isinstance(v, dict):
                        overall_updates[k].update(v)
                    elif isinstance(v, list):
                        overall_updates[k].extend(v)
                    else:
                        overall_updates[k] = v
            except Exception as e:
                field = futures[future]
                logger.error("Orchestrator.field_crash", field=field, error=str(e))
                if "field_status" not in overall_updates:
                    overall_updates["field_status"] = {}
                overall_updates["field_status"][field] = "EXTRACTION_ERROR"
                
    return overall_updates

def check_extraction_done(state: BidExtractionState) -> Literal["retry", "done"]:
    """Conditional edge to check if we need to cycle back for retries."""
    statuses = state.get("field_status", {})
    retry_counts = state.get("retry_counts", {})
    
    needs_retry = False
    new_statuses = {}
    new_retries = {}
    
    for field, status in statuses.items():
        if status == "VALIDATION_FAILED":
            count = retry_counts.get(field, 0)
            if count < 3:
                needs_retry = True
                new_statuses[field] = "RETRY"
                new_retries[field] = count + 1
                logger.info("Orchestrator.trigger_retry", field=field, attempt=count + 1)
            else:
                new_statuses[field] = "EXTRACTION_ERROR"
                logger.warning("Orchestrator.max_retries_reached", field=field)
                
    # We must actually apply these state updates before routing!
    # LangGraph conditional edges shouldn't mutate state directly, 
    # but we can return the route and apply updates in a separate node if needed.
    # To be safe, we will just let `apply_retries_node` do the mutation.
    
    if needs_retry:
        return "retry"
    return "done"
    
def apply_retries_node(state: BidExtractionState) -> Dict[str, Any]:
    """Applies the retry logic to state if the conditional edge decided to loop."""
    statuses = state.get("field_status", {})
    retry_counts = state.get("retry_counts", {})
    
    new_statuses = {}
    new_retries = {}
    
    for field, status in statuses.items():
        if status == "VALIDATION_FAILED":
            count = retry_counts.get(field, 0)
            if count < 3:
                new_statuses[field] = "RETRY"
                new_retries[field] = count + 1
            else:
                new_statuses[field] = "EXTRACTION_ERROR"
                
    return {
        "field_status": new_statuses,
        "retry_counts": new_retries
    }

def finalize_extraction_node(state: BidExtractionState) -> Dict[str, Any]:
    """Compiles the final BidRecord from all draft fields and writes to file (Part D)."""
    import os
    import json
    logger.info("Orchestrator.finalize_extraction")
    
    final_dict = {}
    drafts = state.get("draft_fields", {})
    
    for field in state.get("field_plan", []):
        draft = drafts.get(field)
        if draft and draft.value is not None:
            # Attach sources to the output for transparency
            sources_dict = [
                {"file": s.source_file, "page": s.page_num, "chunk_id": s.metadata.get("chunk_id")}
                for s in draft.sources
            ]
            final_dict[field] = {
                "value": draft.value,
                "confidence": draft.confidence,
                "sources": sources_dict,
                "notes": draft.notes
            }
        else:
            status = state.get("field_status", {}).get(field)
            if status == "EXTRACTION_ERROR":
                notes = "Rate limit error"
            else:
                notes = "Not found in documents"
                
            final_dict[field] = {
                "value": None,
                "sources": [],
                "confidence": draft.confidence if draft else 0.0,
                "notes": notes
            }
            
    record = BidRecord(
        bid_id=state["bid_id"],
        extracted_fields=final_dict
    )
    
    # --- Part D: Output Writer ---
    os.makedirs("./output", exist_ok=True)
    out_path = f"./output/{state['bid_id']}_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record.model_dump(), f, indent=2)
    logger.info("Orchestrator.output_written", file=out_path)
    
    return {"final_output": record}

def router_node(state: BidExtractionState) -> Literal["qa", "extraction"]:
    """Routes based on task mode."""
    if state["task_mode"] == "qa":
        return "qa"
    return "extraction"

def qa_retrieval_node(state: BidExtractionState) -> Dict[str, Any]:
    """Runs the retrieval agent for QA mode."""
    logger.info("Orchestrator.qa_retrieval")
    # 'qa_response' is the placeholder field name used in qa_agent.py
    updates = retrieval_node(state, "qa_response")
    return updates

# --- Graph Assembly ---

def build_graph() -> StateGraph:
    workflow = StateGraph(BidExtractionState)
    
    # Add Nodes
    workflow.add_node("init_plan", init_plan_node)
    workflow.add_node("run_extraction_cycle", run_extraction_cycle_node)
    workflow.add_node("apply_retries", apply_retries_node)
    workflow.add_node("finalize_extraction", finalize_extraction_node)
    workflow.add_node("qa_retrieval", qa_retrieval_node)
    workflow.add_node("qa", qa_node)
    
    # Edges
    workflow.add_edge(START, "init_plan")
    
    workflow.add_conditional_edges(
        "init_plan",
        router_node,
        {
            "qa": "qa_retrieval",
            "extraction": "run_extraction_cycle"
        }
    )
    
    workflow.add_conditional_edges(
        "run_extraction_cycle",
        check_extraction_done,
        {
            "retry": "apply_retries",
            "done": "finalize_extraction"
        }
    )
    
    workflow.add_edge("apply_retries", "run_extraction_cycle")
    workflow.add_edge("finalize_extraction", END)
    workflow.add_edge("qa_retrieval", "qa")
    workflow.add_edge("qa", END)
    
    return workflow.compile()

# Singleton instance of the compiled graph
app = build_graph()
