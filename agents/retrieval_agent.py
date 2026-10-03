import structlog
from typing import Dict, Any, List
from .state import BidExtractionState, Evidence, FIELD_DESCRIPTIONS
from .error_handler import with_error_handling, EmptyRetrievalError
from search.search_api import RFPSearchTool

logger = structlog.get_logger(__name__)

search_tool = RFPSearchTool()

def _get_query_for_field(field: str) -> str:
    """Simple heuristic to get a good search query for a field."""
    description = FIELD_DESCRIPTIONS.get(field, field)
    return f"What is the {field} for this bid or document? Look for: {description}"

@with_error_handling("RetrievalAgent")
def retrieval_node(state: BidExtractionState, field: str) -> Dict[str, Any]:
    """
    Retrieves chunks from Part B search API for a specific field.
    Updates the retrieved_evidence dictionary in the state.
    """
    logger.info("RetrievalAgent.start", field=field, bid_id=state["bid_id"])
    
    if state["task_mode"] == "qa" and state.get("query"):
        query = state["query"]
    else:
        query = _get_query_for_field(field)
        
    results = search_tool.run(
        query=query,
        bid_id=state["bid_id"],
        top_k=25 
    )
    
    if not results:
        raise EmptyRetrievalError(f"No results found for {field}")
        
    evidence_list: List[Evidence] = []
    for r in results:
        # Get the best available score (rerank, rrf, or base)
        score = r.get("rerank_score") or r.get("rrf_score") or r.get("score") or 0.0
        
        # Format the citation string
        sources = r.get("sources", [])
        primary_source = sources[0] if sources else {}
        
        evidence = Evidence(
            content=r.get("text", ""),
            score=score,
            metadata={
                "chunk_id": r.get("chunk_id"),
                "doc_type": r.get("doc_type"),
                "addendum_number": r.get("addendum_number"),
            },
            source_file=primary_source.get("file") or r.get("file_name") or "Unknown",
            page_num=primary_source.get("page") or r.get("page_number")
        )
        evidence_list.append(evidence)
        
    logger.info("RetrievalAgent.done", field=field, evidence_count=len(evidence_list))
    
    return {
        "retrieved_evidence": {
            field: evidence_list
        }
    }
