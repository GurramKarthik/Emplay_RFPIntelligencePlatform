import os
import json
import time
from datetime import datetime
import structlog
from typing import Callable, Any, Dict
from pydantic import ValidationError
from .state import BidExtractionState, DraftField, AgentStep

logger = structlog.get_logger()

class LLMTimeoutError(Exception):
    pass

class EmptyRetrievalError(Exception):
    pass

def _write_trace(bid_id: str, step: AgentStep):
    """Writes a clean, human-readable observability trace to a log file."""
    os.makedirs("./logs", exist_ok=True)
    file_path = f"./logs/{bid_id}_agent_trace.log"
    
    # Format the timestamp for readability (e.g., 14:30:05)
    try:
        timestamp = step.timestamp.split("T")[-1][:8]
    except:
        timestamp = step.timestamp
        
    log_entry = f"[{timestamp}] [{step.agent}] Field: '{step.field}' | Status: {step.status} | Latency: {step.latency_ms}ms"
    
    # Add value if present (truncate if it's massive)
    if step.value is not None:
        val_str = str(step.value).replace("\n", " ")
        if len(val_str) > 150:
            val_str = val_str[:147] + "..."
        log_entry += f"\n  ↳ Extracted/Output: {val_str}"
        
    # Add source count if present
    if step.sources:
        log_entry += f"\n  ↳ Sources Cited: {len(step.sources)} chunk(s)"
        
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n\n")

def with_error_handling(agent_name: str):
    """
    Decorator to wrap agent execution logic.
    Catches errors and automatically writes an observability trace step.
    """
    def decorator(func: Callable):
        def wrapper(state: BidExtractionState, field: str = "global", *args, **kwargs) -> Dict[str, Any]:
            t0 = time.perf_counter()
            updates = {}
            status = "DONE"
            value = None
            sources = []
            
            try:
                updates = func(state, field, *args, **kwargs)
                if updates is None:
                    updates = {}
                    
                # Extract value and sources from the draft field if present
                if "draft_fields" in updates and field in updates["draft_fields"]:
                    draft = updates["draft_fields"][field]
                    value = draft.value
                    sources = [
                        {"file": s.source_file, "page": s.page_num, "chunk_id": s.metadata.get("chunk_id")}
                        for s in draft.sources
                    ]
                
                # If validation failed, extract that
                if updates.get("field_status", {}).get(field) == "VALIDATION_FAILED":
                    status = "VALIDATION_FAILED"
                    value = updates.get("validation_results", {}).get(field).feedback if updates.get("validation_results") else None

            except LLMTimeoutError:
                logger.error("LLM timeout", agent=agent_name, field=field)
                status = "EXTRACTION_ERROR"
                updates = {"field_status": {field: status}}

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error("Output parse error", agent=agent_name, field=field, error=str(e))
                status = "EXTRACTION_ERROR"
                updates = {"field_status": {field: status}}

            except EmptyRetrievalError:
                logger.warning("Empty retrieval", agent=agent_name, field=field)
                status = "NOT_FOUND"
                updates = {
                    "field_status": {field: status},
                    "draft_fields": {
                        field: DraftField(
                            value=None, 
                            sources=[], 
                            confidence=0.0, 
                            notes="Not found in documents"
                        )
                    }
                }
            except Exception as e:
                logger.error("Unexpected error", agent=agent_name, field=field, error=str(e))
                status = "EXTRACTION_ERROR"
                updates = {"field_status": {field: status}}
                
            finally:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                
                # Record the observability trace
                step = AgentStep(
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    agent=agent_name,
                    field=field,
                    latency_ms=latency_ms,
                    status=status,
                    value=value,
                    sources=sources
                )
                
                _write_trace(state["bid_id"], step)
                
                # LangGraph expects updates to be merged
                if "trace" not in updates:
                    updates["trace"] = []
                updates["trace"].append(step)
                
            return updates
        return wrapper
    return decorator
