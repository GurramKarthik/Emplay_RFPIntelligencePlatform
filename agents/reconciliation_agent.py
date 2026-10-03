import json
import re
import structlog
from typing import Dict, Any, List
from .state import BidExtractionState, DraftField, AddendumChange
from .error_handler import with_error_handling, LLMTimeoutError
from config import LLM_PROVIDER, LLM_MODEL, GEMINI_API_KEY, OPENAI_API_KEY

logger = structlog.get_logger(__name__)

RECONCILIATION_SYSTEM_PROMPT = """\
You are an Addendum Reconciliation Agent for RFP documents.
Your job is to check if an extracted field value (often from a base RFP document) has been OVERRIDDEN or CHANGED by a later Addendum.

You will be provided with:
1. Field Name
2. Current Extracted Value
3. All Retrieved Evidence (including Base RFP and Addendums)

Analyze the evidence. Are there any Addendums that explicitly change the Current Extracted Value?
- If YES: provide the new value, the exact chunk ID of the addendum that caused the change, and a brief reason.
- If NO (or if the Current Extracted Value already reflects the addendum): has_change is false.

Respond ONLY with valid JSON:
{
  "has_change": true,
  "new_value": "The updated value",
  "addendum_chunk_id": "chunk_id_X",
  "reason": "Addendum 2 extended the due date from May 1 to May 15."
}
"""

def _call_gemini_recon(prompt: str) -> str:
    import time
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    last_error = None
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                ),
            )
            return response.text
        except Exception as exc:
            last_error = exc
            if "503" in str(exc) or "UNAVAILABLE" in str(exc) or "deadline" in str(exc).lower():
                time.sleep(2 ** attempt)
            else:
                raise LLMTimeoutError(f"Gemini error: {exc}")
    raise LLMTimeoutError(f"Gemini failed after retries: {last_error}")

def _call_openai_recon(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content
    except Exception as exc:
        raise LLMTimeoutError(f"OpenAI error: {exc}")

@with_error_handling("ReconciliationAgent")
def reconciliation_node(state: BidExtractionState, field: str) -> Dict[str, Any]:
    logger.info("ReconciliationAgent.start", field=field)
    
    draft = state.get("draft_fields", {}).get(field)
    evidence_list = state.get("retrieved_evidence", {}).get(field, [])
    
    if not draft or draft.value is None or not evidence_list:
        return {}
        
    # Check if there are actually any addendums in the evidence
    has_addendums = any(str(e.metadata.get("doc_type", "")).lower() == "addendum" for e in evidence_list)
    if not has_addendums:
        logger.info("ReconciliationAgent.skip", field=field, reason="No addendums found in evidence")
        return {}
        
    # Format evidence for the prompt
    chunks_text = "\n\n".join([
        f"--- Chunk ID: {e.metadata.get('chunk_id', idx)} ---\nDoc Type: {e.metadata.get('doc_type', 'unknown')}\nFile: {e.source_file}\n{e.content}" 
        for idx, e in enumerate(evidence_list)
    ])
    
    prompt = f"{RECONCILIATION_SYSTEM_PROMPT}\n\nField: {field}\nCurrent Extracted Value: {draft.value}\n\nEvidence:\n{chunks_text}"
    
    provider = LLM_PROVIDER.lower()
    raw_json = ""
    
    if provider == "gemini":
        raw_json = _call_gemini_recon(prompt)
    elif provider == "openai":
        raw_json = _call_openai_recon(prompt)
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}")
        
    try:
        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_json.strip(), flags=re.MULTILINE)
        data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from Reconciliation", raw=raw_json)
        raise e
        
    if data.get("has_change"):
        new_val = data.get("new_value")
        chunk_id = str(data.get("addendum_chunk_id"))
        reason = str(data.get("reason", "Updated by addendum"))
        
        # Find the addendum source
        addendum_source = next((e for e in evidence_list if str(e.metadata.get("chunk_id")) == chunk_id), evidence_list[0])
        
        change = AddendumChange(
            field=field,
            old_value=draft.value,
            new_value=new_val,
            addendum_source=addendum_source,
            reason=reason
        )
        
        # Update the draft field with the new value and new source
        updated_draft = DraftField(
            value=new_val,
            sources=[addendum_source], # Override source to the addendum
            confidence=0.9, # High confidence if explicit override
            notes=reason
        )
        
        logger.info("ReconciliationAgent.changed", field=field, old=draft.value, new=new_val)
        
        return {
            "draft_fields": {field: updated_draft},
            "addendum_changes": [change]
        }
        
    logger.info("ReconciliationAgent.no_change", field=field)
    return {}
