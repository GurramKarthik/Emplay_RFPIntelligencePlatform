import json
import re
import structlog
from typing import Dict, Any, List
from .state import BidExtractionState, DraftField, Evidence, FIELD_DESCRIPTIONS
from .error_handler import with_error_handling, LLMTimeoutError
from config import LLM_PROVIDER, LLM_MODEL, GEMINI_API_KEY, OPENAI_API_KEY

logger = structlog.get_logger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert RFP (Request for Proposal) data extraction assistant.
You will be given a specific "Field" to extract, its description, and a set of retrieved "Evidence Chunks" from the bid documents.

YOUR RULES:
1. ONLY use the provided Evidence Chunks. DO NOT hallucinate or guess based on outside knowledge.
2. If the field's value cannot be determined from the chunks, set "value" to null and provide a note explaining why.
3. Be as precise and concise as possible for the value. Match the details exactly to the requirements.
4. Indicate your confidence (0.0 to 1.0).
5. List the EXACT chunk IDs you used to formulate your answer in "source_chunk_ids".

Respond ONLY with valid JSON in the following format (no markdown formatting, no code blocks):
{
  "value": "The extracted value, or null",
  "confidence": 0.95,
  "notes": "Short reasoning for how you found the value or why it is missing",
  "source_chunk_ids": ["chunk_id_1", "chunk_id_2"]
}
"""

def _call_gemini_json(prompt: str) -> str:
    import time
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    last_error = None
    for attempt in range(1, 4):
        try:
            # We enforce JSON response via response_mime_type
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
            err_str = str(exc).upper()
            if "503" in err_str or "UNAVAILABLE" in err_str or "DEADLINE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                time.sleep(10 * attempt)
            else:
                raise LLMTimeoutError(f"Gemini error: {exc}")
    raise LLMTimeoutError(f"Gemini failed after retries: {last_error}")

def _call_openai_json(prompt: str) -> str:
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

def _call_groq_json(prompt: str) -> str:
    from openai import OpenAI
    import time
    from config import GROQ_API_KEY
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    
    last_error = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return resp.choices[0].message.content
        except Exception as exc:
            last_error = exc
            err_str = str(exc).upper()
            if "429" in err_str or "RATE_LIMIT" in err_str:
                time.sleep(5 * attempt)
            else:
                raise LLMTimeoutError(f"Groq error: {exc}")
    raise LLMTimeoutError(f"Groq failed after retries: {last_error}")

@with_error_handling("ExtractionAgent")
def extraction_node(state: BidExtractionState, field: str) -> Dict[str, Any]:
    logger.info("ExtractionAgent.start", field=field)
    
    evidence_list = state.get("retrieved_evidence", {}).get(field, [])
    
    if not evidence_list:
        logger.warning("No evidence for extraction", field=field)
        return {
            "field_status": {field: "NOT_FOUND"},
            "draft_fields": {
                field: DraftField(value=None, sources=[], confidence=0.0, notes="No evidence retrieved")
            }
        }
        
    # Format evidence for the prompt
    chunks_text = "\n\n".join([
        f"--- Chunk ID: {e.metadata.get('chunk_id', idx)} ---\nFile: {e.source_file}\n{e.content}" 
        for idx, e in enumerate(evidence_list)
    ])
    
    description = FIELD_DESCRIPTIONS.get(field, "Extract this field accurately.")
    prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nField to Extract: {field}\nDescription: {description}\n\nEvidence Chunks:\n{chunks_text}"
    
    provider = LLM_PROVIDER.lower()
    raw_json = ""
    
    if provider == "gemini":
        raw_json = _call_gemini_json(prompt)
    elif provider == "openai":
        raw_json = _call_openai_json(prompt)
    elif provider == "groq":
        raw_json = _call_groq_json(prompt)
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}")
        
    # Parse JSON
    try:
        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_json.strip(), flags=re.MULTILINE)
        data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from LLM", raw=raw_json)
        raise e
        
    # Map chunk IDs back to actual Evidence objects
    used_chunk_ids = data.get("source_chunk_ids", [])
    used_sources: List[Evidence] = []
    
    for e in evidence_list:
        c_id = str(e.metadata.get('chunk_id'))
        if c_id in used_chunk_ids or str(evidence_list.index(e)) in used_chunk_ids:
            used_sources.append(e)
            
    # If LLM didn't specify properly, but gave an answer, we can attach all chunks or top 1
    if data.get("value") is not None and not used_sources:
        used_sources = [evidence_list[0]] # fallback to top chunk
        
    draft = DraftField(
        value=data.get("value"),
        sources=used_sources,
        confidence=float(data.get("confidence", 0.0)),
        notes=data.get("notes", "")
    )
    
    status = "DONE" if draft.value is not None else "NOT_FOUND"
    
    logger.info("ExtractionAgent.done", field=field, status=status, value=draft.value)
    
    return {
        "field_status": {field: status},
        "draft_fields": {field: draft}
    }
