import json
import re
import structlog
from typing import Dict, Any
from .state import BidExtractionState, ValidationResult
from .error_handler import with_error_handling, LLMTimeoutError
from config import LLM_PROVIDER, LLM_MODEL, GEMINI_API_KEY, OPENAI_API_KEY

logger = structlog.get_logger(__name__)

VALIDATOR_SYSTEM_PROMPT = """\
You are an expert RFP Data Validator.
Your job is to review a piece of extracted data against the raw source evidence to ensure there is NO hallucination.

You will be provided with:
1. Field Name
2. Extracted Value
3. Evidence (The raw text chunks the value was allegedly extracted from)

Rule:
- If the Extracted Value is completely supported by the Evidence, "is_valid" must be true.
- If the Extracted Value includes details NOT present in the Evidence, or if it contradicts the Evidence, "is_valid" must be false.
- If the Extracted Value is null/None, and the Evidence indeed does not contain the answer, "is_valid" is true.

Respond ONLY with valid JSON (no markdown formatting):
{
  "is_valid": true,
  "feedback": "Explain why it is valid or invalid."
}
"""

def _call_gemini_validator(prompt: str) -> str:
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

def _call_openai_validator(prompt: str) -> str:
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

@with_error_handling("ValidatorAgent")
def validator_node(state: BidExtractionState, field: str) -> Dict[str, Any]:
    logger.info("ValidatorAgent.start", field=field)
    
    status = state.get("field_status", {}).get(field)
    
    # If the extraction agent already marked it as an error or pending, no need to validate
    if status in ["EXTRACTION_ERROR", "PENDING", "RETRY"]:
        return {}
        
    draft = state.get("draft_fields", {}).get(field)
    if not draft:
        return {}
        
    # Format evidence for the prompt
    chunks_text = "\n\n".join([f"Source: {e.source_file}\n{e.content}" for e in draft.sources])
    
    prompt = f"{VALIDATOR_SYSTEM_PROMPT}\n\nField: {field}\nExtracted Value: {draft.value}\n\nEvidence:\n{chunks_text}"
    
    provider = LLM_PROVIDER.lower()
    raw_json = ""
    
    if provider == "gemini":
        raw_json = _call_gemini_validator(prompt)
    elif provider == "openai":
        raw_json = _call_openai_validator(prompt)
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}")
        
    try:
        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_json.strip(), flags=re.MULTILINE)
        data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from Validator", raw=raw_json)
        raise e
        
    is_valid = bool(data.get("is_valid", True))
    feedback = str(data.get("feedback", ""))
    
    validation_result = ValidationResult(is_valid=is_valid, feedback=feedback)
    
    updates = {
        "validation_results": {field: validation_result}
    }
    
    if not is_valid:
        updates["field_status"] = {field: "VALIDATION_FAILED"}
        logger.warning("ValidatorAgent.failed", field=field, feedback=feedback)
    else:
        logger.info("ValidatorAgent.passed", field=field)
        
    return updates
