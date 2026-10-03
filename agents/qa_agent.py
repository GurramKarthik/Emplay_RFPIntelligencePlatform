import json
import re
import structlog
from typing import Dict, Any, List
from .state import BidExtractionState, BidRecord, Evidence
from .error_handler import with_error_handling, LLMTimeoutError
from config import LLM_PROVIDER, LLM_MODEL, GEMINI_API_KEY, OPENAI_API_KEY

logger = structlog.get_logger(__name__)

QA_SYSTEM_PROMPT = """\
You are an expert Q&A assistant for RFP (Request for Proposal) documents.
You will be provided with a user Question and a set of retrieved Evidence Chunks.

YOUR RULES:
1. Answer the question comprehensively but concisely based ONLY on the provided evidence.
2. If the answer cannot be found in the evidence, clearly state that.
3. You must list the EXACT chunk IDs that support your answer.

Respond ONLY with valid JSON (no markdown formatting):
{
  "answer": "Your detailed answer here.",
  "source_chunk_ids": ["chunk_id_1", "chunk_id_2"]
}
"""

def _call_gemini_qa(prompt: str) -> str:
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
            err_str = str(exc).upper()
            if "503" in err_str or "UNAVAILABLE" in err_str or "DEADLINE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                time.sleep(10 * attempt)
            else:
                raise LLMTimeoutError(f"Gemini error: {exc}")
    raise LLMTimeoutError(f"Gemini failed after retries: {last_error}")

def _call_openai_qa(prompt: str) -> str:
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

def _call_groq_qa(prompt: str) -> str:
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

@with_error_handling("QAAgent")
def qa_node(state: BidExtractionState, field: str = "qa_response") -> Dict[str, Any]:
    # The error_handler decorator forces field="global" when called by LangGraph,
    # so we must explicitly look up "qa_response".
    logger.info("QAAgent.start", query=state.get("query"))
    
    evidence_list = state.get("retrieved_evidence", {}).get("qa_response", [])
    
    if not evidence_list:
        return {
            "final_output": BidRecord(
                bid_id=state["bid_id"],
                extracted_fields={"answer": "No evidence found to answer this question.", "sources": []}
            )
        }
        
    # Format evidence for the prompt
    chunks_text = "\n\n".join([
        f"--- Chunk ID: {e.metadata.get('chunk_id', idx)} ---\nFile: {e.source_file}\n{e.content}" 
        for idx, e in enumerate(evidence_list)
    ])
    
    prompt = f"{QA_SYSTEM_PROMPT}\n\nQuestion: {state['query']}\n\nEvidence:\n{chunks_text}"
    
    provider = LLM_PROVIDER.lower()
    raw_json = ""
    
    if provider == "gemini":
        raw_json = _call_gemini_qa(prompt)
    elif provider == "openai":
        raw_json = _call_openai_qa(prompt)
    elif provider == "groq":
        raw_json = _call_groq_qa(prompt)
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}")
        
    try:
        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_json.strip(), flags=re.MULTILINE)
        data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from QA Agent", raw=raw_json)
        raise e
        
    answer = data.get("answer", "")
    used_chunk_ids = data.get("source_chunk_ids", [])
    
    # Map back to sources
    used_sources: List[Evidence] = []
    for e in evidence_list:
        if str(e.metadata.get('chunk_id')) in used_chunk_ids or str(evidence_list.index(e)) in used_chunk_ids:
            used_sources.append(e)
            
    # Serialize sources for the final output
    sources_dict = [
        {"file": s.source_file, "page": s.page_num, "chunk_id": s.metadata.get("chunk_id")}
        for s in used_sources
    ]
            
    record = BidRecord(
        bid_id=state["bid_id"],
        extracted_fields={
            "question": state["query"],
            "answer": answer,
            "sources": sources_dict
        }
    )
    
    logger.info("QAAgent.done", success=bool(answer))
    
    return {
        "final_output": record
    }
