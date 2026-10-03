from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from config import LLM_PROVIDER, LLM_MODEL, GEMINI_API_KEY, OPENAI_API_KEY

log = structlog.get_logger(__name__)


@dataclass
class QueryUnderstanding:
    """
    Structured output from the Query Understander LLM pre-step.

    Fields:
        rewritten_query : Cleaner version of the user query for retrieval
        filters         : Metadata filters to narrow Qdrant / BM25 results
                          Possible keys: bid_id, doc_type, addendum_number
        mode            : Retrieval mode — 'hybrid' | 'semantic' | 'keyword'
        reasoning       : (optional) short explanation from the LLM
    """
    rewritten_query: str
    filters: Dict[str, Any] = field(default_factory=dict)
    mode: str = "hybrid"
    reasoning: str = ""


# Allowed values for validation
_VALID_MODES = {"hybrid", "semantic", "keyword"}
_VALID_DOC_TYPES = {"rfp", "addendum", "specs", "contract", "proposal", "other"}

_SYSTEM_PROMPT = """\
You are a query analysis assistant for an RFP (Request for Proposals) document intelligence system.

Given a user question about bid documents, extract:
1. rewritten_query: A cleaner, retrieval-optimised version of the query (expand abbreviations, remove filler words).
2. filters: Metadata filters to narrow results. Only include fields you are CONFIDENT about:
   - bid_id       : e.g. "Bid1", "Bid2" (only if explicitly mentioned)
   - doc_type     : one of [rfp, addendum, specs, contract, proposal] (only if clearly implied)
   - addendum_number : integer (only if a specific addendum number is mentioned)
3. mode: Best retrieval strategy:
   - "keyword"  : for exact lookups (bid numbers, dates, part numbers, names, deadlines)
   - "semantic" : for conceptual / open-ended questions
   - "hybrid"   : when both exact and conceptual matching helps (default if unsure)

Respond ONLY with valid JSON, no markdown, no explanation outside the JSON:
{
  "rewritten_query": "...",
  "filters": {},
  "mode": "hybrid",
  "reasoning": "one sentence"
}
"""


def understand_query(
    user_query: str,
    known_bid_ids: Optional[List[str]] = None,
) -> QueryUnderstanding:
    """
    Run the Query Understander LLM pre-step.

    Calls the configured LLM (Gemini or OpenAI) to rewrite the query,
    extract metadata filters, and select retrieval mode.

    Falls back gracefully to the raw query + hybrid mode if the LLM
    fails, returns bad JSON, or has no API key configured.

    Args:
        user_query     : Raw user question
        known_bid_ids  : Optional list of bid IDs to help validate bid_id filter

    Returns:
        QueryUnderstanding dataclass with rewritten_query, filters, mode
    """
    provider = LLM_PROVIDER.lower()
    log.info("query_understander.start", query=user_query[:80], provider=provider)

    raw_json: Optional[str] = None

    try:
        if provider == "gemini" and GEMINI_API_KEY:
            raw_json = _call_gemini(user_query)
        elif provider == "openai" and OPENAI_API_KEY:
            raw_json = _call_openai(user_query)
        elif provider == "groq":
            raw_json = _call_groq(user_query)
        else:
            log.warning("query_understander.no_llm_configured", provider=provider)
    except Exception as exc:
        log.warning("query_understander.llm_error", error=str(exc))

    if raw_json:
        result = _parse_llm_output(raw_json, user_query, known_bid_ids)
    else:
        result = _fallback(user_query)

    log.info(
        "query_understander.done",
        rewritten=result.rewritten_query[:80],
        filters=result.filters,
        mode=result.mode,
    )
    return result


# -- LLM Backends -------------------------------------------------------------

def _call_groq(user_query: str) -> str:
    """Call Groq and return raw JSON string."""
    from openai import OpenAI
    import time
    from config import GROQ_API_KEY
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    
    last_error = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_query},
                ],
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
                raise
    if last_error:
        raise last_error
    return ""

def _call_gemini(user_query: str) -> str:
    """Call Gemini via the google-genai SDK and return raw JSON string."""
    import time
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"{_SYSTEM_PROMPT}\n\nUser query: {user_query}"

    last_error: Optional[Exception] = None
    for attempt in range(1, 4):   # up to 3 attempts
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1024,
                ),
            )
            return response.text
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                wait = 2 ** attempt   # 2s, 4s, 8s
                log.warning("query_understander.gemini_503_retry",
                             attempt=attempt, wait=wait)
                time.sleep(wait)
            else:
                raise   # non-retriable error, raise immediately
    if last_error is not None:
        raise last_error
    return ""


def _call_openai(user_query: str) -> str:
    """Call OpenAI and return raw JSON string."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_query},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return resp.choices[0].message.content


# -- Parsing & Validation -----------------------------------------------------

def _parse_llm_output(
    raw: str,
    original_query: str,
    known_bid_ids: Optional[List[str]],
) -> QueryUnderstanding:
    """Parse and validate the LLM JSON output. Falls back on any error."""
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data: Dict[str, Any] = json.loads(clean)

        rewritten = str(data.get("rewritten_query", "")).strip() or original_query
        mode = str(data.get("mode", "hybrid")).lower()
        if mode not in _VALID_MODES:
            mode = "hybrid"

        raw_filters: Dict[str, Any] = data.get("filters") or {}
        filters = _validate_filters(raw_filters, known_bid_ids)

        return QueryUnderstanding(
            rewritten_query=rewritten,
            filters=filters,
            mode=mode,
            reasoning=str(data.get("reasoning", "")),
        )

    except Exception as exc:
        log.warning("query_understander.parse_error", error=str(exc), raw=raw[:200])
        return _fallback(original_query)


def _validate_filters(
    raw: Dict[str, Any],
    known_bid_ids: Optional[List[str]],
) -> Dict[str, Any]:
    """Validate and clean extracted filters — drop any suspicious values."""
    clean: Dict[str, Any] = {}

    bid_id = raw.get("bid_id")
    if bid_id:
        bid_id = str(bid_id).strip()
        if known_bid_ids is None or bid_id in known_bid_ids:
            clean["bid_id"] = bid_id
        else:
            log.debug("query_understander.unknown_bid_id_dropped", bid_id=bid_id)

    # doc_type filtering is disabled because it routinely causes false negatives for specs and addendums
    doc_type = raw.get("doc_type")
    # if doc_type and str(doc_type).lower() in _VALID_DOC_TYPES:
    #     clean["doc_type"] = str(doc_type).lower()

    addendum_number = raw.get("addendum_number")
    if addendum_number is not None:
        try:
            clean["addendum_number"] = int(addendum_number)
        except (ValueError, TypeError):
            pass

    return clean


def _fallback(query: str) -> QueryUnderstanding:
    """Safe fallback: return the raw query, no filters, hybrid mode."""
    log.info("query_understander.fallback", query=query[:80])
    return QueryUnderstanding(
        rewritten_query=query,
        filters={},
        mode="hybrid",
        reasoning="fallback — LLM unavailable or returned invalid JSON",
    )

