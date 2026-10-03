import operator
from typing import List, Dict, Optional, Literal, Any, Annotated
from pydantic import BaseModel, Field

def merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    # Returns a new dictionary merging a and b
    res = a.copy() if a else {}
    if b:
        res.update(b)
    return res

FIELD_DESCRIPTIONS = {
    "Bid Number" :"Official solicitation / RFP identifier (e.g. JA-207652)",
    "Title": "Full title of the bid or RFP",
    "Due Date": 'Final submission deadline, including time and time zone. Must reflect any addendum changes',
    "Bid Submission Type" :"Required bid bond or security, amount or percentage, or 'Notrequired'. ",
    "Term of Bid": "The duration or length of the contract agreement, including any initial term (e.g. years) and possible renewal extensions.",
    "Installation": "Requirements or services related to deployment, white-glove services, setup, etching, asset decaling, and software installation.",
    "MFG for Registration": "The Original Equipment Manufacturer (OEM) or brand required for factory-authorized repair, maintenance, and registration.",
    "company_name": "The name of the agency, school district, or organization that is soliciting the bid or issuing the RFP (e.g., Dallas Independent School District, State of Maryland).",
    "Product Specification": "Detailed technical specifications, components, or configurations for all products, tiers, models, devices, and monitors requested in the bid.",
    "contact_info": "Point of contact, buyer name, procurement officer, email, and phone number.",
    "Bid Bond Requirement": "Information on whether a bid bond, security, or guarantee is required, and its amount.",
    "Payment Terms": "The payment terms, invoicing requirements, or net days for payment.",
    "Contract or Cooperative to use": "Any required cooperative contract, master vehicle, or interlocal agreement (e.g. CTPA, DIR) that must be used for pricing.",
    "Model_no": "Specific model numbers requested.",
    "Part_no": "Specific part numbers or SKUs requested.",
    "Product": "Names of the computing devices, laptops, desktops, tablets, or hardware requested.",
    "Bid Submission Type": "How the bid must be submitted (e.g. electronic portal, iSupplier, eMMA, paper in sealed envelope).",
    "Pre Bid Meeting": "Date, time, and location/link for the pre-bid or pre-proposal conference.",
    "Delivery Date": "Required timeline, days, or starting month for delivery of products."
}


class Evidence(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]
    source_file: str
    page_num: Optional[int] = None

class DraftField(BaseModel):
    value: Any
    sources: List[Evidence]
    confidence: float
    notes: Optional[str] = None

class AddendumChange(BaseModel):
    field: str
    old_value: Any
    new_value: Any
    addendum_source: Evidence
    reason: str

class ValidationResult(BaseModel):
    is_valid: bool
    feedback: str

FieldStatus = Literal["PENDING", "DONE", "NOT_FOUND", "EXTRACTION_ERROR", "VALIDATION_FAILED", "RETRY"]

class AgentStep(BaseModel):
    timestamp: str
    agent: str
    field: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    status: str
    value: Optional[Any] = None
    sources: Optional[List[Dict[str, Any]]] = None

class BidRecord(BaseModel):
    bid_id: str
    extracted_fields: Dict[str, Any]

# We use standard TypedDict for LangGraph compatibility with reducers
from typing import TypedDict

class BidExtractionState(TypedDict):
    bid_id: str
    task_mode: Literal["extraction", "qa"]
    query: Optional[str]

    # Planning
    field_plan: List[str]

    # Evidence
    retrieved_evidence: Annotated[Dict[str, List[Evidence]], merge_dict]

    # Extraction
    draft_fields: Annotated[Dict[str, DraftField], merge_dict]

    # Reconciliation (lists typically use operator.add)
    addendum_changes: Annotated[List[AddendumChange], operator.add]

    # Validation
    validation_results: Annotated[Dict[str, ValidationResult], merge_dict]
    retry_counts: Annotated[Dict[str, int], merge_dict]
    field_status: Annotated[Dict[str, FieldStatus], merge_dict]

    # Final
    final_output: Optional[BidRecord]
    trace: Annotated[List[AgentStep], operator.add]
