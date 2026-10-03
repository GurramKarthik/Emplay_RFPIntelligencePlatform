"""
constants.py
------------
Project-wide constants. Import from here — never hardcode strings in logic files.
"""

# ── Document Types ─────────────────────────────────────────────────────────────
DOC_TYPE_BID_PAGE  = "bid_page"
DOC_TYPE_RFP       = "rfp"
DOC_TYPE_ADDENDUM  = "addendum"
DOC_TYPE_SPECS     = "specs"
DOC_TYPE_AFFIDAVIT = "affidavit"
DOC_TYPE_TABLE     = "table"       # used for atomic table chunks

# ── Source Formats ─────────────────────────────────────────────────────────────
FORMAT_PDF  = "pdf"
FORMAT_HTML = "html"

# ── Parse Job Statuses ────────────────────────────────────────────────────────
STATUS_PENDING     = "PENDING"
STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_DONE        = "DONE"
STATUS_FAILED      = "FAILED"

# ── Addendum detection keywords ───────────────────────────────────────────────
ADDENDUM_KEYWORDS = ["addendum", "amendment", "corrigendum"]

# ── Affidavit detection keywords ──────────────────────────────────────────────
AFFIDAVIT_KEYWORDS = ["affidavit", "certification", "certificate"]

# ── Spec detection keywords ───────────────────────────────────────────────────
SPECS_KEYWORDS = ["specification", "technical spec", "product spec"]

# ── File extensions ───────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm"}

# ── Fields to extract (Part D) ────────────────────────────────────────────────
EXTRACTION_FIELDS = [
    "Bid Number",
    "Title",
    "Due Date",
    "Bid Submission Type",
    "Term of Bid",
    "Pre Bid Meeting",
    "Installation",
    "Bid Bond",
    "Delivery Date",
    "Payment Terms",
    "Additional Documentation",
    "MFG for Registration",
    "Contract or Cooperative",
    "Model_no",
    "Part_no",
    "Product",
    "contact_info",
    "company_name",
    "Bid Summary",
    "Product Specification",
]
