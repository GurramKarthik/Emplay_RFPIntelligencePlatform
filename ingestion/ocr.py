"""
ingestion/ocr.py
----------------
BONUS: OCR fallback for scanned / image-only PDFs.

Used only when both pdfplumber and PyMuPDF return empty text,
indicating the PDF has no selectable text layer.

Dependencies (install separately if needed):
  pip install pytesseract pdf2image
  Also requires: Tesseract binary (https://github.com/UB-Mannheim/tesseract/wiki)
"""

import os
from typing import List

import structlog

from models import ParsedPage

log = structlog.get_logger(__name__)


def parse_with_ocr(file_path: str) -> List[ParsedPage]:
    """
    Convert each PDF page to an image and run Tesseract OCR on it.
    Returns a list of ParsedPage objects.

    Raises ImportError if pytesseract or pdf2image are not installed.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:
        raise ImportError(
            "OCR dependencies not installed. Run: pip install pytesseract pdf2image"
        ) from e

    log.info("ocr.starting", file=os.path.basename(file_path))

    # Convert PDF pages to PIL images (300 DPI for good OCR accuracy)
    images = convert_from_path(file_path, dpi=300)

    pages: List[ParsedPage] = []
    for i, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image, lang="eng")
        pages.append(ParsedPage(
            page_number=i,
            text=text or "",
            tables=[],  # OCR does not extract structured tables
        ))
        log.debug("ocr.page_done", page=i, chars=len(text))

    log.info("ocr.complete", file=os.path.basename(file_path), total_pages=len(pages))
    return pages
