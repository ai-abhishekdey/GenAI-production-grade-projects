"""
evidence.py
-----------
Renders a PDF page with bounding boxes highlighted around the
retrieved chunks, so the Streamlit UI can show visual evidence.

POST /evidence/page/{page_num}
  Body: { "boxes": [{ "top": 0.1, "left": 0.2, "right": 0.8, "bottom": 0.4 }, ...] }
  Returns: PNG image of the page with highlighted regions
"""

import fitz  # pymupdf

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from src.observability.metrics import EVIDENCE_COUNTER
from api.schemas import EvidenceRequest

router = APIRouter()


# -------------------------------------------------------------
# render_page: opens the PDF, draws highlight rectangles over
# the given boxes, and returns the page as a PNG byte string.
# ADE bounding boxes are normalized (0-1 relative to page size)
# so we multiply by page width/height to get point coordinates.
# -------------------------------------------------------------
def render_page(pdf_path, page_num, boxes):
    doc = fitz.open(pdf_path)

    if page_num < 1 or page_num > len(doc):
        doc.close()
        raise ValueError(f"Page {page_num} out of range (document has {len(doc)} pages).")

    page = doc[page_num - 1]
    width = page.rect.width
    height = page.rect.height

    for box in boxes:
        rect = fitz.Rect(
            box.left  * width,
            box.top   * height,
            box.right * width,
            box.bottom * height,
        )
        page.draw_rect(rect, color=(1, 0, 0), width=2)

    # render at 2x zoom so text is readable in the UI
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


# -------------------------------------------------------------
# get_page_evidence: endpoint called by the Streamlit UI after
# each query to show which part of the paper the answer came from
# -------------------------------------------------------------
@router.post("/page/{page_num}", response_class=Response)
async def get_page_evidence(page_num: int, body: EvidenceRequest, request: Request):
    pdf_path = request.app.state.current_pdf_path

    if not pdf_path:
        EVIDENCE_COUNTER.labels(status="not_loaded").inc()
        raise HTTPException(status_code=404, detail="No document is currently loaded.")

    if not body.boxes:
        raise HTTPException(status_code=400, detail="No bounding boxes provided.")

    try:
        img_bytes = await run_in_threadpool(render_page, pdf_path, page_num, body.boxes)
    except ValueError as e:
        EVIDENCE_COUNTER.labels(status="render_error").inc()
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        EVIDENCE_COUNTER.labels(status="render_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to render page: {e}") from e

    EVIDENCE_COUNTER.labels(status="success").inc()
    return Response(content=img_bytes, media_type="image/png")
