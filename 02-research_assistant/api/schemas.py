"""
schemas.py
----------
Pydantic models for all API request and response bodies.
"""

from pydantic import BaseModel


# -------------------------------------------------------------
# IngestURLRequest: body for POST /ingest/url — caller provides
# an arXiv or direct PDF URL to be fetched and indexed
# -------------------------------------------------------------
class IngestURLRequest(BaseModel):
    url: str


# -------------------------------------------------------------
# JobResponse: returned immediately after a POST /ingest call.
# The client polls GET /ingest/status/{job_id} for updates.
# -------------------------------------------------------------
class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str = ""


# -------------------------------------------------------------
# JobStatusResponse: returned by GET /ingest/status/{job_id}
# -------------------------------------------------------------
class JobStatusResponse(BaseModel):
    job_id: str
    status: str          # "pending" | "running" | "done" | "failed"
    message: str = ""
    chunks_indexed: int = 0


# -------------------------------------------------------------
# QueryRequest: body for POST /query
# -------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str


# -------------------------------------------------------------
# SourceChunk: a single retrieved chunk included in the response.
# Bbox fields are normalized 0-1 coordinates from the ADE layout
# parser — used by the evidence endpoint to highlight regions.
# -------------------------------------------------------------
class SourceChunk(BaseModel):
    content: str
    source: str = ""
    page: int | None = None
    box_top: float | None = None
    box_left: float | None = None
    box_right: float | None = None
    box_bottom: float | None = None


# -------------------------------------------------------------
# QueryResponse: returned by POST /query
# -------------------------------------------------------------
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: float


# -------------------------------------------------------------
# BoundingBox / EvidenceRequest: body for POST /evidence/page
# -------------------------------------------------------------
class BoundingBox(BaseModel):
    top: float
    left: float
    right: float
    bottom: float


class EvidenceRequest(BaseModel):
    boxes: list[BoundingBox]
