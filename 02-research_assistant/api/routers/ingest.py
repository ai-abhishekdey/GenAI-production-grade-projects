"""
ingest.py
---------
Endpoints for uploading and indexing documents into Qdrant.

POST /ingest/file  — upload a PDF file
POST /ingest/url   — ingest from an arXiv or direct PDF URL
GET  /ingest/status/{job_id} — poll job progress

PDFs are saved to data/uploads/ so the evidence endpoint can
render highlighted pages after ingestion completes.
"""

import time
import uuid
import os
from urllib.parse import urlparse

import requests as http_requests
from fastapi import APIRouter, UploadFile, BackgroundTasks, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from config import ADE_API_KEY, ADE_OUTPUT_DIR
from src.ingestion.loader import load_pdf, normalize_arxiv_url
from src.ingestion.preprocessor import preprocess_documents
from src.ingestion.chunking import get_layout_aware_chunks
from src.indexing.vector_store import add_documents, clear_vector_store
from src.observability.metrics import (
    INGESTION_COUNTER, INGESTION_DURATION, INGESTION_CHUNKS, DOCS_IN_MEMORY,
)
from api.schemas import IngestURLRequest, JobResponse, JobStatusResponse

router = APIRouter()

UPLOADS_DIR = "data/uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)


# -------------------------------------------------------------
# run_ingestion_pipeline: clears the existing index, then
# preprocesses, chunks, and indexes documents. source_override
# replaces the temp file path in chunk metadata with a clean
# display source (filename or URL) before indexing.
# pdf_path is stored in app state for the evidence endpoint.
# -------------------------------------------------------------
async def run_ingestion_pipeline(app_state, job_id, documents, source_override=None, pdf_path=None):
    t0 = time.perf_counter()
    try:
        # vector store may be None if Qdrant was unreachable at startup — retry now
        if app_state.vector_store is None:
            from src.indexing.vector_store import get_vector_store
            app_state.vector_store = await run_in_threadpool(get_vector_store, app_state.embedder)

        app_state.jobs[job_id]["status"] = "running"
        app_state.jobs[job_id]["message"] = "Clearing previous index..."

        await run_in_threadpool(clear_vector_store, app_state.vector_store)
        app_state.documents = []

        app_state.jobs[job_id]["message"] = "Preprocessing documents..."

        processed = await run_in_threadpool(preprocess_documents, documents)

        app_state.jobs[job_id]["message"] = "Chunking with layout-aware strategy..."

        chunks = await run_in_threadpool(
            get_layout_aware_chunks, processed, ADE_API_KEY, ADE_OUTPUT_DIR
        )

        # replace temp file path with the real source so indexed metadata is meaningful
        if source_override:
            for chunk in chunks:
                chunk.metadata["source"] = source_override

        app_state.jobs[job_id]["message"] = "Indexing into Qdrant..."

        await run_in_threadpool(
            add_documents, app_state.vector_store, chunks, app_state.embedder
        )

        app_state.documents.extend(chunks)

        # store the PDF path so the evidence endpoint can render highlighted pages
        if pdf_path:
            app_state.current_pdf_path = pdf_path

        duration = time.perf_counter() - t0
        INGESTION_COUNTER.labels(status="success").inc()
        INGESTION_DURATION.observe(duration)
        INGESTION_CHUNKS.observe(len(chunks))
        DOCS_IN_MEMORY.set(len(app_state.documents))

        app_state.jobs[job_id] = {
            "status": "done",
            "message": f"Indexed {len(chunks)} chunks successfully.",
            "chunks_indexed": len(chunks),
        }

    except Exception as e:
        INGESTION_COUNTER.labels(status="failed").inc()
        app_state.jobs[job_id] = {
            "status": "failed",
            "message": str(e),
            "chunks_indexed": 0,
        }


# -------------------------------------------------------------
# ingest_file: saves the upload to data/uploads/ under its
# original filename so ADE can find it and the evidence endpoint
# can render highlighted pages after ingestion completes.
# -------------------------------------------------------------
@router.post("/file", response_model=JobResponse)
async def ingest_file(file: UploadFile, background_tasks: BackgroundTasks, request: Request):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()

    safe_name = os.path.basename(file.filename)
    pdf_path = os.path.join(UPLOADS_DIR, safe_name)
    with open(pdf_path, "wb") as f:
        f.write(contents)

    job_id = str(uuid.uuid4())
    request.app.state.jobs[job_id] = {"status": "pending", "message": "", "chunks_indexed": 0}

    async def run():
        documents = await run_in_threadpool(load_pdf, pdf_path)
        await run_ingestion_pipeline(
            request.app.state, job_id, documents,
            source_override=safe_name, pdf_path=pdf_path,
        )

    background_tasks.add_task(run)

    return JobResponse(job_id=job_id, status="pending", message="Ingestion started.")


# -------------------------------------------------------------
# ingest_url: downloads the PDF to data/uploads/ so the layout
# chunker can access it on disk and the evidence endpoint can
# render pages. load_pdf_from_url is not used here because it
# deletes its temp file before the chunker gets a chance to run.
# -------------------------------------------------------------
@router.post("/url", response_model=JobResponse)
async def ingest_url(body: IngestURLRequest, background_tasks: BackgroundTasks, request: Request):
    job_id = str(uuid.uuid4())
    request.app.state.jobs[job_id] = {"status": "pending", "message": "", "chunks_indexed": 0}

    async def run():
        url = normalize_arxiv_url(body.url)

        url_filename = os.path.basename(urlparse(url).path) or "document"
        if not url_filename.endswith(".pdf"):
            url_filename += ".pdf"

        pdf_path = os.path.join(UPLOADS_DIR, url_filename)

        def download():
            response = http_requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        await run_in_threadpool(download)

        documents = await run_in_threadpool(load_pdf, pdf_path)
        await run_ingestion_pipeline(
            request.app.state, job_id, documents,
            source_override=url, pdf_path=pdf_path,
        )

    background_tasks.add_task(run)

    return JobResponse(job_id=job_id, status="pending", message="Ingestion started.")


# -------------------------------------------------------------
# get_job_status: returns the current state of an ingestion job
# so the client can show a progress indicator
# -------------------------------------------------------------
@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JobStatusResponse(job_id=job_id, **job)
