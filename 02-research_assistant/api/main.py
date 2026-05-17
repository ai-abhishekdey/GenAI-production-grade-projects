"""
main.py
-------
FastAPI application entry point.

Shared state is created once on startup and reused across all requests:
  - embedder       : OpenAI embedding model
  - vector_store   : Qdrant connection (persistent, survives restarts)
  - documents      : in-memory chunk list for sparse/hybrid retrieval fallback
  - semaphore      : caps concurrent OpenAI calls to avoid rate limits
  - jobs           : tracks background ingestion job state

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import glob
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from prometheus_fastapi_instrumentator import Instrumentator

from src.observability.logger import setup_logging, get_logger
from src.indexing.embedding import get_openai_embedder
from src.indexing.vector_store import get_vector_store
from api.routers.ingest import router as ingest_router
from api.routers.query import router as query_router
from api.routers.evidence import router as evidence_router

logger = get_logger(__name__)

# max concurrent OpenAI API calls — tune this to your OpenAI tier's rate limit
MAX_CONCURRENT_LLM_CALLS = 20


# -------------------------------------------------------------
# lifespan: startup and shutdown hook. All shared objects are
# attached to app.state so any route can access them via request.
# setup_logging() is called first so all subsequent log calls
# from any module are captured.
# -------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log_file = setup_logging(log_type="app")
    logger.info("research-assistant api starting up", extra={"log_file": str(log_file) if log_file else "stdout only"})

    app.state.embedder = get_openai_embedder()
    app.state.documents = []
    app.state.jobs = {}
    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    # restore last uploaded PDF so evidence works immediately after a server restart
    uploads = sorted(glob.glob("data/uploads/*.pdf"), key=os.path.getmtime, reverse=True)
    app.state.current_pdf_path = uploads[0] if uploads else None
    if app.state.current_pdf_path:
        logger.info("restored pdf path", extra={"path": app.state.current_pdf_path})

    # connect to vector store — if Qdrant is unreachable at startup the server
    # still starts; routes return 503 until a successful ingest re-initialises it
    try:
        app.state.vector_store = get_vector_store(app.state.embedder)
    except Exception as e:
        logger.warning("could not connect to vector store at startup", extra={"error": str(e)})
        app.state.vector_store = None

    logger.info("startup complete")
    yield
    logger.info("research-assistant api shut down")


app = FastAPI(
    title="Research Assistant API",
    description="RAG pipeline over research papers — layout-aware chunking + dense retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query_router, prefix="/query", tags=["Query"])
app.include_router(evidence_router, prefix="/evidence", tags=["Evidence"])

# -------------------------------------------------------------
# Prometheus instrumentation: auto-tracks every HTTP route
# (request count, latency histogram, status codes) and exposes
# all metrics — including custom ones from metrics.py — at /metrics
# -------------------------------------------------------------
Instrumentator().instrument(app).expose(app)


# -------------------------------------------------------------
# log_requests: middleware that logs every HTTP request with
# method, path, status code, latency, and a short request_id
# for correlating log lines from the same request
# -------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "http request",
        extra={
            "request_id": request_id,
            "method":      request.method,
            "path":        request.url.path,
            "status_code": response.status_code,
            "latency_ms":  latency_ms,
        },
    )
    return response


# -------------------------------------------------------------
# landing: returns a brief JSON summary of the API so hitting
# the base URL gives something meaningful instead of a 404
# -------------------------------------------------------------
@app.get("/", tags=["Health"])
async def landing():
    return {
        "name": "Research Assistant API",
        "version": "1.0.0",
        "description": "RAG pipeline over research papers — layout-aware chunking + dense retrieval.",
        "docs": "/docs",
        "endpoints": {
            "POST /ingest/file": "Upload a PDF and index it",
            "POST /ingest/url": "Ingest from arXiv or direct PDF URL",
            "GET  /ingest/status/{job_id}": "Poll ingestion job progress",
            "POST /query": "Ask a question, get answer + sources",
            "GET  /health": "Liveness check",
            "GET  /status": "Vector store connection and index stats",
        },
    }


# -------------------------------------------------------------
# health: liveness check used by Streamlit and container probes
# -------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


# -------------------------------------------------------------
# status: reports how many chunks are loaded in memory and
# whether the vector store is connected and ready
# -------------------------------------------------------------
@app.get("/status", tags=["Health"])
async def status(request: Request):
    return {
        "vector_store_ready": request.app.state.vector_store is not None,
        "chunks_in_memory":   len(request.app.state.documents),
        "active_jobs":        len(request.app.state.jobs),
    }
