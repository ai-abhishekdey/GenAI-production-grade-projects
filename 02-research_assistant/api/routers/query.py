"""
query.py
--------
Endpoint for querying the RAG pipeline.

POST /query — accepts a question, returns an answer with source chunks.
"""

import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.generation.rag import run_rag, stream_rag
from src.observability.metrics import QUERY_COUNTER, QUERY_LATENCY, DOCS_IN_MEMORY
from api.schemas import QueryRequest, QueryResponse, SourceChunk

router = APIRouter()


# -------------------------------------------------------------
# query: runs the full RAG pipeline for a single question.
# A semaphore caps concurrent OpenAI calls to avoid rate limits.
# Returns 503 if no documents have been indexed yet.
# -------------------------------------------------------------
@router.post("", response_model=QueryResponse)
async def query(body: QueryRequest, request: Request):
    state = request.app.state

    if not state.vector_store:
        QUERY_COUNTER.labels(status="no_docs").inc()
        raise HTTPException(status_code=503, detail="No documents indexed. Call /ingest first.")

    start = time.perf_counter()

    try:
        async with state.semaphore:
            answer, retrieved_chunks = await run_rag(
                state.vector_store, state.documents, body.question
            )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        QUERY_COUNTER.labels(status="success").inc()
        QUERY_LATENCY.observe(latency_ms)
        DOCS_IN_MEMORY.set(len(state.documents))

    except Exception as e:
        QUERY_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e)) from e

    sources = [
        SourceChunk(
            content=chunk.page_content,
            source=chunk.metadata.get("source", ""),
            page=chunk.metadata.get("page"),
            box_top=chunk.metadata.get("box_top"),
            box_left=chunk.metadata.get("box_left"),
            box_right=chunk.metadata.get("box_right"),
            box_bottom=chunk.metadata.get("box_bottom"),
        )
        for chunk in retrieved_chunks
    ]

    return QueryResponse(answer=answer, sources=sources, latency_ms=latency_ms)


# -------------------------------------------------------------
# query_stream: same pipeline as /query but streams the LLM
# response as Server-Sent Events.
#
# Event sequence:
#   data: {"type": "sources", "sources": [...]}   — before any tokens
#   data: {"type": "token",   "content": "..."}   — one per LLM chunk
#   data: {"type": "done",    "latency_ms": ...}  — signals end of stream
# -------------------------------------------------------------
@router.post("/stream")
async def query_stream(body: QueryRequest, request: Request):
    state = request.app.state

    if not state.vector_store:
        QUERY_COUNTER.labels(status="no_docs").inc()
        raise HTTPException(status_code=503, detail="No documents indexed. Call /ingest first.")

    async def event_generator():
        start = time.perf_counter()
        try:
            async with state.semaphore:
                async for event_type, payload in stream_rag(
                    state.vector_store, state.documents, body.question
                ):
                    if event_type == "sources":
                        sources = [
                            SourceChunk(
                                content=chunk.page_content,
                                source=chunk.metadata.get("source", ""),
                                page=chunk.metadata.get("page"),
                                box_top=chunk.metadata.get("box_top"),
                                box_left=chunk.metadata.get("box_left"),
                                box_right=chunk.metadata.get("box_right"),
                                box_bottom=chunk.metadata.get("box_bottom"),
                            ).model_dump()
                            for chunk in payload
                        ]
                        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

                    elif event_type == "token":
                        yield f"data: {json.dumps({'type': 'token', 'content': payload})}\n\n"

                    elif event_type == "done":
                        latency_ms = round((time.perf_counter() - start) * 1000, 2)
                        QUERY_COUNTER.labels(status="success").inc()
                        QUERY_LATENCY.observe(latency_ms)
                        DOCS_IN_MEMORY.set(len(state.documents))
                        yield f"data: {json.dumps({'type': 'done', 'latency_ms': latency_ms})}\n\n"

        except Exception as e:
            QUERY_COUNTER.labels(status="error").inc()
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
