"""
rag.py
------
End-to-end RAG pipeline:
GUARDRAILS (input) → RETRIEVAL → RERANKING → AUGMENTATION → GENERATION → GUARDRAILS (output)

All functions are async — call with await from async contexts or asyncio.run() from sync.
"""

import asyncio

from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from config import PROMPT_TYPE, USE_RERANKER, RERANKER_TYPE, LANGSMITH_PROJECT
from src.observability.logger import get_logger
from src.retrieval.retriever import retrieve
from src.retrieval.reranker import get_crossencoder_reranker, get_flashrank_reranker
from src.generation.prompts import get_basic_rag_prompt, get_strict_rag_prompt, get_debug_prompt
from src.generation.llm import get_openai_llm
from src.generation.guardrails import run_input_guardrails, run_output_guardrails

logger = get_logger(__name__)


# -------------------------------------------------------------
# get_prompt: selects the prompt template based on PROMPT_TYPE
# set in config.py
# -------------------------------------------------------------
def get_prompt():
    if PROMPT_TYPE == "basic":
        return get_basic_rag_prompt()
    elif PROMPT_TYPE == "strict":
        return get_strict_rag_prompt()
    elif PROMPT_TYPE == "debug":
        return get_debug_prompt()
    else:
        raise ValueError(f"Invalid PROMPT_TYPE: {PROMPT_TYPE}")


# -------------------------------------------------------------
# run_rag: runs the full pipeline for a single query.
# Guardrails and reranking run in a thread executor since they
# use sync libraries. Always returns (answer, sources).
# @traceable creates a parent span in LangSmith so the full
# pipeline (guardrails → retrieval → rerank → generation) is
# visible as a single named trace per query.
# -------------------------------------------------------------
async def _retrieve_and_rerank(vector_store, documents, query):
    """Shared retrieval + optional reranking logic used by both run_rag and stream_rag."""
    try:
        await asyncio.to_thread(run_input_guardrails, query)
    except ValueError as e:
        logger.warning("query blocked by input guardrail", extra={"reason": str(e)})
        return None, [], str(e)

    logger.info("retrieval started", extra={"query": query})
    retrieved_chunks, base_retriever = await retrieve(vector_store, documents, query)

    if USE_RERANKER:
        if RERANKER_TYPE == "flashrank":
            reranker = get_flashrank_reranker(base_retriever)
        elif RERANKER_TYPE == "crossencoder":
            reranker = get_crossencoder_reranker(base_retriever)
        else:
            raise ValueError(f"Invalid RERANKER_TYPE: {RERANKER_TYPE}")

        retrieved_chunks = await asyncio.to_thread(reranker.invoke, query)
        logger.info("reranking complete", extra={"chunk_count": len(retrieved_chunks)})

    logger.info(
        "retrieval complete",
        extra={
            "chunk_count": len(retrieved_chunks),
            "sources": [
                {"source": c.metadata.get("source", "N/A"), "page": c.metadata.get("page")}
                for c in retrieved_chunks
            ],
        },
    )

    if not retrieved_chunks:
        return None, [], None

    context = "\n\n".join([doc.page_content for doc in retrieved_chunks])
    return context, retrieved_chunks, None


@traceable(name="rag-pipeline", project_name=LANGSMITH_PROJECT)
async def run_rag(vector_store, documents, query):

    context, retrieved_chunks, guardrail_error = await _retrieve_and_rerank(
        vector_store, documents, query
    )

    if guardrail_error:
        return guardrail_error, []

    if not retrieved_chunks:
        return "No relevant information found.", []

    logger.info("generation started")
    chain = get_prompt() | get_openai_llm() | StrOutputParser()
    response = await chain.ainvoke(
        {"context": context, "question": query},
        config={"run_name": "answer-generation"},
    )
    logger.info("generation complete", extra={"answer_length": len(response)})

    try:
        await asyncio.to_thread(run_output_guardrails, response, context)
    except ValueError as e:
        logger.warning("response blocked by output guardrail", extra={"reason": str(e)})
        return str(e), []

    return response, retrieved_chunks


@traceable(name="rag-pipeline-stream", project_name=LANGSMITH_PROJECT)
async def stream_rag(vector_store, documents, query):
    """
    Async generator that yields (event_type, payload) tuples:
      - ("sources", list[Document])  — emitted once, before any tokens
      - ("token",   str)             — one per streamed LLM chunk
      - ("done",    None)            — signals end of stream
    On a guardrail block or empty retrieval, yields a single token with the
    fallback message so the caller always gets a complete stream.
    """
    context, retrieved_chunks, guardrail_error = await _retrieve_and_rerank(
        vector_store, documents, query
    )

    if guardrail_error:
        yield ("sources", [])
        yield ("token", guardrail_error)
        yield ("done", None)
        return

    if not retrieved_chunks:
        yield ("sources", [])
        yield ("token", "No relevant information found.")
        yield ("done", None)
        return

    yield ("sources", retrieved_chunks)

    logger.info("streaming generation started")
    chain = get_prompt() | get_openai_llm() | StrOutputParser()
    full_response = []

    async for chunk in chain.astream(
        {"context": context, "question": query},
        config={"run_name": "answer-generation-stream"},
    ):
        full_response.append(chunk)
        yield ("token", chunk)

    response = "".join(full_response)
    logger.info("streaming generation complete", extra={"answer_length": len(response)})

    try:
        await asyncio.to_thread(run_output_guardrails, response, context)
    except ValueError as e:
        logger.warning("response blocked by output guardrail", extra={"reason": str(e)})
        yield ("token", f"\n\n[Response blocked: {e}]")

    yield ("done", None)
