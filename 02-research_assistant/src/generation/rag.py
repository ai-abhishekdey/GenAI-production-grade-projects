"""
rag.py
------
End-to-end RAG pipeline:
GUARDRAILS (input) → RETRIEVAL → RERANKING → AUGMENTATION → GENERATION → GUARDRAILS (output)
"""

from langchain_core.output_parsers import StrOutputParser

from config import PROMPT_TYPE, USE_RERANKER, RERANKER_TYPE
from src.retrieval.retriever import retrieve
from src.retrieval.reranker import get_crossencoder_reranker, get_flashrank_reranker
from src.generation.prompts import get_basic_rag_prompt, get_strict_rag_prompt, get_debug_prompt
from src.generation.llm import get_openai_llm
from src.generation.guardrails import run_input_guardrails, run_output_guardrails


# ── Prompt Selector ────────────────────────────────────

def get_prompt():
    if PROMPT_TYPE == "basic":
        return get_basic_rag_prompt()
    elif PROMPT_TYPE == "strict":
        return get_strict_rag_prompt()
    elif PROMPT_TYPE == "debug":
        return get_debug_prompt()
    else:
        raise ValueError(f"Invalid PROMPT_TYPE: {PROMPT_TYPE}")


# ── Main RAG Function ──────────────────────────────────

def run_rag(vector_store, documents, query: str):
    """
    Runs full RAG pipeline:
    1. Input guardrails
    2. Retrieval
    3. Reranking (optional)
    4. Augmentation
    5. Generation
    6. Output guardrails
    """

    # ── Step 1: Input Guardrails ───────────────────────
    try:
        run_input_guardrails(query)
    except ValueError as e:
        print(f"\n[rag] Blocked by input guardrail: {e}")
        return str(e)

    print("\n===================================================")
    print("              RETRIEVAL STAGE                      ")
    print("===================================================")

    # ── Step 2: Retrieval ──────────────────────────────
    retrieved_chunks, base_retriever = retrieve(vector_store, documents, query)

    # ── Step 3: Reranking (optional) ───────────────────
    if USE_RERANKER:
        print("\n===================================================")
        print("              RERANKING STAGE                      ")
        print("===================================================")

        if RERANKER_TYPE == "flashrank":
            reranker = get_flashrank_reranker(base_retriever)
        elif RERANKER_TYPE == "crossencoder":
            reranker = get_crossencoder_reranker(base_retriever)
        else:
            raise ValueError(f"Invalid RERANKER_TYPE: {RERANKER_TYPE}")

        retrieved_chunks = reranker.invoke(query)
        print(f"[rag] {len(retrieved_chunks)} chunks after reranking")

    for i, r in enumerate(retrieved_chunks):
        print(f"\n[Chunk {i+1}]")
        print("Source :", r.metadata.get("source", "N/A"))
        print("Page   :", r.metadata.get("page", "N/A"))
        print("Text   :", r.page_content[:300])

    if not retrieved_chunks:
        return "No relevant information found."

    print("\n===================================================")
    print("              AUGMENTATION STAGE                   ")
    print("===================================================")

    # ── Step 4: Augmentation ───────────────────────────
    context = "\n\n".join([doc.page_content for doc in retrieved_chunks])
    print("\n--- Context Preview ---")
    print(context[:500])

    prompt = get_prompt()

    print("\n===================================================")
    print("               GENERATION STAGE                    ")
    print("===================================================")

    # ── Step 5: Generation ─────────────────────────────
    llm = get_openai_llm()
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    response = chain.invoke({
        "context": context,
        "question": query
    })

    print("\n---------------------------------------------")
    print("[Question]")
    print("---------------------------------------------")
    print(query)
    print("---------------------------------------------")
    print("[Answer]")
    print("---------------------------------------------")
    print(response)

    # ── Step 6: Output Guardrails ──────────────────────
    try:
        run_output_guardrails(response, context)
    except ValueError as e:
        print(f"\n[rag] Blocked by output guardrail: {e}")
        return str(e)

    return response, retrieved_chunks
