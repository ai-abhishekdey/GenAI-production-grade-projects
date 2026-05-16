"""
prompts.py
----------
Defines prompt templates for RAG-based QA.
"""

from langchain_core.prompts import ChatPromptTemplate


# ── 1. Basic RAG Prompt ───────────────────────────────

def get_basic_rag_prompt():
    """
    Standard RAG prompt.
    Uses retrieved context to answer the question.
    """

    template = """
You are a research assistant helping users understand academic papers.

Answer the question in a complete sentence using only the context provided below.

Make sure your answer explicitly reflects the key terms in the question (e.g., dataset, training set, results, etc.).

Do not return short phrases.

If the answer is not found in the context, say:
"I don't have enough information in the provided context to answer this question."

Keep your answer clear and precise.

---------------------
Context:
{context}
---------------------

Question:
{question}

Answer:
"""

    return ChatPromptTemplate.from_template(template)


# ── 2. Strict Prompt (low hallucination) ──────────────

def get_strict_rag_prompt():
    """
    More strict version to reduce hallucination.
    """

    template = """
You are a precise AI assistant.

Rules:
- Answer ONLY using the provided context
- Do NOT make assumptions
- Do NOT add external knowledge
- If unsure, say: "Not enough information in context"

---------------------
Context:
{context}
---------------------

Question:
{question}

Answer (concise and factual):
"""

    return ChatPromptTemplate.from_template(template)


# ── 3. Debug Prompt (optional) ────────────────────────

def get_debug_prompt():
    """
    Useful for inspecting what the model sees.
    """

    template = """
You are debugging a retrieval system.

Given the context, explain:
1. What information is relevant to the question
2. What is missing

Context:
{context}

Question:
{question}

Analysis:
"""

    return ChatPromptTemplate.from_template(template)
