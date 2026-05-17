"""
prompts.py
----------
Prompt templates for RAG-based QA.
"""

from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------------------
# get_basic_rag_prompt: standard prompt for answering questions
# from retrieved context — enforces complete sentence answers
# -------------------------------------------------------------
def get_basic_rag_prompt():
    template = """
You are a research assistant helping users understand academic papers.

Answer the question using only the context provided below. Write naturally — match the format to the content:
- For conceptual questions (novelty, conclusion, abstract, motivation), answer in clear, flowing prose.
- If the context contains tabular data (numbers, comparisons, metrics across multiple items), present that data as a Markdown table.
- If the context explicitly references a figure or image, mention it by figure number and caption.
- Use bullet points or numbered lists only when the answer is genuinely enumerable.

Do not force structure onto narrative answers.
Do not return short phrases — write complete sentences.

If the answer is not found in the context, say:
"I don't have enough information in the provided context to answer this question."

---------------------
Context:
{context}
---------------------

Question:
{question}

Answer:
"""
    return ChatPromptTemplate.from_template(template)


# -------------------------------------------------------------
# get_strict_rag_prompt: stricter version that explicitly
# forbids assumptions and external knowledge to reduce hallucination
# -------------------------------------------------------------
def get_strict_rag_prompt():
    template = """
You are a precise AI assistant.

Rules:
- Answer ONLY using the provided context
- Do NOT make assumptions or add external knowledge
- If unsure, say: "Not enough information in context"
- For conceptual or narrative questions, answer in prose — do not force structure
- If the context contains tabular data, present it as a Markdown table
- If the context references a figure or image, cite it by figure number and caption
- Use lists only when the answer is genuinely enumerable

---------------------
Context:
{context}
---------------------

Question:
{question}

Answer (concise and factual):
"""
    return ChatPromptTemplate.from_template(template)


# -------------------------------------------------------------
# get_debug_prompt: asks the model to analyse what is and isn't
# in the retrieved context — useful for diagnosing retrieval gaps
# -------------------------------------------------------------
def get_debug_prompt():
    template = """
You are debugging a retrieval system.

Given the context, explain:
1. What information is relevant to the question
2. What is missing
3. Whether the context contains tabular data or figure references that could improve the answer

If relevant tabular data is present in the context, show it as a Markdown table in your analysis.
If figures are referenced, note their number and caption.

Context:
{context}

Question:
{question}

Analysis:
"""
    return ChatPromptTemplate.from_template(template)
