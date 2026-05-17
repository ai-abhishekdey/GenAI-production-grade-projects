"""
llm.py
------
Initialises the LangChain LLM instances used across the pipeline.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config import LLM

load_dotenv()


# -------------------------------------------------------------
# get_openai_llm: returns the main pipeline LLM used for
# answering questions and guardrail checks
# -------------------------------------------------------------
def get_openai_llm():
    return ChatOpenAI(model=LLM, temperature=0.5, max_tokens=512)


# -------------------------------------------------------------
# get_summariser_llm: same model but with a higher token limit
# to accommodate longer report summaries
# -------------------------------------------------------------
def get_summariser_llm():
    return ChatOpenAI(model=LLM, temperature=0.5, max_tokens=2500)
