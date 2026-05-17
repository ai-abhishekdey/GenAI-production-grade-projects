"""
embedding.py
------------
Initialises the embedding model used for indexing and semantic chunking.
"""

from langchain_openai import OpenAIEmbeddings
from config import TEXT_EMBEDDING_MODEL, OPENAI_API_KEY
from src.observability.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------
# get_openai_embedder: returns an OpenAI embedder configured
# from config.py — raises early if the API key is missing
# -------------------------------------------------------------
def get_openai_embedder():
    if not OPENAI_API_KEY:
        raise ValueError("[embedding] OPENAI_API_KEY not found")

    embedder = OpenAIEmbeddings(model=TEXT_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    logger.info("embedder loaded", extra={"model": TEXT_EMBEDDING_MODEL})
    return embedder
