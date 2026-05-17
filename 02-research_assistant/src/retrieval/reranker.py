"""
reranker.py
-----------
Re-ranks retrieved chunks to improve precision before generation.
Both functions wrap a base retriever with ContextualCompressionRetriever,
making them a drop-in replacement for any retriever in the pipeline.
"""

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from config import RERANKER_TOP_N, CROSSENCODER_MODEL
from src.observability.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------
# get_crossencoder_reranker: reranks using a HuggingFace
# cross-encoder model — more accurate but slower than flashrank
# -------------------------------------------------------------
def get_crossencoder_reranker(base_retriever):
    model = HuggingFaceCrossEncoder(model_name=CROSSENCODER_MODEL)
    compressor = CrossEncoderReranker(model=model, top_n=RERANKER_TOP_N)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    logger.info("crossencoder reranker loaded", extra={"model": CROSSENCODER_MODEL, "top_n": RERANKER_TOP_N})
    return retriever


# -------------------------------------------------------------
# get_flashrank_reranker: reranks using FlashRank — lightweight
# and fast, good default when latency matters
# -------------------------------------------------------------
def get_flashrank_reranker(base_retriever):
    compressor = FlashrankRerank(top_n=RERANKER_TOP_N)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    logger.info("flashrank reranker loaded", extra={"top_n": RERANKER_TOP_N})
    return retriever
