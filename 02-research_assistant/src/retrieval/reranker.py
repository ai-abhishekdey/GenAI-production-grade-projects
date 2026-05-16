"""
reranker.py
-----------
Re-ranks retrieved chunks using a cross-encoder model.
Wraps a base retriever with ContextualCompressionRetriever.

Both functions take a base_retriever as input and return a
ContextualCompressionRetriever — drop-in replacement for any retriever.
"""


from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from config import RERANKER_TOP_N, CROSSENCODER_MODEL


def get_crossencoder_reranker(base_retriever):
    """
    Re-ranks retrieved chunks using a HuggingFace cross-encoder model.
    Runs locally, no API key needed.

    Args:
        base_retriever: Any LangChain retriever object from retriever.py

    Returns:
        ContextualCompressionRetriever with cross-encoder reranking
    """

    model = HuggingFaceCrossEncoder(model_name=CROSSENCODER_MODEL)

    compressor = CrossEncoderReranker(
        model=model,
        top_n=RERANKER_TOP_N
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    print(
        f"[reranker] CrossEncoder loaded (model={CROSSENCODER_MODEL}, top_n={RERANKER_TOP_N})")
    return retriever


def get_flashrank_reranker(base_retriever):
    """
    Re-ranks retrieved chunks using FlashrankRerank.
    Lightweight and fast, runs locally, no API key needed.

    Args:
        base_retriever: Any LangChain retriever object from retriever.py

    Returns:
        ContextualCompressionRetriever with flashrank reranking
    """

    compressor = FlashrankRerank(top_n=RERANKER_TOP_N)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    print(f"[reranker] Flashrank loaded (top_n={RERANKER_TOP_N})")
    return retriever
