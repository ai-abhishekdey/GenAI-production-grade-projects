"""
retriever.py
------------
Retrieval strategies for fetching relevant chunks from the vector store.
Supports dense, MMR, sparse (BM25), and hybrid retrieval.
All functions are async — call with await from async contexts or asyncio.run() from sync.
"""

from config import RETRIEVAL_STRATEGY, TOP_K, MMR_LAMBDA, DENSE_WEIGHT, SPARSE_WEIGHT
from src.observability.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------
# retrieve: async entry point that dispatches to the correct
# strategy based on RETRIEVAL_STRATEGY set in config.py
# -------------------------------------------------------------
async def retrieve(vector_store, documents, query):
    if RETRIEVAL_STRATEGY == "dense":
        return await get_dense_retrieval(vector_store, query)
    elif RETRIEVAL_STRATEGY == "mmr":
        return await get_mmr_retrieval(vector_store, query)
    elif RETRIEVAL_STRATEGY == "sparse":
        return await get_sparse_retrieval(documents, query)
    elif RETRIEVAL_STRATEGY == "hybrid":
        return await get_hybrid_retrieval(vector_store, documents, query)
    else:
        raise ValueError(f"Invalid RETRIEVAL_STRATEGY: {RETRIEVAL_STRATEGY}")


# -------------------------------------------------------------
# get_dense_retrieval: standard cosine similarity search
# against the vector store
# -------------------------------------------------------------
async def get_dense_retrieval(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    results = await retriever.ainvoke(query)
    logger.info("dense retrieval", extra={"result_count": len(results)})
    return results, retriever


# -------------------------------------------------------------
# get_mmr_retrieval: maximal marginal relevance search —
# balances relevance with diversity to reduce redundant chunks
# -------------------------------------------------------------
async def get_mmr_retrieval(vector_store, query):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "lambda_mult": MMR_LAMBDA}
    )
    results = await retriever.ainvoke(query)
    logger.info("mmr retrieval", extra={"result_count": len(results)})
    return results, retriever


# -------------------------------------------------------------
# get_sparse_retrieval: keyword-based BM25 retrieval —
# works directly on raw document text, no embeddings needed
# -------------------------------------------------------------
async def get_sparse_retrieval(documents, query):
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = TOP_K
    # BM25 has no native async — ainvoke() falls back to a thread executor internally
    results = await retriever.ainvoke(query)
    logger.info("sparse retrieval", extra={"result_count": len(results)})
    return results, retriever


# -------------------------------------------------------------
# get_hybrid_retrieval: combines BM25 and dense retrieval using
# a weighted ensemble for better recall across query types
# -------------------------------------------------------------
async def get_hybrid_retrieval(vector_store, documents, query):
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    sparse_retriever = BM25Retriever.from_documents(documents)
    sparse_retriever.k = TOP_K

    dense_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[SPARSE_WEIGHT, DENSE_WEIGHT]
    )

    results = await retriever.ainvoke(query)
    logger.info("hybrid retrieval", extra={"result_count": len(results), "sparse_weight": SPARSE_WEIGHT, "dense_weight": DENSE_WEIGHT})
    return results, retriever
