import os

from langchain_core.documents import Document

from config import RETRIEVAL_STRATEGY, TOP_K, MMR_LAMBDA, DENSE_WEIGHT, SPARSE_WEIGHT

# ── Entry Point ─────────────────────────────────────────


def retrieve(vector_store, documents, query: str):
    if RETRIEVAL_STRATEGY == "dense":
        return get_dense_retrieval(vector_store, query)

    elif RETRIEVAL_STRATEGY == "mmr":
        return get_mmr_retrieval(vector_store, query)

    elif RETRIEVAL_STRATEGY == "sparse":
        return get_sparse_retrieval(documents, query)

    elif RETRIEVAL_STRATEGY == "hybrid":
        return get_hybrid_retrieval(vector_store, documents, query)

    else:
        raise ValueError(f"Invalid RETRIEVAL_STRATEGY: {RETRIEVAL_STRATEGY}")


# ── 1. Dense ───────────────────────────────────────────

def get_dense_retrieval(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    results = retriever.invoke(query)
    print(f"[retrieval:dense] {len(results)} results")
    return results, retriever


# ── 2. MMR ─────────────────────────────────────────────

def get_mmr_retrieval(vector_store, query):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "lambda_mult": MMR_LAMBDA}
    )
    results = retriever.invoke(query)
    print(f"[retrieval:mmr] {len(results)} results")
    return results, retriever


# ── 3. Sparse (BM25) ───────────────────────────────────

def get_sparse_retrieval(documents, query):
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = TOP_K
    results = retriever.invoke(query)
    print(f"[retrieval:sparse] {len(results)} results")
    return results, retriever


# ── 4. Hybrid (Dense + Sparse) ─────────────────────────

def get_hybrid_retrieval(vector_store, documents, query):
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    sparse_retriever = BM25Retriever.from_documents(documents)
    sparse_retriever.k = TOP_K

    dense_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[SPARSE_WEIGHT, DENSE_WEIGHT]
    )

    results = retriever.invoke(query)
    print(
        f"[retrieval:hybrid] {len(results)} results (sparse={SPARSE_WEIGHT}, dense={DENSE_WEIGHT})")
    return results, retriever
