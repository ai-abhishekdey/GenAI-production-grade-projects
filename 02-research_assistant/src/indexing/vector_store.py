"""
vector_store.py
---------------
Creates and populates the vector store used for retrieval.
Supports Chroma (local) and Qdrant (remote).
"""

import os
import hashlib
import shutil

from config import (
    VECTOR_DB,
    CHROMA_DB_DIR,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
)
from src.observability.logger import get_logger

logger = get_logger(__name__)

# vector dimension for text-embedding-3-small — used when creating a new Qdrant collection
QDRANT_VECTOR_SIZE = 1536


# -------------------------------------------------------------
# generate_doc_id: creates a stable MD5 ID from content and
# source path to prevent duplicate embeddings across runs
# -------------------------------------------------------------
def generate_doc_id(doc, idx):
    base = doc.page_content + str(doc.metadata.get("source", "")) + str(idx)
    return hashlib.md5(base.encode("utf-8")).hexdigest()


# -------------------------------------------------------------
# ensure_qdrant_collection: creates the Qdrant collection if it
# doesn't exist yet. Uses get_collection() instead of
# collection_exists() because the /exists endpoint is missing
# on some Qdrant cloud versions.
# -------------------------------------------------------------
def ensure_qdrant_collection(client):
    from qdrant_client.models import Distance, VectorParams

    try:
        client.get_collection(QDRANT_COLLECTION)
        logger.debug("collection exists", extra={"collection": QDRANT_COLLECTION})
    except Exception:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("collection created", extra={"collection": QDRANT_COLLECTION})


# -------------------------------------------------------------
# get_vector_store: initialises the vector store backend.
# Chroma is cleared on each run to avoid stale embeddings.
# Qdrant uses the direct constructor to avoid a remote call on
# startup — collection is created lazily on first add_documents.
# -------------------------------------------------------------
def get_vector_store(embedder):
    if VECTOR_DB == "chroma":
        from langchain_chroma import Chroma

        # clear on each run so stale chunks from a previous experiment don't pollute results
        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)
            logger.info("chroma db cleared", extra={"path": CHROMA_DB_DIR})

        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedder)
        logger.info("chroma db ready", extra={"path": CHROMA_DB_DIR})
        return db

    elif VECTOR_DB == "qdrant":
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore

        # check_compatibility=False skips the version handshake that fails on some Qdrant cloud configs.
        # validate_collection_config=False skips a network call in the constructor — connection is
        # deferred to first use so startup doesn't block or fail if Qdrant is temporarily unreachable.
        # timeout=120 prevents write timeouts on large ingestion batches (default is 5 s).
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False, timeout=120)
        db = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embedder,
            validate_collection_config=False,
        )
        logger.info("qdrant client ready", extra={"collection": QDRANT_COLLECTION})
        return db

    else:
        raise ValueError(f"[vector-store] Unsupported VECTOR_DB: {VECTOR_DB}")


# -------------------------------------------------------------
# clear_vector_store: removes all vectors from the store so a
# fresh ingestion starts clean. For Qdrant this drops and
# recreates the collection. For Chroma it resets the directory.
# -------------------------------------------------------------
def clear_vector_store(vector_store):
    if VECTOR_DB == "qdrant":
        vector_store.client.delete_collection(QDRANT_COLLECTION)
        ensure_qdrant_collection(vector_store.client)
        logger.info("qdrant collection reset", extra={"collection": QDRANT_COLLECTION})

    elif VECTOR_DB == "chroma":
        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)
            logger.info("chroma db cleared", extra={"path": CHROMA_DB_DIR})


# -------------------------------------------------------------
# add_documents: embeds and adds documents to the vector store.
# For Qdrant, ensures the collection exists before upserting.
# Uses deterministic IDs so re-adding the same chunks is safe.
# -------------------------------------------------------------
UPSERT_BATCH_SIZE = 20   # max chunks per Qdrant write to avoid write timeouts


def add_documents(vector_store, documents, embedder):
    if not documents:
        raise ValueError("[vector-store] No documents to add")

    if VECTOR_DB == "qdrant":
        ensure_qdrant_collection(vector_store.client)

    embeddings = embedder.embed_documents([doc.page_content for doc in documents])
    logger.info("embeddings computed", extra={"count": len(embeddings), "dimension": len(embeddings[0])})

    ids = [generate_doc_id(doc, i) for i, doc in enumerate(documents)]

    # upsert in small batches so each write stays well within the timeout window
    for start in range(0, len(documents), UPSERT_BATCH_SIZE):
        batch_docs = documents[start: start + UPSERT_BATCH_SIZE]
        batch_ids  = ids[start: start + UPSERT_BATCH_SIZE]
        vector_store.add_documents(batch_docs, ids=batch_ids)
        logger.debug(
            "batch upserted",
            extra={"batch": start // UPSERT_BATCH_SIZE + 1, "batch_size": len(batch_docs)},
        )

    logger.info("indexing complete", extra={"doc_count": len(documents), "backend": VECTOR_DB})
