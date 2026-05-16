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


# ── Helper: Generate deterministic ID ───────────────────

def generate_doc_id(doc, idx):
    """
    Creates a stable ID based on content + source.
    Prevents duplicate embeddings across runs.
    """
    base = doc.page_content + str(doc.metadata.get("source", "") + str(idx))
    return hashlib.md5(base.encode("utf-8")).hexdigest()


# ── Create / Load Vector Store ──────────────────────────

def get_vector_store(embedder):
    if VECTOR_DB == "chroma":
        from langchain_chroma import Chroma

        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)
            print(f"[vector-store:chroma] Cleared existing DB: {CHROMA_DB_DIR}")

        os.makedirs(CHROMA_DB_DIR, exist_ok=True)

        db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedder,
        )

        print(f"[vector-store:chroma] Using {CHROMA_DB_DIR}")
        return db

    elif VECTOR_DB == "qdrant":
        from langchain_qdrant import QdrantVectorStore

        db = QdrantVectorStore.from_documents(
            documents=[],
            embedding=embedder,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=QDRANT_COLLECTION,
        )

        print(f"[vector-store:qdrant] Using collection: {QDRANT_COLLECTION}")
        return db

    else:
        raise ValueError(f"[vector-store] Unsupported VECTOR_DB: {VECTOR_DB}")


# ── Add Documents (with deduplication + embedding stats) ─

def add_documents(vector_store, documents: list, embedder):
    if not documents:
        raise ValueError("[vector-store] No documents to add")

    # --- Generate embeddings (for stats) ---
    texts = [doc.page_content for doc in documents]
    vector_embeddings = embedder.embed_documents(texts)

    print(
        f"[vector-store:{VECTOR_DB}] No. of embedding vectors :", len(vector_embeddings))
    print(f"[vector-store:{VECTOR_DB}] Dimension of each embedding vector :",
          len(vector_embeddings[0]))

    # --- Deduplication IDs ---

    ids = [generate_doc_id(doc, i) for i, doc in enumerate(documents)]

    # --- Add to vector store ---
    vector_store.add_documents(documents, ids=ids)

    print(
        f"[vector-store:{VECTOR_DB}] Added {len(documents)} documents in vector stor(de-duplicated)")
