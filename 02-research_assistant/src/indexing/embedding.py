from langchain_openai import OpenAIEmbeddings
from config import TEXT_EMBEDDING_MODEL, OPENAI_API_KEY


def get_openai_embedder():
    """
    Returns an OpenAI embedding model using config values.
    """

    if not OPENAI_API_KEY:
        raise ValueError("[embedding] OPENAI_API_KEY not found")

    embedder = OpenAIEmbeddings(
        model=TEXT_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )

    print(f"[text-embedding] Loaded OpenAI embedder: {TEXT_EMBEDDING_MODEL}")
    return embedder
