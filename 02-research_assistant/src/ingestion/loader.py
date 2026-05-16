"""
data_ingestion.py
-----------------
Loads research papers (local or remote) into LangChain Document objects.
Each page becomes one Document with metadata: {"source": ..., "page": ...}
"""

import os
import tempfile
from collections import defaultdict

import requests
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


def normalize_arxiv_url(url: str) -> str:
    """Converts arxiv abstract URL to direct PDF URL. Non-arxiv URLs are unchanged."""
    if "arxiv.org/abs/" in url:
        url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/")
    return url


def load_pdf(pdf_path: str) -> list:
    """
    Loads a single local PDF file.

    Args:
        pdf_path: Path to a .pdf file

    Returns:
        List of Document objects, one per page
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"[data-ingestion] PDF not found: {pdf_path}")

    if not pdf_path.endswith(".pdf"):
        raise ValueError(
            f"[data-ingestion] Expected a .pdf file, got: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"[data-ingestion] Loaded {len(documents)} pages from '{pdf_path}'")
    return documents


def load_pdfs_from_folder(folder_path: str) -> list:
    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"[data-ingestion] Folder not found: {folder_path}")

    if not os.path.isdir(folder_path):
        raise ValueError(
            f"[data-ingestion] Expected a folder, got a file: {folder_path}")

    loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()

    print("-----------------------------------------------------")
    print(f"[data-ingestion] Total pages loaded : {len(documents)}")
    print("-----------------------------------------------------")

    # ── Group by PDF source ──────────────────────────────
    pdf_page_counts = defaultdict(int)

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        pdf_page_counts[source] += 1

    for pdf, count in pdf_page_counts.items():
        print(f"PDF Source : {pdf}")
        print(f"Page Count = {count}")
        print("-----------------------------------------------------")

    return documents


def load_pdf_from_url(url: str) -> list:
    """
    Downloads and loads a PDF from a remote URL.
    Handles arxiv abstract URLs, arxiv PDF URLs, and any direct .pdf link.

    Args:
        url: Arxiv or direct PDF URL

    Returns:
        List of Document objects, one per page
    """
    url = normalize_arxiv_url(url)

    response = requests.get(url, stream=True, timeout=30)

    if response.status_code != 200:
        raise ValueError(
            f"[data-ingestion] Failed to download PDF (status {response.status_code}): {url}")

    # PyPDFLoader requires a file path, so we write to a temp file then clean up
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_path = tmp_file.name
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Replace temp path with original URL in metadata
        for doc in documents:
            doc.metadata["source"] = url

    finally:
        os.remove(tmp_path)

    print(f"[data-ingestion] Loaded {len(documents)} pages from URL '{url}'")
    return documents
