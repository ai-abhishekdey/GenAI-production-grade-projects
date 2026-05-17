"""
loader.py
---------
Loads research papers (local or remote) into LangChain Document objects.
Each page becomes one Document with metadata: {"source": ..., "page": ...}
"""

import os
import tempfile
from collections import defaultdict

import requests
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

from src.observability.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------
# normalize_arxiv_url: converts an arxiv abstract URL to the
# direct PDF URL so it can be downloaded
# -------------------------------------------------------------
def normalize_arxiv_url(url):
    # arxiv abstract pages don't serve PDFs — rewrite to the /pdf/ endpoint
    if "arxiv.org/abs/" in url:
        url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/")
    return url


# -------------------------------------------------------------
# load_pdf: loads a single local PDF file and returns one
# Document per page
# -------------------------------------------------------------
def load_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"[loader] PDF not found: {pdf_path}")
    if not pdf_path.endswith(".pdf"):
        raise ValueError(f"[loader] Expected a .pdf file, got: {pdf_path}")

    documents = PyPDFLoader(pdf_path).load()
    logger.info("pdf loaded", extra={"path": pdf_path, "page_count": len(documents)})
    return documents


# -------------------------------------------------------------
# load_pdfs_from_folder: loads all PDFs in a folder recursively
# and prints a per-file page breakdown for easy verification
# -------------------------------------------------------------
def load_pdfs_from_folder(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"[loader] Folder not found: {folder_path}")
    if not os.path.isdir(folder_path):
        raise ValueError(f"[loader] Expected a folder, got a file: {folder_path}")

    documents = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    ).load()

    page_counts = defaultdict(int)
    for doc in documents:
        page_counts[doc.metadata.get("source", "unknown")] += 1

    logger.info(
        "folder loaded",
        extra={"total_pages": len(documents), "files": dict(page_counts)},
    )
    return documents


# -------------------------------------------------------------
# load_pdf_from_url: downloads a PDF from a URL and loads it.
# Handles arxiv abstract URLs automatically.
# -------------------------------------------------------------
def load_pdf_from_url(url):
    url = normalize_arxiv_url(url)

    response = requests.get(url, stream=True, timeout=30)
    if response.status_code != 200:
        raise ValueError(f"[loader] Failed to download PDF (status {response.status_code}): {url}")

    # PyPDFLoader only accepts a file path, so we write to a temp file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        documents = PyPDFLoader(tmp_path).load()
        # replace the temp path with the original URL so downstream metadata is meaningful
        for doc in documents:
            doc.metadata["source"] = url
    finally:
        os.remove(tmp_path)

    logger.info("url loaded", extra={"url": url, "page_count": len(documents)})
    return documents
