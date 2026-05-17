"""
preprocessor.py
---------------
Text cleaning and filtering applied to raw documents before chunking.
"""

from langchain_core.documents import Document

from src.observability.logger import get_logger

logger = get_logger(__name__)

REFERENCE_HEADINGS = ["references", "bibliography", "works cited"]


# -------------------------------------------------------------
# clean_text: fixes hyphenated line breaks and normalises
# whitespace in a single page of text
# -------------------------------------------------------------
def clean_text(text):
    # rejoin words split across lines with a hyphen (common in PDFs)
    lines = text.split("\n")
    for i in range(len(lines) - 1):
        if lines[i].endswith("-"):
            lines[i] = lines[i][:-1] + lines[i + 1].strip()
            lines[i + 1] = ""

    text = "\n".join(lines)
    text = " ".join(text.split())
    return text.strip()


# -------------------------------------------------------------
# remove_references: scans pages in order and drops everything
# from the first reference heading onwards
# -------------------------------------------------------------
def remove_references(documents):
    filtered = []

    for doc in documents:
        lines = doc.page_content.split("\n")
        ref_line_index = None

        for i, line in enumerate(lines):
            if line.strip().lower() in REFERENCE_HEADINGS:
                ref_line_index = i
                break

        if ref_line_index is not None:
            trimmed_content = "\n".join(lines[:ref_line_index]).strip()
            if trimmed_content:
                filtered.append(Document(page_content=trimmed_content, metadata=doc.metadata))
            logger.debug("reference section found — truncating")
            break

        filtered.append(doc)

    logger.info("reference removal done", extra={"doc_count": len(filtered)})
    return filtered


# -------------------------------------------------------------
# preprocess_documents: runs the full cleaning pipeline —
# reference removal first, then whitespace normalisation.
# Order matters: clean_text collapses newlines which would
# break the heading detection in remove_references.
# -------------------------------------------------------------
def preprocess_documents(documents):
    logger.info("preprocessing started", extra={"doc_count": len(documents)})

    # remove references before cleaning — clean_text collapses newlines
    # which breaks heading detection
    documents = remove_references(documents)

    processed = [
        Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
        for doc in documents
    ]

    logger.info("preprocessing complete", extra={"doc_count": len(processed)})
    return processed
