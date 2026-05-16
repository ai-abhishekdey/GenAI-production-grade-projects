from langchain_core.documents import Document

# Common reference section headings in research papers
REFERENCE_HEADINGS = ["references", "bibliography", "works cited"]


# ── Basic Text Cleaning ─────────────────────────────────

def clean_text(text: str) -> str:
    # Fix hyphenated line breaks
    lines = text.split("\n")
    for i in range(len(lines) - 1):
        if lines[i].endswith("-"):
            lines[i] = lines[i][:-1] + lines[i + 1].strip()
            lines[i + 1] = ""

    text = "\n".join(lines)

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


def remove_references(documents: list) -> list:
    # Scan each document for a reference heading
    # Once found, trim that document and drop everything after it
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
                filtered.append(Document(
                    page_content=trimmed_content,
                    metadata=doc.metadata
                ))
            print(
                f"[data-preprocessing] References found in the document — stopping here")
            break

        filtered.append(doc)

    if filtered:
        last_doc_lines = [
            line for line in filtered[-1].page_content.split("\n") if line.strip() != ""]
        print(f"[data-preprocessing] Last 2 lines of filtered content:")
        print(f"  {last_doc_lines[-2] if len(last_doc_lines) >= 2 else ''}")
        print(f"  {last_doc_lines[-1] if len(last_doc_lines) >= 1 else ''}")

    print(
        f"[data-preprocessing] {len(filtered)} documents after reference removal")
    return filtered


def preprocess_documents(documents: list) -> list:
    print(f"[data-preprocessing] Starting with {len(documents)} documents")

    # Remove references first — before whitespace normalization
    # clean_text collapses newlines which breaks heading detection
    documents = remove_references(documents)

    processed = [
        Document(
            page_content=clean_text(doc.page_content),
            metadata=doc.metadata
        )
        for doc in documents
    ]

    print(f"[data-preprocessing] Cleaned {len(processed)} documents")
    return processed
