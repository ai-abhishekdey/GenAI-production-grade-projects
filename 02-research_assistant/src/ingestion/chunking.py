import os
import json
import re
from pathlib import Path
from langchain_core.documents import Document

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from landingai_ade import LandingAIADE
from src.ingestion.preprocessor import remove_references

# ── Utils ───────────────────────────────────────────────


def _validate_inputs(documents, chunk_size=None, chunk_overlap=None):
    if not documents:
        raise ValueError("[chunking] No documents provided")

    if chunk_size is not None and chunk_overlap is not None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "[chunking] chunk_overlap must be smaller than chunk_size")


# ── Fixed Size Chunking ─────────────────────────────────

def get_fixed_size_chunks(documents: list, chunk_size: int = 512, chunk_overlap: int = 64) -> list:
    _validate_inputs(documents, chunk_size, chunk_overlap)

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" "
    )

    chunks = splitter.split_documents(documents)

    print(
        f"[chunking:fixed] {len(chunks)} chunks | size={chunk_size}, overlap={chunk_overlap}")
    return chunks


# ── Recursive Chunking ──────────────────────────────────

def get_recursive_chunks(documents: list, chunk_size: int = 512, chunk_overlap: int = 64) -> list:
    _validate_inputs(documents, chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],  # added "" fallback
    )

    chunks = splitter.split_documents(documents)

    print(
        f"[chunking:recursive] {len(chunks)} chunks | size={chunk_size}, overlap={chunk_overlap}")
    return chunks


# ── Page-Level Chunking ─────────────────────────────────

def get_page_level_chunks(documents: list) -> list:

    _validate_inputs(documents)

    documents = remove_references(documents)

    print(f"[chunking:page] {len(documents)} chunks (1 per page)")
    return documents


# ── Semantic Chunking ───────────────────────────────────

def get_semantic_chunks(documents: list, embedder, breakpoint_threshold_type: str = "percentile") -> list:
    _validate_inputs(documents)

    splitter = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type=breakpoint_threshold_type
    )

    chunks = splitter.split_documents(documents)

    print(
        f"[chunking:semantic] {len(chunks)} chunks | threshold={breakpoint_threshold_type}")
    return chunks


# ── Layout-Aware Chunking (ADE) ─────────────────────────


def get_layout_aware_chunks(documents: list, api_key: str, output_dir: str = "ade_outputs") -> list:
    """
    Uses LandingAI ADE to parse document layout and generate structured chunks.
    """

    _validate_inputs(documents)
    os.makedirs(output_dir, exist_ok=True)

    client = None
    chunks = []
    pdf_sources = list(dict.fromkeys(
        doc.metadata.get("source")
        for doc in documents
        if doc.metadata.get("source")
    ))

    for pdf_path in pdf_sources:
        if not os.path.exists(pdf_path):
            print(f"[chunking:layout] Skipping invalid source: {pdf_path}")
            continue

        pdf_name = Path(pdf_path).name
        pdf_stem = Path(pdf_path).stem
        json_path = os.path.join(output_dir, f"{pdf_stem}_parse_output.json")
        markdown_path = os.path.join(output_dir, f"{pdf_name}.md")

        if not os.path.exists(json_path):
            print(f"[chunking:layout] Processing: {pdf_path}")
            if client is None:
                client = LandingAIADE(apikey=api_key)

            response = client.parse(
                document=Path(pdf_path),
                model="dpt-2",
                save_to=output_dir
            )

            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(response.markdown)
        else:
            print(f"[chunking:layout] Using cached ADE output: {json_path}")

        if not os.path.exists(json_path):
            print(f"[chunking:layout] ADE JSON not found: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ade_metadata = data.get("metadata", {})
        grounding_map = data.get("grounding", {})
        source_chunks = []
        for i, chunk in enumerate(data.get("chunks", [])):
            content = chunk.get("markdown", "").strip()
            content = re.sub(r"<a\s+id=['\"][^'\"]+['\"]></a>\s*", "", content)

            if not content:
                continue

            chunk_id = chunk.get("id")
            grounding = grounding_map.get(chunk_id, chunk.get("grounding", {}))
            box = grounding.get("box", {})
            metadata = {
                "source": pdf_path,
                "type": "layout",
                "layout_index": i,
                "ade_id": chunk_id,
                "ade_type": chunk.get("type"),
                "ade_object_type": grounding.get("type"),
                "page": grounding.get("page"),
                "box_top": box.get("top"),
                "box_left": box.get("left"),
                "box_right": box.get("right"),
                "box_bottom": box.get("bottom"),
                "confidence": grounding.get("confidence"),
                "low_confidence_spans": json.dumps(
                    grounding.get("low_confidence_spans", [])
                ),
                "ade_model": ade_metadata.get("version"),
                "page_count": ade_metadata.get("page_count"),
            }

            source_chunks.append(Document(
                page_content=content,
                metadata=metadata
            ))

        chunks.extend(remove_references(source_chunks))

    print(f"[chunking:layout] {len(chunks)} chunks created (layout-aware)")

    return chunks


# ── Hybrid Chunking (Layout + Semantic) ─────────────────

def get_hybrid_chunks(
    documents: list,
    embedder,
    api_key: str,
    output_dir: str = "ade_outputs",
    breakpoint_threshold_type: str = "percentile"
) -> list:
    """
    Combines layout-aware chunking with semantic chunking.

    Flow:
        1. Extract structured sections using ADE
        2. Apply semantic chunking on those sections

    Args:
        documents: List of LangChain Document objects
        embedder: Embedding model
        api_key: ADE API key
        output_dir: ADE output directory
        breakpoint_threshold_type: Semantic split strategy

    Returns:
        List of refined chunks
    """

    _validate_inputs(documents)

    # Step 1: Layout-aware chunking
    layout_chunks = get_layout_aware_chunks(
        documents,
        api_key=api_key,
        output_dir=output_dir
    )

    if not layout_chunks:
        print("[chunking:hybrid] No layout chunks found, falling back to semantic")
        return get_semantic_chunks(
            documents,
            embedder,
            breakpoint_threshold_type
        )

    # Step 2: Semantic refinement on layout chunks
    splitter = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type=breakpoint_threshold_type
    )

    refined_chunks = splitter.split_documents(layout_chunks)

    # Step 3: Clean + enrich metadata
    final_chunks = []
    for i, chunk in enumerate(refined_chunks):
        if not chunk.page_content.strip():
            continue

        # preserve original metadata + mark hybrid
        metadata = chunk.metadata.copy()
        metadata["type"] = "hybrid"

        final_chunks.append(Document(
            page_content=chunk.page_content,
            metadata=metadata
        ))

    print(
        f"[chunking:hybrid] {len(final_chunks)} chunks created (layout + semantic)"
    )

    return final_chunks
