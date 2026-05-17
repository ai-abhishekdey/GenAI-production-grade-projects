"""
chunking.py
-----------
Chunking strategies for splitting documents before indexing.
Supports fixed-size, recursive, page-level, semantic, layout-aware, and hybrid.
"""

import os
import json
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from landingai_ade import LandingAIADE
from src.ingestion.preprocessor import remove_references
from src.observability.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------
# validate_inputs: checks documents are non-empty and that
# chunk_overlap is strictly less than chunk_size
# -------------------------------------------------------------
def validate_inputs(documents, chunk_size=None, chunk_overlap=None):
    if not documents:
        raise ValueError("[chunking] No documents provided")
    if chunk_size is not None and chunk_overlap is not None:
        if chunk_overlap >= chunk_size:
            raise ValueError("[chunking] chunk_overlap must be smaller than chunk_size")


# -------------------------------------------------------------
# get_fixed_size_chunks: splits text into equal-sized chunks
# using a single space as the separator to avoid mid-word cuts
# -------------------------------------------------------------
def get_fixed_size_chunks(documents, chunk_size=512, chunk_overlap=64):
    validate_inputs(documents, chunk_size, chunk_overlap)

    # split on spaces to avoid cutting mid-word
    chunks = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" "
    ).split_documents(documents)

    logger.info("fixed chunks created", extra={"chunk_count": len(chunks), "chunk_size": chunk_size, "overlap": chunk_overlap})
    return chunks


# -------------------------------------------------------------
# get_recursive_chunks: splits text by trying progressively
# smaller separators until chunks fit within chunk_size
# -------------------------------------------------------------
def get_recursive_chunks(documents, chunk_size=512, chunk_overlap=64):
    validate_inputs(documents, chunk_size, chunk_overlap)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],  # "" ensures nothing is left unsplit
    ).split_documents(documents)

    logger.info("recursive chunks created", extra={"chunk_count": len(chunks), "chunk_size": chunk_size, "overlap": chunk_overlap})
    return chunks


# -------------------------------------------------------------
# get_page_level_chunks: treats each page as a single chunk
# after stripping reference sections
# -------------------------------------------------------------
def get_page_level_chunks(documents):
    validate_inputs(documents)
    documents = remove_references(documents)
    logger.info("page chunks created", extra={"chunk_count": len(documents)})
    return documents


# -------------------------------------------------------------
# get_semantic_chunks: uses an embedding model to find natural
# topic boundaries and splits there instead of at fixed sizes
# -------------------------------------------------------------
def get_semantic_chunks(documents, embedder, breakpoint_threshold_type="percentile"):
    validate_inputs(documents)

    chunks = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type=breakpoint_threshold_type
    ).split_documents(documents)

    logger.info("semantic chunks created", extra={"chunk_count": len(chunks), "threshold": breakpoint_threshold_type})
    return chunks


# -------------------------------------------------------------
# get_layout_aware_chunks: uses LandingAI ADE to parse the
# document layout (tables, figures, sections) before chunking.
# ADE output is cached to avoid repeated API calls.
# -------------------------------------------------------------
def get_layout_aware_chunks(documents, api_key, output_dir="ade_outputs"):
    validate_inputs(documents)
    os.makedirs(output_dir, exist_ok=True)

    # deduplicate sources while preserving order
    pdf_sources = list(dict.fromkeys(
        doc.metadata.get("source")
        for doc in documents
        if doc.metadata.get("source")
    ))

    client = None  # lazy init — only create if cached ADE output doesn't exist
    chunks = []

    for pdf_path in pdf_sources:
        if not os.path.exists(pdf_path):
            logger.warning("skipping invalid source", extra={"path": pdf_path})
            continue

        pdf_name = Path(pdf_path).name
        pdf_stem = Path(pdf_path).stem
        json_path = os.path.join(output_dir, f"{pdf_stem}_parse_output.json")
        markdown_path = os.path.join(output_dir, f"{pdf_name}.md")

        if not os.path.exists(json_path):
            logger.info("processing pdf", extra={"path": pdf_path})
            if client is None:
                client = LandingAIADE(apikey=api_key)

            response = client.parse(document=Path(pdf_path), model="dpt-2", save_to=output_dir)
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(response.markdown)
        else:
            logger.debug("using cached ade output", extra={"path": json_path})

        # ADE parse can occasionally fail to produce the JSON even after a successful call
        if not os.path.exists(json_path):
            logger.warning("ade json not found, skipping", extra={"path": json_path})
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ade_metadata = data.get("metadata", {})
        grounding_map = data.get("grounding", {})
        source_chunks = []

        for i, chunk in enumerate(data.get("chunks", [])):
            content = chunk.get("markdown", "").strip()
            # ADE adds HTML anchor tags to markdown headings — strip them out
            content = re.sub(r"<a\s+id=['\"][^'\"]+['\"]></a>\s*", "", content)
            if not content:
                continue

            chunk_id = chunk.get("id")
            grounding = grounding_map.get(chunk_id, chunk.get("grounding", {}))
            box = grounding.get("box", {})

            source_chunks.append(Document(
                page_content=content,
                metadata={
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
                    # metadata values must be scalars, so serialize the list
                    "low_confidence_spans": json.dumps(grounding.get("low_confidence_spans", [])),
                    "ade_model": ade_metadata.get("version"),
                    "page_count": ade_metadata.get("page_count"),
                }
            ))

        chunks.extend(remove_references(source_chunks))

    logger.info("layout chunks created", extra={"chunk_count": len(chunks)})
    return chunks


# -------------------------------------------------------------
# get_hybrid_chunks: runs layout-aware chunking first, then
# applies semantic chunking on top for finer-grained splits.
# Falls back to pure semantic if ADE produces no chunks.
# -------------------------------------------------------------
def get_hybrid_chunks(documents, embedder, api_key, output_dir="ade_outputs", breakpoint_threshold_type="percentile"):
    validate_inputs(documents)

    layout_chunks = get_layout_aware_chunks(documents, api_key=api_key, output_dir=output_dir)

    # ADE can fail on some PDFs — fall back to pure semantic chunking
    if not layout_chunks:
        logger.warning("no layout chunks found, falling back to semantic")
        return get_semantic_chunks(documents, embedder, breakpoint_threshold_type)

    refined_chunks = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type=breakpoint_threshold_type
    ).split_documents(layout_chunks)

    final_chunks = []
    for chunk in refined_chunks:
        if not chunk.page_content.strip():
            continue
        metadata = chunk.metadata.copy()
        metadata["type"] = "hybrid"
        final_chunks.append(Document(page_content=chunk.page_content, metadata=metadata))

    logger.info("hybrid chunks created", extra={"chunk_count": len(final_chunks)})
    return final_chunks
