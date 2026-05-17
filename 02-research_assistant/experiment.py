"""
experiment.py
-------------
Runs the full RAG pipeline and evaluates results against a test set.

Usage:
    python experiment.py
"""

import os
import json
import asyncio
from datetime import datetime
from config import *
from src.ingestion.loader import *
from src.ingestion.preprocessor import *
from src.ingestion.chunking import *
from src.indexing.embedding import *
from src.indexing.vector_store import *
from src.generation.rag import run_rag
# from src.evaluation.eval_ragas import evaluate_predictions
from src.evaluation.eval_deepeval import evaluate_predictions
from src.evaluation.save_results import save_experiment_results
from src.observability.latency_tracker import LatencyTracker
from src.observability.token_tracker import TokenTracker
from src.observability.logger import setup_logging, get_logger


if __name__ == "__main__":

    # run_id ties the log file to the results files saved by save_experiment_results
    run_id   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = setup_logging(log_type="experiment", run_id=run_id)
    logger   = get_logger(__name__)
    logger.info("experiment started", extra={"run_id": run_id, "log_file": str(log_file) if log_file else "stdout only"})

    latency_tracker = LatencyTracker()
    token_tracker   = TokenTracker()
    embedder        = get_openai_embedder()

    # ---------------------------------------------------------
    # Step 1: Data Ingestion
    # ---------------------------------------------------------
    logger.info("step started", extra={"step": 1, "step_name": "data ingestion"})
    latency_tracker.start("ingestion")

    if SOURCE_TYPE == "local":
        documents = load_pdf(LOCAL_PDF_PATH)
    elif SOURCE_TYPE == "folder":
        documents = load_pdfs_from_folder(LOCAL_PDF_FOLDER)
    elif SOURCE_TYPE == "url":
        documents = load_pdf_from_url(ARXIV_URL)
    else:
        raise ValueError(f"Invalid SOURCE_TYPE: {SOURCE_TYPE}")

    latency_tracker.stop("ingestion")

    # ---------------------------------------------------------
    # Step 2: Preprocessing
    # ---------------------------------------------------------
    logger.info("step started", extra={"step": 2, "step_name": "preprocessing"})
    latency_tracker.start("preprocessing")
    processed_docs = preprocess_documents(documents)
    latency_tracker.stop("preprocessing")

    # ---------------------------------------------------------
    # Step 3: Chunking
    # ---------------------------------------------------------
    logger.info("step started", extra={"step": 3, "step_name": "chunking", "strategy": CHUNK_STRATEGY})
    latency_tracker.start("chunking")

    if CHUNK_STRATEGY == "fixed":
        chunks = get_fixed_size_chunks(processed_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    elif CHUNK_STRATEGY == "recursive":
        chunks = get_recursive_chunks(processed_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    elif CHUNK_STRATEGY == "page":
        chunks = get_page_level_chunks(processed_docs)
    elif CHUNK_STRATEGY == "semantic":
        chunks = get_semantic_chunks(processed_docs, embedder)
    elif CHUNK_STRATEGY == "layout":
        if not ADE_API_KEY:
            raise ValueError("[experiment] ADE_API_KEY not set in config")
        chunks = get_layout_aware_chunks(processed_docs, ADE_API_KEY, ADE_OUTPUT_DIR)
    elif CHUNK_STRATEGY == "hybrid":
        if not ADE_API_KEY:
            raise ValueError("[experiment] ADE_API_KEY not set in config")
        chunks = get_hybrid_chunks(processed_docs, embedder, ADE_API_KEY, ADE_OUTPUT_DIR)
    else:
        raise ValueError(f"Invalid CHUNK_STRATEGY: {CHUNK_STRATEGY}")

    latency_tracker.stop("chunking")

    # ---------------------------------------------------------
    # Step 4: Vector Store
    # ---------------------------------------------------------
    logger.info("step started", extra={"step": 4, "step_name": "vector store"})
    latency_tracker.start("vector_store")
    vector_store = get_vector_store(embedder)
    add_documents(vector_store, chunks, embedder)
    latency_tracker.stop("vector_store")

    # ---------------------------------------------------------
    # Step 5: RAG
    # ---------------------------------------------------------
    logger.info("step started", extra={"step": 5, "step_name": "rag + evaluation"})

    if not os.path.exists(TESTSET_PATH):
        raise FileNotFoundError(f"Test set not found: {TESTSET_PATH}")

    with open(TESTSET_PATH, "r") as f:
        test_data = json.load(f)

    questions, predictions, ground_truths, contexts = [], [], [], []

    latency_tracker.start("rag")
    token_tracker.start()

    for i, qa in enumerate(test_data):
        logger.info("running question", extra={"q_num": i + 1, "total": len(test_data)})
        answer, retrieved_chunks = asyncio.run(run_rag(vector_store, chunks, qa["question"]))
        questions.append(qa["question"])
        predictions.append(answer)
        ground_truths.append(qa["answer"])
        contexts.append([c.page_content for c in retrieved_chunks])

    token_tracker.stop("rag")
    latency_tracker.stop("rag")

    # ---------------------------------------------------------
    # Step 6: Evaluation
    # ---------------------------------------------------------
    logger.info("step started", extra={"step": 6, "step_name": "evaluation"})
    latency_tracker.start("evaluation")
    results = evaluate_predictions(questions, predictions, ground_truths, contexts)
    latency_tracker.stop("evaluation")

    # ---------------------------------------------------------
    # Step 7: Save Results
    # ---------------------------------------------------------
    logger.info("step started", extra={"step": 7, "step_name": "saving results"})

    config_dict = {
        "chunking":  CHUNK_STRATEGY,
        "retrieval": RETRIEVAL_STRATEGY,
        "reranker":  {"enabled": USE_RERANKER, "type": RERANKER_TYPE if USE_RERANKER else None}
    }

    save_experiment_results(
        results=results,
        test_data=test_data,
        predictions=predictions,
        config=config_dict,
        latency=latency_tracker.get_latency(),
        token_usage=token_tracker.get_usage(),
        runs_dir=RUNS_DIR,
        run_id=run_id
    )

    latency_tracker.log_report(logger)
    token_tracker.log_report(logger)

    logger.info("experiment complete", extra={"run_id": run_id})
