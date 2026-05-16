"""
experiments.py
--------------
Run full RAG pipeline + evaluation on test set.

Usage:
    python experiments.py
"""

import os
import json
from datetime import datetime
from config import *
from src.ingestion.loader import *
from src.ingestion.preprocessor import *
from src.ingestion.chunking import *
from src.indexing.embedding import *
from src.indexing.vector_store import *
from src.generation.rag import run_rag
from src.retrieval.retriever import retrieve
# from src.evaluation.eval_ragas import evaluate_predictions
from src.evaluation.eval_deepeval import evaluate_predictions
from src.evaluation.save_results import save_experiment_results
from src.observability.latency_tracker import LatencyTracker
from src.observability.token_tracker import TokenTracker
from src.observability.logger import get_logger, tee_stdout_to_file

if __name__ == "__main__":

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"log_{run_id}.txt")
    tee_stdout_to_file(log_path)
    logger = get_logger("pipeline")
    logger.info("Run started | run_id=%s", run_id)

    latency_tracker = LatencyTracker()
    token_tracker = TokenTracker()
    embedder = get_openai_embedder()

    print("\n========================================")
    print("STEP 1 : DATA INGESTION")
    print("========================================")

    # ── Ingestion ───────────────────────────────────────

    latency_tracker.start("ingestion")

    if SOURCE_TYPE == "local":
        if not os.path.exists(LOCAL_PDF_PATH):
            raise FileNotFoundError(f"{LOCAL_PDF_PATH} not found")
        documents = load_pdf(LOCAL_PDF_PATH)

    elif SOURCE_TYPE == "folder":
        if not os.path.exists(LOCAL_PDF_FOLDER):
            raise FileNotFoundError(f"{LOCAL_PDF_FOLDER} not found")
        documents = load_pdfs_from_folder(LOCAL_PDF_FOLDER)

    elif SOURCE_TYPE == "url":
        documents = load_pdf_from_url(ARXIV_URL)

    else:
        raise ValueError("Invalid SOURCE_TYPE")

    latency_tracker.stop("ingestion")

    print("\n========================================")
    print("STEP 2 : DATA PRE-PROCESSING")
    print("========================================")

    latency_tracker.start("preprocessing")

    processed_docs = preprocess_documents(documents)

    latency_tracker.stop("preprocessing")

    print("\n========================================")
    print("STEP 3 : CHUNKING")
    print("========================================")

    # ── Chunking ────────────────────────────────────────

    latency_tracker.start("chunking")

    if CHUNK_STRATEGY == "fixed":
        chunks = get_fixed_size_chunks(
            processed_docs, CHUNK_SIZE, CHUNK_OVERLAP)

    elif CHUNK_STRATEGY == "recursive":
        chunks = get_recursive_chunks(
            processed_docs, CHUNK_SIZE, CHUNK_OVERLAP)

    elif CHUNK_STRATEGY == "page":
        chunks = get_page_level_chunks(processed_docs)

    elif CHUNK_STRATEGY == "semantic":
        chunks = get_semantic_chunks(processed_docs, embedder)

    elif CHUNK_STRATEGY == "layout":
        if not ADE_API_KEY:
            raise ValueError("[experiments] ADE_API_KEY not found in config")
        chunks = get_layout_aware_chunks(
            processed_docs, ADE_API_KEY, ADE_OUTPUT_DIR)

    elif CHUNK_STRATEGY == "hybrid":
        if not ADE_API_KEY:
            raise ValueError("[experiments] ADE_API_KEY not found in config")

        chunks = get_hybrid_chunks(
            processed_docs,
            embedder,
            ADE_API_KEY,
            ADE_OUTPUT_DIR
        )

    else:
        raise ValueError("Invalid CHUNK_STRATEGY")

    latency_tracker.stop("chunking")

    # ── Embedding + Vector Store ────────────────────────

    print("\n========================================")
    print("STEP 4 : VECTOR STORE")
    print("========================================")

    latency_tracker.start("vector_store")

    vector_store = get_vector_store(embedder)

    add_documents(vector_store, chunks, embedder)

    latency_tracker.stop("vector_store")

    print("\n========================================")
    print("STEP 5 : RAG + EVALUATION")
    print("========================================")

    # ── Load test set ───────────────────────────────────

    if not os.path.exists(TESTSET_PATH):
        raise FileNotFoundError(f"{TESTSET_PATH} not found")

    with open(TESTSET_PATH, "r") as f:
        test_data = json.load(f)

    questions = []
    predictions = []
    ground_truths = []
    contexts = []

    # ── Run RAG for each question ───────────────────────

    latency_tracker.start("rag")
    token_tracker.start()

    for i, qa in enumerate(test_data):

        print(f"\n[Running Q{i+1}]")

        query = qa["question"]
        gt = qa["answer"]

        # run rag pipeline to generate the answer

        answer, retrieved_chunks = run_rag(vector_store, chunks, query)
        context_texts = [c.page_content for c in retrieved_chunks]

        # Collect
        questions.append(query)
        predictions.append(answer)
        ground_truths.append(gt)
        contexts.append(context_texts)

    token_tracker.stop("rag")
    latency_tracker.stop("rag")

    # ── Evaluation ──────────────────────────────────────

    print("\n========================================")
    print("STEP 6 : EVALUATION")
    print("========================================")

    latency_tracker.start("evaluation")

    results = evaluate_predictions(
        questions,
        predictions,
        ground_truths,
        contexts
    )

    latency_tracker.stop("evaluation")

    # ── Save Results ────────────────────────────────────

    print("\n========================================")
    print("STEP 7 : SAVING RESULTS")
    print("========================================")

    config_dict = {
        "chunking": CHUNK_STRATEGY,
        "retrieval": RETRIEVAL_STRATEGY,
        "reranker": {
            "enabled": USE_RERANKER,
            "type": RERANKER_TYPE if USE_RERANKER else None
        }
    }

    latency_dict = latency_tracker.get_latency()

    token_usage = token_tracker.get_usage()

    save_experiment_results(
        results=results,
        test_data=test_data,
        predictions=predictions,
        config=config_dict,
        latency=latency_dict,
        token_usage=token_usage,
        runs_dir=RUNS_DIR,
        run_id=run_id
    )

    latency_tracker.print_latency()

    token_tracker.print_report()

    print("\n=======================================")
    print("Done !")
    print("========================================")
