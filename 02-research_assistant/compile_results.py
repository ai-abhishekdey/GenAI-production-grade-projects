"""
compile_results.py
------------------
Reads all experiment run JSON files and produces:
  1. experiment_summary.csv   — one row per run with all metrics flattened
  2. experiment_analytics.json — best/worst per metric, used by generate_report.py

Usage:
    python compile_results.py
"""

import os
import json
import pandas as pd

from config import (
    RUNS_DIR,
    COMPILED_RESULTS_DIR,
    COMPILED_CSV_PATH,
    COMPILED_ANALYTICS_JSON,
)


# -------------------------------------------------------------
# get_best_worst: returns the best and worst run for a given
# metric field, along with the chunking strategy and value
# -------------------------------------------------------------
def get_best_worst(df, field, higher_is_better=True):
    best_row  = df.loc[df[field].idxmax() if higher_is_better else df[field].idxmin()]
    worst_row = df.loc[df[field].idxmin() if higher_is_better else df[field].idxmax()]
    return {
        "best":  {"strategy": best_row["chunking"],  "value": round(float(best_row[field]),  4)},
        "worst": {"strategy": worst_row["chunking"], "value": round(float(worst_row[field]), 4)},
    }


if __name__ == "__main__":

    files = sorted(f for f in os.listdir(RUNS_DIR) if f.endswith(".json"))
    print(f"\n[compile] Found {len(files)} experiment files")

    # -------------------------------------------------------------
    # load each run file and flatten config, metrics, latency,
    # and token usage into a single row
    # -------------------------------------------------------------
    rows = []
    for file_name in files:
        with open(os.path.join(RUNS_DIR, file_name), "r") as f:
            data = json.load(f)

        config      = data.get("config", {})
        summary     = data.get("summary", {})
        latency     = data.get("latency", {})
        overall_tok = data.get("token_usage", {}).get("overall", {})

        row = {
            "run_file":            file_name,
            "chunking":            config.get("chunking"),
            "retrieval":           config.get("retrieval"),
            "reranker_enabled":    config.get("reranker", {}).get("enabled"),
            "reranker_type":       config.get("reranker", {}).get("type"),
            "answer_correctness":  summary.get("answer_correctness"),
            "answer_similarity":   summary.get("answer_similarity"),
            "faithfulness":        summary.get("faithfulness"),
            "answer_relevancy":    summary.get("answer_relevancy"),
            "context_precision":   summary.get("context_precision"),
            "context_recall":      summary.get("context_recall"),
            "ingestion_latency":   latency.get("ingestion"),
            "preprocessing_latency": latency.get("preprocessing"),
            "chunking_latency":    latency.get("chunking"),
            "vector_store_latency": latency.get("vector_store"),
            "rag_latency":         latency.get("rag"),
            "evaluation_latency":  latency.get("evaluation"),
            "prompt_tokens":       overall_tok.get("prompt_tokens"),
            "completion_tokens":   overall_tok.get("completion_tokens"),
            "total_tokens":        overall_tok.get("total_tokens"),
            "total_cost_usd":      overall_tok.get("total_cost_usd"),
        }

        # overall_score averages the three most user-facing quality metrics
        row["overall_score"] = round(
            (row["answer_correctness"] + row["faithfulness"] + row["answer_relevancy"]) / 3, 4
        )

        rows.append(row)

    # -------------------------------------------------------------
    # save CSV
    # -------------------------------------------------------------
    df = pd.DataFrame(rows)
    os.makedirs(COMPILED_RESULTS_DIR, exist_ok=True)
    df.to_csv(COMPILED_CSV_PATH, index=False)
    print(f"[compile] Saved CSV → {COMPILED_CSV_PATH}")

    # -------------------------------------------------------------
    # save analytics JSON
    # -------------------------------------------------------------
    analytics = {
        "num_experiments":   len(df),
        "overall_score":     get_best_worst(df, "overall_score"),
        "answer_correctness": get_best_worst(df, "answer_correctness"),
        "answer_similarity": get_best_worst(df, "answer_similarity"),
        "faithfulness":      get_best_worst(df, "faithfulness"),
        "answer_relevancy":  get_best_worst(df, "answer_relevancy"),
        "context_precision": get_best_worst(df, "context_precision"),
        "context_recall":    get_best_worst(df, "context_recall"),
        "rag_latency":       get_best_worst(df, "rag_latency",       higher_is_better=False),
        "evaluation_latency": get_best_worst(df, "evaluation_latency", higher_is_better=False),
        "total_tokens":      get_best_worst(df, "total_tokens",      higher_is_better=False),
        "total_cost_usd":    get_best_worst(df, "total_cost_usd",    higher_is_better=False),
    }

    with open(COMPILED_ANALYTICS_JSON, "w") as f:
        json.dump(analytics, f, indent=2)
    print(f"[compile] Saved analytics JSON → {COMPILED_ANALYTICS_JSON}")

    # -------------------------------------------------------------
    # preview
    # -------------------------------------------------------------
    print("\n========== Experiment Summary ==========")
    print(df[["chunking", "retrieval", "overall_score", "total_tokens", "rag_latency", "evaluation_latency"]])

    print("\n========== Analytics ==========")
    print(json.dumps(analytics, indent=2))
