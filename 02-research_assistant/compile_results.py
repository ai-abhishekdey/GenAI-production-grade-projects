# compile_results.py

"""
compile_results.py
------------------
Reads all experiment JSON files and creates:

1. comparison CSV
2. analytics summary JSON

The analytics JSON is later used by
experiment_summariser.py
"""

import os
import json
import pandas as pd

from config import (
    RUNS_DIR,
    COMPILED_RESULTS_DIR,
    COMPILED_CSV_PATH,
    COMPILED_ANALYTICS_JSON
)

# ── Helper: Extract Best/Worst ────────────────────────


def get_best_worst(df, field, higher_is_better=True):

    if higher_is_better:

        best_row = df.loc[df[field].idxmax()]
        worst_row = df.loc[df[field].idxmin()]

    else:

        best_row = df.loc[df[field].idxmin()]
        worst_row = df.loc[df[field].idxmax()]

    return {

        "best": {
            "strategy": best_row["chunking"],
            "value": round(float(best_row[field]), 4)
        },

        "worst": {
            "strategy": worst_row["chunking"],
            "value": round(float(worst_row[field]), 4)
        }
    }


# ── Main ──────────────────────────────────────────────

if __name__ == "__main__":

    rows = []

    files = sorted(
        [
            f for f in os.listdir(RUNS_DIR)
            if f.endswith(".json")
        ]
    )

    print(f"\n[compile] Found {len(files)} experiment files")

    # =================================================
    # LOAD RESULTS
    # =================================================

    for file_name in files:

        file_path = os.path.join(
            RUNS_DIR,
            file_name
        )

        with open(file_path, "r") as f:

            data = json.load(f)

        config = data.get("config", {})
        summary = data.get("summary", {})
        latency = data.get("latency", {})
        token_usage = data.get("token_usage", {})

        overall_tokens = token_usage.get(
            "overall",
            {}
        )

        row = {

            # ── File ────────────────────────────────

            "run_file": file_name,

            # ── Config ──────────────────────────────

            "chunking": config.get("chunking"),

            "retrieval": config.get("retrieval"),

            "reranker_enabled":
                config.get("reranker", {}).get("enabled"),

            "reranker_type":
                config.get("reranker", {}).get("type"),

            # ── Metrics ─────────────────────────────

            "answer_correctness":
                summary.get("answer_correctness"),

            "answer_similarity":
                summary.get("answer_similarity"),

            "faithfulness":
                summary.get("faithfulness"),

            "answer_relevancy":
                summary.get("answer_relevancy"),

            "context_precision":
                summary.get("context_precision"),

            "context_recall":
                summary.get("context_recall"),

            # ── Latency ────────────────────────────

            "ingestion_latency":
                latency.get("ingestion"),

            "preprocessing_latency":
                latency.get("preprocessing"),

            "chunking_latency":
                latency.get("chunking"),

            "vector_store_latency":
                latency.get("vector_store"),

            "rag_latency":
                latency.get("rag"),

            "evaluation_latency":
                latency.get("evaluation"),

            # ── Token Usage ────────────────────────

            "prompt_tokens":
                overall_tokens.get("prompt_tokens"),

            "completion_tokens":
                overall_tokens.get("completion_tokens"),

            "total_tokens":
                overall_tokens.get("total_tokens"),

            "total_cost_usd":
                overall_tokens.get("total_cost_usd"),
        }

        # ── Derived Overall Score ────────────────────

        row["overall_score"] = round(
            (
                row["answer_correctness"] +
                row["faithfulness"] +
                row["answer_relevancy"]
            ) / 3,
            4
        )

        rows.append(row)

    # =================================================
    # CREATE DATAFRAME
    # =================================================

    df = pd.DataFrame(rows)

    os.makedirs(
        COMPILED_RESULTS_DIR,
        exist_ok=True
    )

    # =================================================
    # SAVE CSV
    # =================================================

    df.to_csv(
        COMPILED_CSV_PATH,
        index=False
    )

    print(f"\n[compile] Saved CSV → {COMPILED_CSV_PATH}")

    # =================================================
    # ANALYTICS JSON
    # =================================================

    analytics = {

        "num_experiments": len(df),

        "overall_score":
            get_best_worst(
                df,
                "overall_score",
                higher_is_better=True
        ),

        "answer_correctness":
            get_best_worst(
                df,
                "answer_correctness",
                higher_is_better=True
        ),

        "answer_similarity":
            get_best_worst(
                df,
                "answer_similarity",
                higher_is_better=True
        ),

        "faithfulness":
            get_best_worst(
                df,
                "faithfulness",
                higher_is_better=True
        ),

        "answer_relevancy":
            get_best_worst(
                df,
                "answer_relevancy",
                higher_is_better=True
        ),

        "context_precision":
            get_best_worst(
                df,
                "context_precision",
                higher_is_better=True
        ),

        "context_recall":
            get_best_worst(
                df,
                "context_recall",
                higher_is_better=True
        ),

        "rag_latency":
            get_best_worst(
                df,
                "rag_latency",
                higher_is_better=False
        ),

        "evaluation_latency":
            get_best_worst(
                df,
                "evaluation_latency",
                higher_is_better=False
        ),

        "total_tokens":
            get_best_worst(
                df,
                "total_tokens",
                higher_is_better=False
        ),

        "total_cost_usd":
            get_best_worst(
                df,
                "total_cost_usd",
                higher_is_better=False
        ),
    }

    # =================================================
    # SAVE ANALYTICS JSON
    # =================================================

    with open(COMPILED_ANALYTICS_JSON, "w") as f:

        json.dump(
            analytics,
            f,
            indent=2
        )

    print(
        f"\n[compile] Saved analytics JSON → "
        f"{COMPILED_ANALYTICS_JSON}"
    )

    # =================================================
    # Preview
    # =================================================

    print("\n========== Experiment Summary ==========")

    print(df[
        [
            "chunking",
            "retrieval",
            "overall_score",
            "total_tokens",
            "rag_latency",
            "evaluation_latency"
        ]
    ])

    print("\n========== Analytics ==========")

    print(json.dumps(
        analytics,
        indent=2
    ))
