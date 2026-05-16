"""
save_results.py
---------------
Handles formatting and saving experiment outputs
"""

import os
import json
from datetime import datetime


def save_experiment_results(
    results,
    test_data,
    predictions,
    config,
    latency,
    token_usage,
    runs_dir,
    run_id=None
):
    """
    Save experiment results to disk

    Args:
        results: output from eval.py
        test_data: original test.json
        predictions: list of model outputs
        config: dict of experiment config
        latency: latency dictionary
        token_usage: token usage dictionary
        results_dir: path to save results
    """

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_results = {
        "config": config,
        "latency": latency,
        "token_usage": token_usage,
        "summary": results["summary"],
        "results": []
    }

    for i, qa in enumerate(test_data):

        row = {
            "id": qa["id"],
            "question": qa["question"],
            "type": qa.get("type", "unknown"),
            "ground_truth": qa["answer"],
            "prediction": predictions[i],
            "metrics": {}
        }

        for metric_name in results["summary"].keys():

            row["metrics"][metric_name] = \
                results["per_sample"][i][metric_name]

        final_results["results"].append(row)

    os.makedirs(runs_dir, exist_ok=True)

    output_path = os.path.join(
        runs_dir,
        f"run_{run_id}.json"
    )

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n[Saved Results] → {output_path}")

    return output_path
