"""
save_results.py
---------------
Formats and saves experiment outputs to disk as a JSON run file.
"""

import os
import json
from datetime import datetime


# -------------------------------------------------------------
# save_experiment_results: assembles all experiment data into
# a single JSON file named run_<run_id>.json.
# run_id is passed in from experiment.py so the filename matches
# the corresponding log file.
# -------------------------------------------------------------
def save_experiment_results(results, test_data, predictions, config, latency, token_usage, runs_dir, run_id=None):
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_results = {
        "config":      config,
        "latency":     latency,
        "token_usage": token_usage,
        "summary":     results["summary"],
        "results":     []
    }

    for i, qa in enumerate(test_data):
        row = {
            "id":           qa["id"],
            "question":     qa["question"],
            "type":         qa.get("type", "unknown"),
            "ground_truth": qa["answer"],
            "prediction":   predictions[i],
            "metrics":      {metric: results["per_sample"][i][metric] for metric in results["summary"]}
        }
        final_results["results"].append(row)

    os.makedirs(runs_dir, exist_ok=True)
    output_path = os.path.join(runs_dir, f"run_{run_id}.json")

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n[save_results] Saved → {output_path}")
    return output_path
