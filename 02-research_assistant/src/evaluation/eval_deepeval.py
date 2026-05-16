"""
eval_deepeval.py
----------------
RAG evaluation using DeepEval framework.
Drop-in replacement for eval.py — same function names, same return format.

Metrics (matching RAGAS eval.py):
    - answer_correctness  → GEval (custom criteria)
    - answer_similarity   → GEval (semantic similarity)
    - faithfulness        → FaithfulnessMetric
    - answer_relevancy    → AnswerRelevancyMetric
    - context_precision   → ContextualPrecisionMetric
    - context_recall      → ContextualRecallMetric

Install:
    pip install deepeval
"""

import os
import sys
import numpy as np
from typing import List, Dict

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval,
)

from config import EVAL_LLM

# ── Metric Name Mapping ────────────────────────────────────────────────────────
# Maps DeepEval's internal metric names to our standard keys

NAME_MAP = {
    "answer_correctness [geval]": "answer_correctness",
    "answer_similarity [geval]":  "answer_similarity",
    "faithfulness":               "faithfulness",
    "answer relevancy":           "answer_relevancy",
    "contextual precision":       "context_precision",
    "contextual recall":          "context_recall",
}

# ── Setup Metrics Once ─────────────────────────────────────────────────────────

answer_correctness_metric = GEval(
    name="answer_correctness",
    criteria="""Determine if the actual output is factually correct based on the expected output.
    Pay careful attention to:
    - Numerical values (counts, durations, percentages)
    - Mathematical equivalences (e.g. '31 male and 31 female' is equivalent to 'evenly split among 62')
    - Do NOT penalise semantically equivalent expressions of the same fact.""",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=EVAL_LLM,
    threshold=0.5,
    verbose_mode=False,
)


answer_similarity_metric = GEval(
    name="answer_similarity",
    criteria="""Determine if the actual output is semantically similar to the expected output.
    Consider the following:
    - Treat numerically equivalent expressions as similar (e.g. '62 speakers evenly split' ≈ '31 male, 31 female')
    - Paraphrased sentences with same meaning should score high
    - Minor differences in phrasing should not significantly lower the score.""",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=EVAL_LLM,
    threshold=0.5,
    verbose_mode=False,
)

faithfulness_metric = FaithfulnessMetric(
    model=EVAL_LLM,
    threshold=0.5,
    async_mode=True,
    verbose_mode=False,
)

answer_relevancy_metric = AnswerRelevancyMetric(
    model=EVAL_LLM,
    threshold=0.5,
    async_mode=True,
    verbose_mode=False,
)

context_precision_metric = ContextualPrecisionMetric(
    model=EVAL_LLM,
    threshold=0.5,
    async_mode=True,
    verbose_mode=False,
)

context_recall_metric = ContextualRecallMetric(
    model=EVAL_LLM,
    threshold=0.5,
    async_mode=True,
    verbose_mode=False,
)

metrics = [
    answer_correctness_metric,
    answer_similarity_metric,
    faithfulness_metric,
    answer_relevancy_metric,
    context_precision_metric,
    context_recall_metric,
]


# ── Build Test Cases ───────────────────────────────────────────────────────────

def build_test_cases(
    questions: List[str],
    predictions: List[str],
    ground_truths: List[str],
    contexts: List[List[str]]
) -> List[LLMTestCase]:
    """
    Converts raw Q&A data into DeepEval LLMTestCase objects.
    """
    test_cases = []

    for question, prediction, ground_truth, context in zip(
        questions, predictions, ground_truths, contexts
    ):
        test_cases.append(LLMTestCase(
            input=question,
            actual_output=prediction,
            expected_output=ground_truth,
            retrieval_context=context
        ))

    return test_cases


# ── Public API — same signature as eval.py ─────────────────────────────────────

def evaluate_predictions(
    questions: List[str],
    predictions: List[str],
    ground_truths: List[str],
    contexts: List[List[str]],
) -> Dict:
    """
    Main entry point — same function name and return format as eval.py.
    Drop-in replacement: swap import to use DeepEval instead of RAGAS.

    Args:
        questions:     List of questions from benchmark
        predictions:   List of RAG generated answers
        ground_truths: List of ground truth answers from benchmark
        contexts:      List of retrieved context lists per question

    Returns:
        Dict with 'per_sample' scores+reasons and 'summary' averages
    """
    print(f"[eval:deepeval] Evaluating {len(questions)} samples...")

    test_cases = build_test_cases(
        questions, predictions, ground_truths, contexts)

    # Suppress DeepEval's built-in console output
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            results = evaluate(
                test_cases=test_cases,
                metrics=metrics,
            )
        finally:
            sys.stdout = old_stdout

    # ── Build lookup by input question to handle async reordering ─────
    result_lookup = {}
    for test_result in results.test_results:
        result_lookup[test_result.input] = test_result

    # ── Extract per-sample scores in original input order ─────────────
    per_sample = []

    for test_case in test_cases:
        test_result = result_lookup.get(test_case.input)
        sample_scores = {}

        if test_result:
            for metric_data in test_result.metrics_data:
                normalized = NAME_MAP.get(
                    metric_data.name.lower(),
                    metric_data.name.lower()
                )
                sample_scores[normalized] = {
                    "score":  metric_data.score if metric_data.score is not None else 0.0,
                    "reason": metric_data.reason if metric_data.reason is not None else ""
                }

        per_sample.append(sample_scores)

    # ── Aggregate summary — scores only ───────────────────────────────
    all_keys = set()
    for sample in per_sample:
        all_keys.update(sample.keys())

    summary = {
        key: float(np.mean([
            sample.get(key, {}).get("score", 0.0)
            for sample in per_sample
        ]))
        for key in all_keys
    }

    print("\n[eval:deepeval] Summary Scores:")
    for metric, score in summary.items():
        print(f"  {metric:25} : {score:.4f}")

    return {
        "per_sample": per_sample,
        "summary":    summary,
    }
