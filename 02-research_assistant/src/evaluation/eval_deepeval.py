"""
eval_deepeval.py
----------------
RAG evaluation using the DeepEval framework.
Drop-in replacement for eval_ragas.py — same function signature, same return format.

Metrics:
    answer_correctness  → GEval (custom criteria)
    answer_similarity   → GEval (semantic similarity)
    faithfulness        → FaithfulnessMetric
    answer_relevancy    → AnswerRelevancyMetric
    context_precision   → ContextualPrecisionMetric
    context_recall      → ContextualRecallMetric
"""

import numpy as np

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval,
)
from config import EVAL_LLM


# maps DeepEval's internal metric names to our standard keys
NAME_MAP = {
    "answer_correctness [geval]": "answer_correctness",
    "answer_similarity [geval]":  "answer_similarity",
    "faithfulness":               "faithfulness",
    "answer relevancy":           "answer_relevancy",
    "contextual precision":       "context_precision",
    "contextual recall":          "context_recall",
}

# metrics are initialised once at module load to avoid redundant setup
metrics = [
    GEval(
        name="answer_correctness",
        criteria="""Determine if the actual output is factually correct based on the expected output.
    Pay careful attention to:
    - Numerical values (counts, durations, percentages)
    - Mathematical equivalences (e.g. '31 male and 31 female' is equivalent to 'evenly split among 62')
    - Do NOT penalise semantically equivalent expressions of the same fact.""",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                           LLMTestCaseParams.EXPECTED_OUTPUT],
        model=EVAL_LLM, threshold=0.5, verbose_mode=False,
    ),
    GEval(
        name="answer_similarity",
        criteria="""Determine if the actual output is semantically similar to the expected output.
    Consider the following:
    - Treat numerically equivalent expressions as similar (e.g. '62 speakers evenly split' ≈ '31 male, 31 female')
    - Paraphrased sentences with same meaning should score high
    - Minor differences in phrasing should not significantly lower the score.""",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                           LLMTestCaseParams.EXPECTED_OUTPUT],
        model=EVAL_LLM, threshold=0.5, verbose_mode=False,
    ),
    FaithfulnessMetric(model=EVAL_LLM, threshold=0.5,
                       async_mode=True, verbose_mode=False),
    AnswerRelevancyMetric(model=EVAL_LLM, threshold=0.5,
                          async_mode=True, verbose_mode=False),
    ContextualPrecisionMetric(
        model=EVAL_LLM, threshold=0.5, async_mode=True, verbose_mode=False),
    ContextualRecallMetric(model=EVAL_LLM, threshold=0.5,
                           async_mode=True, verbose_mode=False),
]


# -------------------------------------------------------------
# build_test_cases: converts raw Q&A lists into DeepEval
# LLMTestCase objects expected by the evaluate() function
# -------------------------------------------------------------
def build_test_cases(questions, predictions, ground_truths, contexts):
    return [
        LLMTestCase(
            input=question,
            actual_output=prediction,
            expected_output=ground_truth,
            retrieval_context=context
        )
        for question, prediction, ground_truth, context in zip(questions, predictions, ground_truths, contexts)
    ]


# -------------------------------------------------------------
# evaluate_predictions: public entry point — runs all metrics
# and returns per-sample scores plus aggregated summary.
# DeepEval's console output is suppressed to keep logs clean.
# -------------------------------------------------------------
def evaluate_predictions(questions, predictions, ground_truths, contexts):
    print(f"[eval:deepeval] Evaluating {len(questions)} samples...")

    test_cases = build_test_cases(
        questions, predictions, ground_truths, contexts)

    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=True, max_concurrent=3, throttle_value=0.5),
        display_config=DisplayConfig(
            show_indicator=False, print_results=False),
        error_config=ErrorConfig(ignore_errors=True),
    )

    # build a lookup by input question to handle async result reordering
    result_lookup = {r.input: r for r in results.test_results}

    per_sample = []
    for test_case in test_cases:
        test_result = result_lookup.get(test_case.input)
        sample_scores = {}

        if test_result:
            for metric_data in test_result.metrics_data:
                key = NAME_MAP.get(metric_data.name.lower(),
                                   metric_data.name.lower())
                sample_scores[key] = {
                    "score":  metric_data.score if metric_data.score is not None else 0.0,
                    "reason": metric_data.reason if metric_data.reason is not None else ""
                }

        per_sample.append(sample_scores)

    all_keys = set(k for sample in per_sample for k in sample)
    summary = {
        key: float(np.mean([sample.get(key, {}).get("score", 0.0)
                   for sample in per_sample]))
        for key in all_keys
    }

    print("\n[eval:deepeval] Summary Scores:")
    for metric, score in summary.items():
        print(f"  {metric:25} : {score:.4f}")

    return {"per_sample": per_sample, "summary": summary}
