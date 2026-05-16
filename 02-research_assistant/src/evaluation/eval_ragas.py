"""
eval.py
-------
Generic evaluation using latest RAGAS async scorers.

Optimized with:
- bounded concurrency
- timeout protection
- retry-safe execution
- shared scorers
- progress logging
"""

import asyncio
import numpy as np
from typing import List, Dict
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    SemanticSimilarity,
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from config import (
    LLM,
    TEXT_EMBEDDING_MODEL,
    EVAL_CONCURRENCY,
    METRIC_TIMEOUT
)


# ── Setup Once ────────────────────────────────────────

client = AsyncOpenAI()

llm = llm_factory(
    LLM,
    client=client
)

embeddings = embedding_factory(
    "openai",
    model=TEXT_EMBEDDING_MODEL,
    client=client
)

# ── Shared Scorers ────────────────────────────────────

scorers = {

    "answer_correctness": AnswerCorrectness(
        llm=llm,
        embeddings=embeddings
    ),

    "answer_similarity": SemanticSimilarity(
        embeddings=embeddings
    ),

    "faithfulness": Faithfulness(
        llm=llm
    ),

    "answer_relevancy": AnswerRelevancy(
        llm=llm,
        embeddings=embeddings
    ),

    "context_precision": ContextPrecision(
        llm=llm
    ),

    "context_recall": ContextRecall(
        llm=llm
    ),
}


# ── Safe Metric Wrapper ───────────────────────────────

async def safe_ascore(
    metric_name,
    coro
):

    try:

        result = await asyncio.wait_for(
            coro,
            timeout=METRIC_TIMEOUT
        )

        return result.value

    except asyncio.TimeoutError:

        print(f"[eval] Timeout : {metric_name}")

        return 0.0

    except Exception as e:

        print(f"[eval] Error in {metric_name} : {e}")

        return 0.0


# ── Per-sample Evaluation ─────────────────────────────

async def evaluate_one(
    idx: int,
    question: str,
    prediction: str,
    ground_truth: str,
    contexts: List[str]
) -> Dict:

    print(f"\n[eval] Running sample {idx+1}")

    results = {}

    # ── Answer Correctness ────────────────────────────

    print("[eval] Running : answer_correctness")

    results["answer_correctness"] = await safe_ascore(
        "answer_correctness",
        scorers["answer_correctness"].ascore(
            user_input=question,
            response=prediction,
            reference=ground_truth
        )
    )

    # ── Semantic Similarity ───────────────────────────

    print("[eval] Running : answer_similarity")

    results["answer_similarity"] = await safe_ascore(
        "answer_similarity",
        scorers["answer_similarity"].ascore(
            reference=ground_truth,
            response=prediction
        )
    )

    # ── Faithfulness ──────────────────────────────────

    print("[eval] Running : faithfulness")

    results["faithfulness"] = await safe_ascore(
        "faithfulness",
        scorers["faithfulness"].ascore(
            user_input=question,
            response=prediction,
            retrieved_contexts=contexts
        )
    )

    # ── Answer Relevancy ──────────────────────────────

    print("[eval] Running : answer_relevancy")

    results["answer_relevancy"] = await safe_ascore(
        "answer_relevancy",
        scorers["answer_relevancy"].ascore(
            user_input=question,
            response=prediction
        )
    )

    # ── Context Precision ─────────────────────────────

    print("[eval] Running : context_precision")

    results["context_precision"] = await safe_ascore(
        "context_precision",
        scorers["context_precision"].ascore(
            user_input=question,
            reference=ground_truth,
            retrieved_contexts=contexts
        )
    )

    # ── Context Recall ────────────────────────────────

    print("[eval] Running : context_recall")

    results["context_recall"] = await safe_ascore(
        "context_recall",
        scorers["context_recall"].ascore(
            user_input=question,
            reference=ground_truth,
            retrieved_contexts=contexts
        )
    )

    print(f"[eval] Sample {idx+1} completed")

    return results


# ── Batch Evaluation ──────────────────────────────────

async def evaluate_all(
    questions,
    predictions,
    ground_truths,
    contexts,
    max_concurrent=EVAL_CONCURRENCY
):

    semaphore = asyncio.Semaphore(
        max_concurrent
    )

    async def sem_task(
        idx,
        q,
        p,
        gt,
        ctx
    ):

        async with semaphore:

            return await evaluate_one(
                idx,
                q,
                p,
                gt,
                ctx
            )

    tasks = []

    for idx, (q, p, gt, ctx) in enumerate(
        zip(
            questions,
            predictions,
            ground_truths,
            contexts
        )
    ):

        tasks.append(
            sem_task(
                idx,
                q,
                p,
                gt,
                ctx
            )
        )

    return await asyncio.gather(*tasks)


# ── Public API ────────────────────────────────────────

def evaluate_predictions(
    questions: List[str],
    predictions: List[str],
    ground_truths: List[str],
    contexts: List[List[str]],
):

    per_sample = asyncio.run(
        evaluate_all(
            questions,
            predictions,
            ground_truths,
            contexts,
            max_concurrent=EVAL_CONCURRENCY
        )
    )

    # ── Aggregate ─────────────────────────────────────

    summary = {}

    for key in per_sample[0].keys():

        summary[key] = float(
            np.mean(
                [x[key] for x in per_sample]
            )
        )

    return {
        "per_sample": per_sample,
        "summary": summary,
    }
