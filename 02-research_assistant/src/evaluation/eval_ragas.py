"""
eval_ragas.py
-------------
RAG evaluation using RAGAS async scorers.
Runs all six metrics concurrently with bounded concurrency and timeout protection.
"""

import asyncio
import numpy as np
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
from config import LLM, TEXT_EMBEDDING_MODEL, EVAL_CONCURRENCY, METRIC_TIMEOUT


# shared client and scorers — initialised once to avoid redundant API connections
client = AsyncOpenAI()
llm = llm_factory(LLM, client=client)
embeddings = embedding_factory("openai", model=TEXT_EMBEDDING_MODEL, client=client)

scorers = {
    "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
    "answer_similarity":  SemanticSimilarity(embeddings=embeddings),
    "faithfulness":       Faithfulness(llm=llm),
    "answer_relevancy":   AnswerRelevancy(llm=llm, embeddings=embeddings),
    "context_precision":  ContextPrecision(llm=llm),
    "context_recall":     ContextRecall(llm=llm),
}


# -------------------------------------------------------------
# safe_ascore: wraps a single metric coroutine with a timeout
# and returns 0.0 on failure so one bad metric doesn't abort the run
# -------------------------------------------------------------
async def safe_ascore(metric_name, coro):
    try:
        result = await asyncio.wait_for(coro, timeout=METRIC_TIMEOUT)
        return result.value
    except asyncio.TimeoutError:
        print(f"[eval:ragas] Timeout: {metric_name}")
        return 0.0
    except Exception as e:
        print(f"[eval:ragas] Error in {metric_name}: {e}")
        return 0.0


# -------------------------------------------------------------
# evaluate_one: runs all six metrics for a single Q&A sample
# -------------------------------------------------------------
async def evaluate_one(idx, question, prediction, ground_truth, contexts):
    print(f"\n[eval:ragas] Running sample {idx + 1}")

    results = {
        "answer_correctness": await safe_ascore("answer_correctness",
            scorers["answer_correctness"].ascore(user_input=question, response=prediction, reference=ground_truth)),

        "answer_similarity": await safe_ascore("answer_similarity",
            scorers["answer_similarity"].ascore(reference=ground_truth, response=prediction)),

        "faithfulness": await safe_ascore("faithfulness",
            scorers["faithfulness"].ascore(user_input=question, response=prediction, retrieved_contexts=contexts)),

        "answer_relevancy": await safe_ascore("answer_relevancy",
            scorers["answer_relevancy"].ascore(user_input=question, response=prediction)),

        "context_precision": await safe_ascore("context_precision",
            scorers["context_precision"].ascore(user_input=question, reference=ground_truth, retrieved_contexts=contexts)),

        "context_recall": await safe_ascore("context_recall",
            scorers["context_recall"].ascore(user_input=question, reference=ground_truth, retrieved_contexts=contexts)),
    }

    print(f"[eval:ragas] Sample {idx + 1} completed")
    return results


# -------------------------------------------------------------
# evaluate_all: runs evaluate_one for all samples with a
# semaphore to cap concurrent API calls at EVAL_CONCURRENCY
# -------------------------------------------------------------
async def evaluate_all(questions, predictions, ground_truths, contexts):
    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)

    async def sem_task(idx, q, p, gt, ctx):
        async with semaphore:
            return await evaluate_one(idx, q, p, gt, ctx)

    tasks = [
        sem_task(idx, q, p, gt, ctx)
        for idx, (q, p, gt, ctx) in enumerate(zip(questions, predictions, ground_truths, contexts))
    ]

    return await asyncio.gather(*tasks)


# -------------------------------------------------------------
# evaluate_predictions: public entry point — runs evaluation
# and returns per-sample scores plus aggregated summary
# -------------------------------------------------------------
def evaluate_predictions(questions, predictions, ground_truths, contexts):
    per_sample = asyncio.run(evaluate_all(questions, predictions, ground_truths, contexts))

    summary = {
        key: float(np.mean([x[key] for x in per_sample]))
        for key in per_sample[0].keys()
    }

    return {"per_sample": per_sample, "summary": summary}
