"""
generate_test_set.py
--------------------
Topic-controlled Q&A benchmark generation.

python generate_test_set.py \
        --source data/papers/MizoPRS.pdf  \
        --dest data/eval/qna_dataset.json \
        --qna_topic "dataset"  \
        --size 10
"""

import json
import argparse
from json_repair import repair_json

from src.ingestion.loader import load_pdf
from src.ingestion.preprocessor import preprocess_documents
from src.generation.llm import get_openai_llm


# ── Helpers ────────────────────────────────────────────

def build_full_text(documents):
    text = "\n\n".join([doc.page_content for doc in documents])
    return text[:15000]


def generate_qna(llm, text, topic, num_questions, existing_questions=None):

    existing_text = ""
    if existing_questions:
        existing_text = "\n".join(existing_questions[:20])

    prompt = f"""
You are creating a high-quality Q&A benchmark dataset.

Generate {num_questions} UNIQUE question-answer pairs STRICTLY about: {topic}

STRICT RULE:
- Do NOT repeat or rephrase these questions:
{existing_text}

Topic guidance:
- Focus ONLY on: {topic}
- Ignore unrelated parts of the text

Requirements:
- Questions must be clear and standalone
- Answers MUST be grounded in the text
- Avoid repetition

STRICT JSON ONLY.

Format:
[
  {{
    "question": "string",
    "answer": "string",
    "type": "{topic}"
  }}
]

TEXT:
\"\"\"{text}\"\"\"
"""

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else response


def parse_and_clean(qna_str):
    try:
        data = json.loads(qna_str)
    except Exception:
        print("[testset] JSON invalid → attempting repair...")
        try:
            repaired = repair_json(qna_str)
            data = json.loads(repaired)
            print("[testset] ✅ JSON repaired")
        except Exception:
            print("[testset] ❌ Failed → skipping batch")
            return []

    cleaned = []
    for d in data:
        if not isinstance(d, dict):
            continue
        if not d.get("question") or not d.get("answer"):
            continue
        if len(d["question"].strip()) < 5:
            continue
        cleaned.append(d)

    return cleaned


def generate_qna_batched(llm, text, topic, total_size, batch_size=5):

    all_data = []
    seen_questions = set()

    num_batches = (total_size + batch_size - 1) // batch_size

    for i in range(num_batches):
        print(f"[testset] Batch {i+1}/{num_batches}")

        qna_str = generate_qna(
            llm,
            text,
            topic,
            batch_size,
            existing_questions=list(seen_questions)
        )

        batch_data = parse_and_clean(qna_str)

        for item in batch_data:
            q = item["question"].strip().lower()

            if q in seen_questions:
                continue

            seen_questions.add(q)
            all_data.append(item)

            if len(all_data) >= total_size:
                break

        if len(all_data) >= total_size:
            break

    return all_data[:total_size]


def add_serial_numbers(data):
    for i, d in enumerate(data):
        d["id"] = i + 1
    return data


def save_output(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[testset] Saved {len(data)} Q&A pairs → {path}")


# ── Main ───────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--qna_topic", required=True)
    parser.add_argument("--size", type=int, default=10)

    args = parser.parse_args()

    print("\n========================================")
    print("STEP 1 : LOAD + PREPROCESS")
    print("========================================")

    documents = load_pdf(args.source)
    documents = preprocess_documents(documents)

    print(f"[testset] Documents: {len(documents)}")

    print("\n========================================")
    print("STEP 2 : GENERATE QNA (TOPIC-BASED)")
    print("========================================")

    full_text = build_full_text(documents)
    llm = get_openai_llm()

    qna_data = generate_qna_batched(
        llm,
        full_text,
        args.qna_topic,
        args.size
    )

    qna_data = add_serial_numbers(qna_data)

    print("\n--- Sample ---")
    print(json.dumps(qna_data[:2], indent=2))

    print("\n========================================")
    print("STEP 3 : SAVE")
    print("========================================")

    save_output(qna_data, args.dest)

    print("\n========================================")
    print("Done!")
    print("========================================")
