"""
generate_test_set.py
--------------------
Topic-controlled Q&A benchmark generation from a research paper PDF.

Usage:
    python generate_test_set.py \
        --source data/papers/MizoPRS.pdf \
        --dest data/eval/qna_dataset.json \
        --qna_topic "dataset" \
        --size 10
"""

import json
import argparse
from json_repair import repair_json

from src.ingestion.loader import load_pdf
from src.ingestion.preprocessor import preprocess_documents
from src.generation.llm import get_openai_llm


# -------------------------------------------------------------
# build_full_text: joins all page content and caps at 15k chars
# to stay within LLM context limits
# -------------------------------------------------------------
def build_full_text(documents):
    return "\n\n".join(doc.page_content for doc in documents)[:15000]


# -------------------------------------------------------------
# generate_qna: prompts the LLM to generate Q&A pairs for a
# topic, passing already-seen questions to avoid repetition
# -------------------------------------------------------------
def generate_qna(llm, text, topic, num_questions, existing_questions=None):
    existing_text = "\n".join((existing_questions or [])[:20])

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


# -------------------------------------------------------------
# parse_and_clean: parses the LLM JSON response, attempts repair
# if malformed, and filters out incomplete or too-short entries
# -------------------------------------------------------------
def parse_and_clean(qna_str):
    try:
        data = json.loads(qna_str)
    except Exception:
        print("[testset] JSON invalid — attempting repair...")
        try:
            data = json.loads(repair_json(qna_str))
            print("[testset] JSON repaired successfully")
        except Exception:
            print("[testset] Repair failed — skipping batch")
            return []

    return [
        d for d in data
        if isinstance(d, dict)
        and d.get("question")
        and d.get("answer")
        and len(d["question"].strip()) >= 5
    ]


# -------------------------------------------------------------
# generate_qna_batched: generates Q&A in small batches and
# deduplicates across batches until the target size is reached
# -------------------------------------------------------------
def generate_qna_batched(llm, text, topic, total_size, batch_size=5):
    all_data = []
    seen_questions = set()
    num_batches = (total_size + batch_size - 1) // batch_size

    for i in range(num_batches):
        print(f"[testset] Batch {i+1}/{num_batches}")

        batch_data = parse_and_clean(
            generate_qna(llm, text, topic, batch_size, existing_questions=list(seen_questions))
        )

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


# -------------------------------------------------------------
# add_serial_numbers: adds a 1-based id field to each Q&A entry
# -------------------------------------------------------------
def add_serial_numbers(data):
    for i, d in enumerate(data):
        d["id"] = i + 1
    return data


# -------------------------------------------------------------
# save_output: writes the Q&A dataset to a JSON file
# -------------------------------------------------------------
def save_output(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[testset] Saved {len(data)} Q&A pairs → {path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True)
    parser.add_argument("--dest",      required=True)
    parser.add_argument("--qna_topic", required=True)
    parser.add_argument("--size",      type=int, default=10)
    args = parser.parse_args()

    print("\n========================================")
    print("STEP 1 : LOAD + PREPROCESS")
    print("========================================")

    documents = preprocess_documents(load_pdf(args.source))
    print(f"[testset] Loaded {len(documents)} pages")

    print("\n========================================")
    print("STEP 2 : GENERATE QNA (TOPIC-BASED)")
    print("========================================")

    qna_data = generate_qna_batched(
        get_openai_llm(),
        build_full_text(documents),
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
