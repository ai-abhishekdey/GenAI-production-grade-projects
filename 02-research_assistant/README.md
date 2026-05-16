# Research Assistant

A configurable RAG (Retrieval-Augmented Generation) pipeline for question answering over research papers. Designed for systematic experimentation across chunking strategies, retrieval methods, rerankers, and prompts — with built-in evaluation, latency tracking, and cost reporting.

## Features

- **Ingestion**: Load PDFs from a local file, folder, or URL (including arxiv)
- **Chunking**: Fixed-size, recursive, page-level, semantic, layout-aware (via LandingAI ADE), or hybrid
- **Vector stores**: Chroma (local) or Qdrant (hosted)
- **Retrieval**: Dense, MMR, sparse (BM25), or hybrid
- **Reranking**: FlashRank or CrossEncoder
- **Evaluation**: DeepEval and RAGAS metrics (correctness, faithfulness, relevancy, precision, recall)
- **Observability**: Per-step latency tracking, token usage, and cost estimation
- **Reporting**: CSV summary, analytics JSON, and PDF report with plots

## Project Structure

```
research_assistant/
├── config.py                   # All pipeline settings
├── experiment.py               # Main pipeline runner
├── generate_test_set.py        # Q&A benchmark generation from PDFs
├── compile_results.py          # Aggregate experiment runs into CSV/JSON
├── generate_report.py          # Generate PDF report with plots
├── src/
│   ├── ingestion/              # PDF loading, preprocessing, chunking
│   ├── indexing/               # Embeddings and vector store setup
│   ├── retrieval/              # Retriever implementations
│   ├── generation/             # LLM, prompts, RAG chain, guardrails
│   ├── evaluation/             # DeepEval and RAGAS evaluation
│   └── observability/          # Latency, token, and logging utilities
├── data/
│   ├── papers/                 # Input PDFs
│   ├── eval/                   # Q&A test sets (JSON)
│   └── ade_outputs/            # Layout-aware chunking outputs
└── experimental_results/
    ├── runs/                   # Per-run JSON results
    ├── logs/                   # Run logs
    └── compiled_results/       # Aggregated CSV, analytics, and plots
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
OPENAI_API_KEY=...
VISION_AGENT_API_KEY=...   # Required for layout/hybrid chunking
QDRANT_URL=...             # Required if using Qdrant
QDRANT_API_KEY=...         # Required if using Qdrant
```

## Usage

### 1. Generate a test set

```bash
python generate_test_set.py \
    --source data/papers/MizoPRS.pdf \
    --dest data/eval/qna_dataset.json \
    --qna_topic "dataset" \
    --size 10
```

### 2. Configure the pipeline

Edit [config.py](config.py) to set your desired pipeline configuration:

```python
SOURCE_TYPE = "local"           # "local" | "folder" | "url"
CHUNK_STRATEGY = "recursive"    # "fixed" | "recursive" | "page" | "semantic" | "layout" | "hybrid"
VECTOR_DB = "chroma"            # "chroma" | "qdrant"
RETRIEVAL_STRATEGY = "dense"    # "dense" | "mmr" | "sparse" | "hybrid"
USE_RERANKER = False
TESTSET_PATH = "data/eval/test_5qna.json"
```

### 3. Run an experiment

```bash
python experiment.py
```

Results are saved to `experimental_results/runs/` as timestamped JSON files.

### 4. Compile results across runs

```bash
python compile_results.py
```

Outputs a summary CSV and analytics JSON to `experimental_results/compiled_results/`.

### 5. Generate a report

```bash
python generate_report.py
```

Produces a PDF report with leaderboard, metric comparisons, latency, and cost plots.

## Evaluation Metrics

Metrics are computed via DeepEval (default) or RAGAS:

| Metric | Description |
|---|---|
| Answer Correctness | Factual accuracy vs. ground truth |
| Answer Similarity | Semantic similarity to ground truth |
| Faithfulness | Answer grounded in retrieved context |
| Answer Relevancy | Answer addresses the question |
| Context Precision | Retrieved context quality |
| Context Recall | Retrieved context coverage |
