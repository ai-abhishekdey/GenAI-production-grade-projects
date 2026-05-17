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
├── compile_results.py          # Aggregate experiment runs into CSV/JSON
├── generate_report.py          # Generate PDF report with plots
├── app.py                      # Streamlit frontend
├── api/                        # FastAPI backend
├── src/
│   ├── ingestion/              # PDF loading, preprocessing, chunking
│   ├── indexing/               # Embeddings and vector store setup
│   ├── retrieval/              # Retriever implementations
│   ├── generation/             # LLM, prompts, RAG chain, guardrails
│   ├── evaluation/             # DeepEval and RAGAS evaluation
│   └── observability/          # Latency, token, and logging utilities
├── docker/                     # Dockerfiles and compose files
├── monitoring/                 # Prometheus and Grafana config
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

Create a `.env` file at the project root with your API keys:

```
OPENAI_API_KEY=...
VISION_AGENT_API_KEY=...   # Required for layout/hybrid chunking
QDRANT_URL=...             # Required if using Qdrant
QDRANT_API_KEY=...         # Required if using Qdrant
```

## Running with Docker (recommended)

### 1. Build all images

```bash
docker compose -f docker/docker-compose.yml build
```

### 2. Start all services

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 3. Verify all containers are running

```bash
docker compose -f docker/docker-compose.yml ps
```

Expected output:

```
NAME          SERVICE     STATUS              PORTS
ra-backend    backend     Up (healthy)        0.0.0.0:8000->8000/tcp
ra-frontend   frontend    Up                  0.0.0.0:8501->8501/tcp
ra-prometheus prometheus  Up                  0.0.0.0:9090->9090/tcp
ra-grafana    grafana     Up                  0.0.0.0:3000->3000/tcp
```

### 4. Open the services

| Service | URL |
|---|---|
| UI (Streamlit) | http://localhost:8501 |
| API (FastAPI) | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |
| LangSmith | https://smith.langchain.com/project/rag-research-assistant |
| Qdrant Cloud | https://cloud.qdrant.io |

### Useful commands

Rebuild a single service after code changes:

```bash
docker compose -f docker/docker-compose.yml up -d --build backend
```

Stream logs from a service:

```bash
docker compose -f docker/docker-compose.yml logs -f <service-name>
```

Stop and remove all containers:

```bash
docker compose -f docker/docker-compose.yml down
```

## Running Locally (without Docker)

1. Create and activate a virtual environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start the backend:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

3. Start the frontend (separate terminal):

```bash
streamlit run app.py
```

## Running Experiments

### 1. Configure the pipeline

Edit [config.py](config.py) to set your desired configuration:

```python
SOURCE_TYPE = "local"           # "local" | "folder" | "url"
CHUNK_STRATEGY = "recursive"    # "fixed" | "recursive" | "page" | "semantic" | "layout" | "hybrid"
VECTOR_DB = "chroma"            # "chroma" | "qdrant"
RETRIEVAL_STRATEGY = "dense"    # "dense" | "mmr" | "sparse" | "hybrid"
USE_RERANKER = False
TESTSET_PATH = "data/eval/test_5qna.json"
```

### 2. Run an experiment

```bash
python experiment.py
```

Results are saved to `experimental_results/runs/` as timestamped JSON files.

### 3. Compile results across runs

```bash
python compile_results.py
```

### 4. Generate a report

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

---

## Appendix

### Generating a Test Set

Use `generate_test_set.py` to create a Q&A benchmark from any PDF. The script generates topic-focused question-answer pairs using the LLM and saves them as a JSON file that can be used as `TESTSET_PATH` in experiments.

```bash
python generate_test_set.py \
    --source data/papers/MizoPRS.pdf \
    --dest data/eval/qna_dataset.json \
    --qna_topic "dataset" \
    --size 10
```

| Argument | Description |
|---|---|
| `--source` | Path to the input PDF |
| `--dest` | Output path for the JSON test set |
| `--qna_topic` | Topic to focus questions on (e.g. "results", "methodology") |
| `--size` | Number of Q&A pairs to generate |
