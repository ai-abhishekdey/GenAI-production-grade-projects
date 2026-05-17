import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ────────────────────────────────────────────

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
ADE_API_KEY     = os.getenv("VISION_AGENT_API_KEY")

# LangSmith — set LANGCHAIN_TRACING_V2=true in .env to enable
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "research-assistant")

# ── Models ──────────────────────────────────────────────

TEXT_EMBEDDING_MODEL = "text-embedding-3-small"

LLM = "gpt-4.1-nano"
EVAL_LLM = "gpt-4o-mini"    # faster for evaluation, separate from pipeline LLM


# ── Data Sources ────────────────────────────────────────

SOURCE_TYPE = "local"   # "local" | "folder" | "url"

LOCAL_PDF_PATH = "data/papers/MizoPRS.pdf"
LOCAL_PDF_FOLDER = "data/papers/"
ARXIV_URL = "https://arxiv.org/pdf/2409.10545"


# ── Chunking ────────────────────────────────────────────

# best strategy from experiments — layout-aware chunking via LandingAI ADE
CHUNK_STRATEGY = "layout"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
ADE_OUTPUT_DIR = "data/ade_outputs"

# ── Vector Store ────────────────────────────────────────

VECTOR_DB = "qdrant"   # "chroma" | "qdrant"

# Chroma
CHROMA_DB_DIR = "vector_db/chroma_db"

# Qdrant

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "research_assistant"

# ── Retrieval Config ─────────────────────────

RETRIEVAL_STRATEGY = "dense"  # "dense" | "mmr" | "sparse" | "hybrid"

TOP_K = 5

# for MMR
MMR_LAMBDA = 0.5

# for hybrid weighting (future use)
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5

# ── Reranker Config ─────────────────────────

USE_RERANKER = False
RERANKER_TYPE = "crossencoder"    # "flashrank" or "crossencoder"
RERANKER_TOP_N = 3
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Prompt Config ─────────────────────────

PROMPT_TYPE = "basic"   # "basic" | "strict" | "debug"

# ── Guardrails Config ─────────────────────────

ENABLE_INPUT_GUARDRAILS = False
ENABLE_OUTPUT_GUARDRAILS = False

# ── Evaluation Paths ───────────────────────────────────

TESTSET_PATH = "data/eval/test_5qna.json"
RESULTS_DIR = "experimental_results"
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")


# ── Experiment Compilation ───────────────────────────

COMPILED_RESULTS_DIR = os.path.join(RESULTS_DIR, "compiled_results")

COMPILED_CSV_PATH = os.path.join(
    COMPILED_RESULTS_DIR, "experiment_summary.csv")

COMPILED_ANALYTICS_JSON = os.path.join(
    COMPILED_RESULTS_DIR, "experiment_analytics.json")

EXPERIMENT_SUMMARY_PDF = os.path.join(
    COMPILED_RESULTS_DIR, "experiment_summary.pdf")
