"""
metrics.py
----------
Custom Prometheus metrics for the Research Assistant API.

Prometheus metric types used here:
  Counter   — monotonically increasing (ingestion runs, chunks indexed)
  Histogram — distribution of values across buckets (latency, chunk count)
  Gauge     — current snapshot that can rise or fall (docs in memory)

These are module-level singletons — importing this module registers the
metrics with the Prometheus client. The /metrics endpoint (wired up in
api/main.py) serialises all registered metrics on each scrape.

Usage in route handlers:
    from src.observability.metrics import (
        INGESTION_COUNTER, INGESTION_DURATION, INGESTION_CHUNKS,
        QUERY_LATENCY, QUERY_COUNTER, DOCS_IN_MEMORY,
    )

    INGESTION_COUNTER.labels(status="success").inc()
    INGESTION_CHUNKS.observe(len(chunks))
    DOCS_IN_MEMORY.set(len(app.state.documents))
"""

from prometheus_client import Counter, Histogram, Gauge


# -------------------------------------------------------------
# Ingestion metrics
# -------------------------------------------------------------

# total ingestion runs, labelled by outcome so you can track
# success vs failure rates separately
INGESTION_COUNTER = Counter(
    "ingestion_runs_total",
    "Total number of ingestion pipeline runs",
    ["status"],           # labels: "success" | "failed"
)

# how long each ingestion pipeline takes end-to-end
INGESTION_DURATION = Histogram(
    "ingestion_duration_seconds",
    "End-to-end ingestion pipeline duration in seconds",
    buckets=[5, 10, 30, 60, 120, 300],
)

# distribution of chunk counts per ingestion — tells you if
# document size is consistent across uploads
INGESTION_CHUNKS = Histogram(
    "ingestion_chunks_per_run",
    "Number of chunks produced per ingestion run",
    buckets=[10, 20, 40, 60, 100, 200],
)


# -------------------------------------------------------------
# Query metrics
# -------------------------------------------------------------

# total queries, labelled by outcome
QUERY_COUNTER = Counter(
    "query_requests_total",
    "Total number of /query requests",
    ["status"],           # labels: "success" | "error" | "no_docs"
)

# end-to-end latency for the RAG pipeline per query
QUERY_LATENCY = Histogram(
    "query_latency_ms",
    "RAG pipeline latency in milliseconds",
    buckets=[500, 1000, 2000, 3000, 5000, 10000],
)


# -------------------------------------------------------------
# System state metrics
# -------------------------------------------------------------

# current number of chunks held in memory — useful for spotting
# if a new ingestion replaced the previous index or failed silently
DOCS_IN_MEMORY = Gauge(
    "docs_in_memory",
    "Number of document chunks currently held in app memory",
)


# -------------------------------------------------------------
# Evidence metrics
# -------------------------------------------------------------

# total evidence render requests, labelled by outcome:
#   "success"      — page rendered and returned
#   "not_loaded"   — no document ingested yet (current_pdf_path is None)
#   "render_error" — fitz/page-range error during rendering
EVIDENCE_COUNTER = Counter(
    "evidence_requests_total",
    "Total number of /evidence/page requests",
    ["status"],
)
