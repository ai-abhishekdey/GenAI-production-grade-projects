"""
logger.py
---------
Centralised logging setup for the Research Assistant.

Two log types with distinct directories:
  logs/app/           — one file per API server startup
  logs/experiments/   — one file per experiment run

Each run gets its own timestamped file so concurrent runs never share
or overwrite each other's logs.

Output channels:
  stdout  : always active — human-readable "timestamp | LEVEL | logger | message"
  logfile : only when LOG_TO_FILE=true (default) — JSON lines per run

Set LOG_TO_FILE=false in container environments (HuggingFace Spaces, ECS,
Cloud Run, etc.) where the filesystem is ephemeral and the platform already
captures stdout and routes it to its own log store.

Usage:
    # API (api/main.py lifespan):
    log_file = setup_logging(log_type="app")

    # Experiments (experiment.py):
    log_file = setup_logging(log_type="experiment", run_id=run_id)

    # Any module:
    logger = get_logger(__name__)
    logger.info("chunks indexed", extra={"chunk_count": 40, "source": "paper.pdf"})
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pythonjsonlogger import jsonlogger


CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT    = "%Y-%m-%d %H:%M:%S"

LOG_DIRS = {
    "app":        Path("logs/app"),
    "experiment": Path("logs/experiments"),
}

# set LOG_TO_FILE=false in containers — stdout is captured by the platform
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"


# -------------------------------------------------------------
# JsonFormatter: enriches every JSON log record with a UTC
# timestamp, level, and logger name as top-level fields so
# log aggregators (Datadog, CloudWatch, Loki) can filter by them
# -------------------------------------------------------------
class JsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.now(timezone.utc).strftime(DATE_FORMAT)
        log_record["level"]     = record.levelname
        log_record["logger"]    = record.name


# -------------------------------------------------------------
# setup_logging: configures the root logger for one run.
#
# log_type : "app" | "experiment" — determines the subdirectory
# run_id   : optional label used as the filename so experiment
#            logs and result files share the same identifier.
#            Auto-generated from the current timestamp if omitted.
# level    : minimum log level (default INFO)
#
# Returns the Path of the log file if LOG_TO_FILE=true, else None.
# Guards against duplicate setup (safe for uvicorn --reload).
# -------------------------------------------------------------
def setup_logging(
    log_type: str = "app",
    run_id: str = None,
    level: int = logging.INFO,
) -> Path | None:
    if log_type not in LOG_DIRS:
        raise ValueError(f"log_type must be one of {list(LOG_DIRS)}")

    root = logging.getLogger()

    # guard: don't add handlers a second time (e.g. uvicorn --reload)
    if root.handlers:
        return None

    root.setLevel(level)

    # --- console: always active, captured by every platform ---
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(console)

    # --- file: only when LOG_TO_FILE=true (local / persistent-volume deploys) ---
    log_file = None
    if LOG_TO_FILE:
        log_dir = LOG_DIRS[log_type]
        log_dir.mkdir(parents=True, exist_ok=True)
        file_id  = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_dir / f"{file_id}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter("%(message)s"))
        root.addHandler(file_handler)

    # silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "watchdog", "deepeval"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "logging initialised",
        extra={
            "log_type":    log_type,
            "log_to_file": LOG_TO_FILE,
            "log_file":    str(log_file) if log_file else "stdout only",
        },
    )

    return log_file


# -------------------------------------------------------------
# get_logger: returns a named logger. All configuration is
# inherited from the root logger configured by setup_logging().
# -------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
