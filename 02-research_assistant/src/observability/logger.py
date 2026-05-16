"""
logger.py
---------
Centralised logging setup for the RAG pipeline.
"""

import logging
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class TeeStream:
    """Writes to multiple streams at once — mirrors stdout to a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def fileno(self):
        return self.streams[0].fileno()


def tee_stdout_to_file(path):
    """Redirect stdout so every print() also writes to the given file path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(path, "a", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_file)


def get_logger(name, level=logging.INFO):
    """Return a named logger that writes to stdout."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)

    logger.propagate = False

    return logger
