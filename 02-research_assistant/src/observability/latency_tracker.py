"""
latency_tracker.py
------------------
Tracks wall-clock time for each stage of the RAG pipeline.
"""

import time


class LatencyTracker:

    def __init__(self):
        self.start_times = {}
        self.latency = {}

    # -------------------------------------------------------------
    # start: records the start time for a named stage
    # -------------------------------------------------------------
    def start(self, stage):
        self.start_times[stage] = time.time()

    # -------------------------------------------------------------
    # stop: calculates elapsed time since start() and stores it
    # -------------------------------------------------------------
    def stop(self, stage):
        self.latency[stage] = round(time.time() - self.start_times[stage], 4)

    # -------------------------------------------------------------
    # get_latency: returns the latency dict for saving to results
    # -------------------------------------------------------------
    def get_latency(self):
        return self.latency

    # -------------------------------------------------------------
    # log_report: emits per-stage latency and total as structured
    # log records so the breakdown lands in the experiment log file
    # -------------------------------------------------------------
    def log_report(self, logger):
        total = sum(self.latency.values())
        logger.info(
            "latency report",
            extra={"stages": self.latency, "total_sec": round(total, 4)},
        )
