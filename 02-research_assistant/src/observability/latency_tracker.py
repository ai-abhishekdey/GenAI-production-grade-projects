"""
latency_tracker.py
------------------
Simple utility for tracking latency of different stages.
"""

import time


class LatencyTracker:

    def __init__(self):

        self.start_time = {}
        self.latency = {}

    def start(self, stage):

        self.start_time[stage] = time.time()

    def stop(self, stage):

        elapsed = time.time() - self.start_time[stage]

        self.latency[stage] = round(elapsed, 4)

    def get_latency(self):

        return self.latency

    def print_latency(self):

        print("\n========================================")
        print("           LATENCY REPORT               ")
        print("========================================")

        total = 0

        for stage, value in self.latency.items():

            print(f"{stage} : {value} sec")

            total += value

        print("----------------------------------------")
        print(f"total : {round(total, 4)} sec")
        print("========================================")
