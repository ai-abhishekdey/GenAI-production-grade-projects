"""
token_tracker.py
----------------
Stage-wise token tracking using LangChain callbacks.
"""

from langchain_community.callbacks.manager import get_openai_callback


class TokenTracker:

    def __init__(self):

        self.usage = {}

        self.callback_manager = None
        self.cb = None

    # ── Start Tracking ─────────────────────────────────

    def start(self):

        self.callback_manager = get_openai_callback()

        self.cb = self.callback_manager.__enter__()

    # ── Stop Tracking ──────────────────────────────────

    def stop(self, stage):

        self.callback_manager.__exit__(None, None, None)

        self.usage[stage] = {
            "prompt_tokens": self.cb.prompt_tokens,
            "completion_tokens": self.cb.completion_tokens,
            "total_tokens": self.cb.total_tokens,
            "total_cost_usd": round(self.cb.total_cost, 6)
        }

    # ── Overall Summary ────────────────────────────────

    def get_total_usage(self):

        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        total_cost = 0

        for stage in self.usage.values():

            total_prompt += stage["prompt_tokens"]
            total_completion += stage["completion_tokens"]
            total_tokens += stage["total_tokens"]
            total_cost += stage["total_cost_usd"]

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6)
        }

    # ── Full Usage ─────────────────────────────────────

    def get_usage(self):

        return {
            "stages": self.usage,
            "overall": self.get_total_usage()
        }

    # ── Print Report ───────────────────────────────────

    def print_report(self):

        print("\n========================================")
        print("             TOKEN REPORT               ")
        print("========================================")

        for stage_name, usage in self.usage.items():

            print(f"\n[{stage_name}]")

            for k, v in usage.items():
                print(f"{k} : {v}")

        '''
        print("\n----------------------------------------")
        print("[overall]")

        overall = self.get_total_usage()

        for k, v in overall.items():
            print(f"{k} : {v}")

        print("========================================")
        '''
