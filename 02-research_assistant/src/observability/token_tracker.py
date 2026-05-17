"""
token_tracker.py
----------------
Tracks token usage and cost for each stage of the RAG pipeline
using LangChain's OpenAI callback.
"""

from langchain_community.callbacks.manager import get_openai_callback


class TokenTracker:

    def __init__(self):
        self.usage = {}
        self.callback_manager = None
        self.cb = None

    # -------------------------------------------------------------
    # start: opens an OpenAI callback context to begin counting
    # tokens from this point forward
    # -------------------------------------------------------------
    def start(self):
        self.callback_manager = get_openai_callback()
        # manually enter the context manager so we control when it closes
        self.cb = self.callback_manager.__enter__()

    # -------------------------------------------------------------
    # stop: closes the callback context and records token usage
    # for the named stage
    # -------------------------------------------------------------
    def stop(self, stage):
        self.callback_manager.__exit__(None, None, None)
        self.usage[stage] = {
            "prompt_tokens":    self.cb.prompt_tokens,
            "completion_tokens": self.cb.completion_tokens,
            "total_tokens":     self.cb.total_tokens,
            "total_cost_usd":   round(self.cb.total_cost, 6),
        }

    # -------------------------------------------------------------
    # get_total_usage: sums token counts across all tracked stages
    # -------------------------------------------------------------
    def get_total_usage(self):
        total_prompt = total_completion = total_tokens = total_cost = 0
        for stage in self.usage.values():
            total_prompt     += stage["prompt_tokens"]
            total_completion += stage["completion_tokens"]
            total_tokens     += stage["total_tokens"]
            total_cost       += stage["total_cost_usd"]
        return {
            "prompt_tokens":     total_prompt,
            "completion_tokens": total_completion,
            "total_tokens":      total_tokens,
            "total_cost_usd":    round(total_cost, 6),
        }

    # -------------------------------------------------------------
    # get_usage: returns per-stage and overall usage combined,
    # used when saving results to disk
    # -------------------------------------------------------------
    def get_usage(self):
        return {"stages": self.usage, "overall": self.get_total_usage()}

    # -------------------------------------------------------------
    # log_report: emits per-stage token counts and totals as
    # structured log records so they land in the experiment log file
    # -------------------------------------------------------------
    def log_report(self, logger):
        logger.info(
            "token report",
            extra={"stages": self.usage, "overall": self.get_total_usage()},
        )
