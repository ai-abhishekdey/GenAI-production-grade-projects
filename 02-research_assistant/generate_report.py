"""
report_generator.py
────────────────────────────────────────────────────────────────────────────
Generates a professional PDF report from compiled RAG experiment results.
Supports multi-dimensional experiments: chunking × retrieval × reranker.

Inputs  (from config.py):
    - COMPILED_CSV_PATH          → experiment_summary.csv
    - COMPILED_ANALYTICS_JSON    → experiment_analytics.json

Output  (from config.py):
    - EXPERIMENT_SUMMARY_PDF     → experiment_summary.pdf

Intermediate plots are saved to:
    - COMPILED_RESULTS_DIR/plots/

Usage:
    python report_generator.py
"""

from src.generation.llm import get_summariser_llm
from config import (
    COMPILED_ANALYTICS_JSON,
    COMPILED_CSV_PATH,
    COMPILED_RESULTS_DIR,
    EXPERIMENT_SUMMARY_PDF,
)
from reportlab.platypus.flowables import Flowable
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib import colors
from langchain_core.messages import HumanMessage
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
NAVY = HexColor("#0D1B2A")
DEEP_BLUE = HexColor("#1B3A5C")
ACCENT = HexColor("#2563EB")
ACCENT_LIGHT = HexColor("#BFDBFE")
TEAL = HexColor("#0D9488")
GOLD = HexColor("#D97706")
LIGHT_BG = HexColor("#F1F5F9")
BORDER = HexColor("#CBD5E1")
WHITE = colors.white
BLACK = HexColor("#0F172A")
GRAY_TEXT = HexColor("#64748B")
SUCCESS = HexColor("#15803D")
DANGER = HexColor("#B91C1C")
WARN = HexColor("#D97706")

MPL_PALETTE = ["#2563EB", "#0D9488", "#D97706",
               "#7C3AED", "#DB2777", "#0F766E", "#B45309"]

QUALITY_METRICS = {
    "overall_score":       ("Overall Score",       True),
    "answer_correctness":  ("Answer Correctness",  True),
    "answer_similarity":   ("Answer Similarity",   True),
    "faithfulness":        ("Faithfulness",        True),
    "answer_relevancy":    ("Answer Relevancy",    True),
    "context_precision":   ("Context Precision",   True),
    "context_recall":      ("Context Recall",      True),
}

LATENCY_METRICS = {
    "ingestion_latency":      "Ingestion",
    "preprocessing_latency":  "Preprocessing",
    "chunking_latency":       "Chunking",
    "vector_store_latency":   "Vector Store",
    "rag_latency":            "RAG Total",
    "evaluation_latency":     "Evaluation",
}

COST_METRICS = {
    "prompt_tokens":     "Prompt Tokens",
    "completion_tokens": "Completion Tokens",
    "total_tokens":      "Total Tokens",
    "total_cost_usd":    "Total Cost (USD)",
}

PLOTS_DIR = Path(COMPILED_RESULTS_DIR) / "plots"


# ─────────────────────────────────────────────────────────────────────────────
# RUN LABEL — human-readable identifier combining all experiment dimensions
# ─────────────────────────────────────────────────────────────────────────────
def build_run_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'run_label' column combining chunking + retrieval + reranker.
    Used as the x-axis label across all charts.

    Example: "recursive | hybrid | flashrank"
              "semantic  | dense  | none"
    """
    def make_label(row):
        label = f"{row['chunking']} | {row['retrieval']}"
        if row.get("reranker_enabled", False):
            label += f" | {row.get('reranker_type', 'reranker')}"
        return label

    df = df.copy()
    df["run_label"] = df.apply(make_label, axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_plots_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, filename: str) -> Path:
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _ax_style(ax, title=""):
    ax.set_facecolor("#F8FAFC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color="#0D1B2A", pad=10)


def _img(path: Path, width_pts: float, caption: str, styles: dict):
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            w, h = im.size
        asp = h / w
    except Exception:
        asp = 0.55
    img = Image(str(path), width=width_pts, height=width_pts * asp)
    cap = Paragraph(caption, styles["caption"])
    return img, cap


def _rotate_labels(ax, df, rotation=25):
    """Apply rotated x-tick labels using run_label."""
    ax.set_xticklabels(df["run_label"], fontsize=8,
                       rotation=rotation, ha="right")


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_overall_leaderboard(df: pd.DataFrame) -> Path:
    """Horizontal bar — overall score per experiment run."""
    df_s = df.sort_values("overall_score", ascending=True)
    best_label = df_s["run_label"].iloc[-1]

    bar_colors = [
        "#15803D" if l == best_label else "#93C5FD"
        for l in df_s["run_label"]
    ]

    fig, ax = plt.subplots(
        figsize=(9, max(3, len(df_s) * 0.9)), facecolor="#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    bars = ax.barh(df_s["run_label"], df_s["overall_score"],
                   color=bar_colors, edgecolor="white", height=0.5, zorder=3)

    for bar, v in zip(bars, df_s["overall_score"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9,
                fontweight="bold", color="#0F172A")

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Overall Score", fontsize=10)
    legend_patches = [
        mpatches.Patch(color="#15803D", label="Best"),
        mpatches.Patch(color="#93C5FD", label="Others"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    _ax_style(ax, "Overall Score — Experiment Leaderboard")
    fig.tight_layout()
    return _save_fig(fig, "01_overall_leaderboard.png")


def chart_quality_grouped_bar(df: pd.DataFrame) -> Path:
    """Grouped bar chart for all quality metrics across all runs."""
    metric_keys = list(QUALITY_METRICS.keys())
    labels = [QUALITY_METRICS[k][0] for k in metric_keys]
    x = np.arange(len(metric_keys))
    n = len(df)
    width = 0.65 / n

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[k] for k in metric_keys]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=row["run_label"],
                      color=MPL_PALETTE[i % len(MPL_PALETTE)],
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6, fontweight="bold", color="#1e293b")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Score", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.9, loc="upper right",
              title="Run", title_fontsize=8, ncol=2)
    _ax_style(ax, "RAG Quality Metrics — All Experiment Runs")
    fig.tight_layout()
    return _save_fig(fig, "02_quality_grouped_bar.png")


def chart_radar(df: pd.DataFrame) -> Path:
    """Spider / radar chart — one line per experiment run."""
    metric_keys = list(QUALITY_METRICS.keys())
    labels = [QUALITY_METRICS[k][0] for k in metric_keys]
    N = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True),
                           facecolor="#F8FAFC")
    ax.set_facecolor("#EEF2F7")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=7, color="#94a3b8")

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[k] for k in metric_keys] + [row[metric_keys[0]]]
        col = MPL_PALETTE[i % len(MPL_PALETTE)]
        ax.plot(angles, vals, color=col, linewidth=2.0, label=row["run_label"])
        ax.fill(angles, vals, color=col, alpha=0.07)

    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.15),
              fontsize=8, framealpha=0.9, title="Run", title_fontsize=8)
    ax.set_title("Multi-Metric Radar", fontsize=12,
                 fontweight="bold", color="#0D1B2A", pad=22)
    fig.tight_layout()
    return _save_fig(fig, "03_radar.png")


def chart_latency_stacked(df: pd.DataFrame) -> Path:
    """Stacked bar — latency breakdown per experiment run."""
    latency_keys = ["ingestion_latency", "preprocessing_latency",
                    "chunking_latency", "vector_store_latency"]
    latency_labels = [LATENCY_METRICS[k] for k in latency_keys]
    x = np.arange(len(df))
    width = 0.45
    lat_palette = ["#2563EB", "#0D9488", "#D97706", "#7C3AED"]

    fig, ax = plt.subplots(
        figsize=(max(9, len(df) * 1.5), 5), facecolor="#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    bottom = np.zeros(len(df))
    for j, (key, label) in enumerate(zip(latency_keys, latency_labels)):
        vals = df[key].values.astype(float)
        ax.bar(x, vals, width, bottom=bottom,
               label=label, color=lat_palette[j],
               edgecolor="white", linewidth=0.6, zorder=3)
        bottom += vals

    for i, total in enumerate(bottom):
        ax.text(i, total + 0.1, f"{total:.2f}s",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#0F172A")

    ax.set_xticks(x)
    _rotate_labels(ax, df)
    ax.set_ylabel("Latency (seconds)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    _ax_style(ax, "Ingestion Latency Breakdown by Experiment Run")
    fig.tight_layout()
    return _save_fig(fig, "04_latency_stacked.png")


def chart_rag_latency_bar(df: pd.DataFrame) -> Path:
    """Bar chart — RAG inference latency per run."""
    best_val = df["rag_latency"].min()
    bar_cols = ["#15803D" if v ==
                best_val else "#93C5FD" for v in df["rag_latency"]]
    x = np.arange(len(df))

    fig, ax = plt.subplots(
        figsize=(max(8, len(df) * 1.5), 4), facecolor="#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    bars = ax.bar(x, df["rag_latency"], color=bar_cols,
                  edgecolor="white", width=0.5, zorder=3)

    for bar, v in zip(bars, df["rag_latency"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{v:.2f}s", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#0F172A")

    ax.set_xticks(x)
    _rotate_labels(ax, df)
    ax.set_ylabel("RAG Latency (seconds)", fontsize=10)
    legend_patches = [
        mpatches.Patch(color="#15803D", label="Fastest"),
        mpatches.Patch(color="#93C5FD", label="Others"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)
    _ax_style(ax, "RAG Inference Latency by Experiment Run")
    fig.tight_layout()
    return _save_fig(fig, "05_rag_latency.png")


def chart_cost_vs_quality(df: pd.DataFrame) -> Path:
    """Scatter — cost vs overall score, bubble size = total tokens."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    for i, (_, row) in enumerate(df.iterrows()):
        col = MPL_PALETTE[i % len(MPL_PALETTE)]
        ax.scatter(row["total_cost_usd"], row["overall_score"],
                   s=row["total_tokens"] / 25,
                   color=col, alpha=0.85,
                   edgecolors="white", linewidth=1.5, zorder=4)
        ax.annotate(
            row["run_label"],
            (row["total_cost_usd"], row["overall_score"]),
            xytext=(7, 5), textcoords="offset points",
            fontsize=7.5, color=col, fontweight="bold",
        )

    ax.set_xlabel("Total Cost (USD)", fontsize=10)
    ax.set_ylabel("Overall Score", fontsize=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    _ax_style(ax, "Cost vs Quality Trade-off  (bubble size = total tokens)")
    fig.tight_layout()
    return _save_fig(fig, "06_cost_vs_quality.png")


def chart_token_usage(df: pd.DataFrame) -> Path:
    """Stacked bar — prompt vs completion tokens per run."""
    x = np.arange(len(df))
    width = 0.45

    fig, ax = plt.subplots(
        figsize=(max(9, len(df) * 1.5), 4.5), facecolor="#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    ax.bar(x, df["prompt_tokens"], width,
           label="Prompt Tokens", color="#2563EB",
           edgecolor="white", zorder=3)
    ax.bar(x, df["completion_tokens"], width,
           bottom=df["prompt_tokens"],
           label="Completion Tokens", color="#F97316",
           edgecolor="white", zorder=3)

    for i, (_, row) in enumerate(df.iterrows()):
        total = row["total_tokens"]
        ax.text(i, total + 80, f"{int(total):,}",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#0F172A")

    ax.set_xticks(x)
    _rotate_labels(ax, df)
    ax.set_ylabel("Tokens", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    _ax_style(ax, "Token Usage by Experiment Run")
    fig.tight_layout()
    return _save_fig(fig, "07_token_usage.png")


def chart_dimension_comparison(df: pd.DataFrame) -> Path:
    """
    Box plots comparing overall score by each experiment dimension.
    Shows chunking, retrieval, and reranker side by side.
    """

    dims = []

    if df["chunking"].nunique() > 1:
        dims.append(("chunking", "Chunking Strategy"))

    if df["retrieval"].nunique() > 1:
        dims.append(("retrieval", "Retrieval Strategy"))

    if "reranker_type" in df.columns and df["reranker_type"].nunique() > 1:
        dims.append(("reranker_type", "Reranker"))

    if not dims:
        return None

    fig, axes = plt.subplots(
        1,
        len(dims),
        figsize=(5 * len(dims), 5),
        facecolor="#F8FAFC"
    )

    if len(dims) == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, dims):

        unique_vals = list(df[col].unique())

        groups = [
            df[df[col] == v]["overall_score"].values
            for v in unique_vals
        ]

        bp = ax.boxplot(
            groups,
            patch_artist=True,
            tick_labels=unique_vals   # FIXED (Matplotlib 3.9+)
        )

        # Style box colors
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(MPL_PALETTE[i % len(MPL_PALETTE)])
            patch.set_alpha(0.7)

        # Median line styling
        for median in bp["medians"]:
            median.set_color("#0F172A")
            median.set_linewidth(2)

        # Whiskers styling
        for whisker in bp["whiskers"]:
            whisker.set_color("#475569")

        # Caps styling
        for cap in bp["caps"]:
            cap.set_color("#475569")

        # Outlier styling
        for flier in bp["fliers"]:
            flier.set(
                marker="o",
                markersize=4,
                markerfacecolor="#EF4444",
                alpha=0.6
            )

        ax.set_ylabel("Overall Score", fontsize=9)

        ax.set_xticklabels(
            unique_vals,
            fontsize=8,
            rotation=15,
            ha="right"
        )

        _ax_style(ax, title)

    # FIXED COLOR ISSUE
    fig.suptitle(
        "Overall Score Distribution by Experiment Dimension",
        fontsize=11,
        fontweight="bold",
        color="#0D1B2A",
        y=1.02
    )

    fig.tight_layout()

    return _save_fig(fig, "08_dimension_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS — best/worst per dimension
# ─────────────────────────────────────────────────────────────────────────────
def compute_analytics(df: pd.DataFrame) -> dict:
    """
    Computes best/worst per metric and per experiment dimension.
    Also identifies the single best overall configuration.
    """
    analytics = {"num_experiments": len(df)}

    # Best/worst per metric (by run_label)
    metric_keys = list(QUALITY_METRICS.keys()) + [
        "rag_latency", "evaluation_latency", "total_tokens", "total_cost_usd"
    ]
    lower_is_better = {"rag_latency", "evaluation_latency",
                       "total_tokens", "total_cost_usd"}

    for key in metric_keys:
        if key not in df.columns:
            continue
        best_idx = df[key].idxmin(
        ) if key in lower_is_better else df[key].idxmax()
        worst_idx = df[key].idxmax(
        ) if key in lower_is_better else df[key].idxmin()
        analytics[key] = {
            "best":  {"strategy": df.loc[best_idx,  "run_label"], "value": float(df.loc[best_idx,  key])},
            "worst": {"strategy": df.loc[worst_idx, "run_label"], "value": float(df.loc[worst_idx, key])},
        }

    # Best config per dimension
    for dim in ["chunking", "retrieval", "reranker_type"]:
        if dim not in df.columns or df[dim].nunique() <= 1:
            continue
        dim_summary = df.groupby(
            dim)["overall_score"].mean().sort_values(ascending=False)
        analytics[f"best_{dim}"] = {
            "value": dim_summary.index[0],
            "avg_score": float(dim_summary.iloc[0])
        }

    # Best overall combination
    best_row = df.loc[df["overall_score"].idxmax()]
    analytics["best_combination"] = {
        "run_label":   best_row["run_label"],
        "chunking":    best_row["chunking"],
        "retrieval":   best_row["retrieval"],
        "reranker":    best_row.get("reranker_type", "none"),
        "overall_score": float(best_row["overall_score"]),
    }

    return analytics


# ─────────────────────────────────────────────────────────────────────────────
# LLM NARRATIVE
# ─────────────────────────────────────────────────────────────────────────────
def generate_narrative(df: pd.DataFrame, analytics: dict) -> dict:
    logger.info("Calling LLM for narrative generation...")

    llm = get_summariser_llm()

    csv_snapshot = df[[
        "run_label", "chunking", "retrieval", "reranker_enabled",
        "overall_score", "answer_correctness", "faithfulness",
        "answer_relevancy", "context_recall", "rag_latency", "total_cost_usd"
    ]].to_string(index=False)

    best_combo = analytics.get("best_combination", {})

    prompt = f"""You are a senior RAG systems researcher writing a professional experiment report.

You ran {analytics['num_experiments']} RAG pipeline experiments varying:
- Chunking strategies (fixed, recursive, semantic, page, layout)  
- Retrieval strategies (dense, sparse, hybrid, mmr)
- Reranker options (none, flashrank, crossencoder)

Best overall configuration: {best_combo.get('run_label', 'N/A')} 
with overall score: {best_combo.get('overall_score', 0):.4f}

CSV SNAPSHOT:
{csv_snapshot}

ANALYTICS:
{json.dumps(analytics, indent=2)}

Write the following three sections. Be precise, technical, and concise.
Use concrete numbers from the data. Do NOT use markdown headers or bullet symbols.
Separate each section with the exact delimiter shown.

---EXECUTIVE_SUMMARY---
Write 3-4 sentences summarising the overall experiment outcomes.
Mention the best configuration (chunking + retrieval + reranker) and what it achieved.
Compare the impact of each dimension on overall score.

---KEY_FINDINGS---
Write exactly 5 key findings, each as a single sentence starting with a number and period (e.g. "1. ...").
Focus on: which dimension had most impact, trade-offs between quality/cost/latency, 
surprising results, and the best configuration recommendation.

---NEXT_STEPS---
Write exactly 5 concrete next steps as actionable recommendations, each starting with a number and period.
Include: production deployment, further hyperparameter tuning, monitoring, and cost optimization.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content

    def _extract(tag: str) -> str:
        try:
            start = text.index(f"---{tag}---") + len(f"---{tag}---")
            tags = ["EXECUTIVE_SUMMARY", "KEY_FINDINGS", "NEXT_STEPS"]
            ends = [
                text.index(f"---{t}---")
                for t in tags
                if f"---{t}---" in text and text.index(f"---{t}---") > start
            ]
            end = min(ends) if ends else len(text)
            return text[start:end].strip()
        except ValueError:
            return ""

    return {
        "executive_summary": _extract("EXECUTIVE_SUMMARY"),
        "key_findings":      _extract("KEY_FINDINGS"),
        "next_steps":        _extract("NEXT_STEPS"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM REPORTLAB FLOWABLES
# ─────────────────────────────────────────────────────────────────────────────
class HeaderBanner(Flowable):
    def __init__(self, title: str, subtitle: str, date_str: str, width: float):
        super().__init__()
        self._title, self._subtitle = title, subtitle
        self._date, self._w = date_str, width
        self._h = 100

    def draw(self):
        c = self.canv
        c.setFillColor(NAVY)
        c.rect(0, 0, self._w, self._h, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.rect(0, self._h - 5, self._w, 5, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(0, 0, self._w, 3, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(26, self._h - 42, self._title)
        c.setFont("Helvetica", 11)
        c.setFillColor(ACCENT_LIGHT)
        c.drawString(26, self._h - 60, self._subtitle)
        c.setFont("Helvetica", 8)
        c.setFillColor(GRAY_TEXT)
        c.drawRightString(self._w - 26, 22, "RAG EVALUATION REPORT")
        c.drawRightString(self._w - 26, 10, self._date)

    def wrap(self, aw, ah):
        return self._w, self._h


class SectionHeader(Flowable):
    def __init__(self, text: str, width: float, level: int = 1):
        super().__init__()
        self._text, self._w, self._level = text, width, level
        self._h = 34 if level == 1 else 26

    def draw(self):
        c = self.canv
        bar_color = ACCENT if self._level == 1 else TEAL
        c.setFillColor(bar_color)
        c.rect(0, 0, 5, self._h, fill=1, stroke=0)
        c.setFillColor(LIGHT_BG)
        c.rect(5, 0, self._w - 5, self._h, fill=1, stroke=0)
        fs = 13 if self._level == 1 else 11
        c.setFillColor(NAVY if self._level == 1 else DEEP_BLUE)
        c.setFont("Helvetica-Bold", fs)
        c.drawString(16, 10, self._text)

    def wrap(self, aw, ah):
        return self._w, self._h


class MetricBadge(Flowable):
    def __init__(self, label: str, value: str, color: HexColor, sub: str = ""):
        super().__init__()
        self._label, self._value = label, value
        self._color, self._sub = color, sub
        self._w, self._h = 112, 52

    def draw(self):
        c = self.canv
        c.setFillColor(self._color)
        c.roundRect(0, 0, self._w, self._h, 7, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica", 7.5)
        c.drawCentredString(self._w / 2, 38, self._label.upper())
        c.setFont("Helvetica-Bold", 15)
        c.drawCentredString(self._w / 2, 20, self._value)
        if self._sub:
            c.setFont("Helvetica", 7)
            c.setFillColor(ACCENT_LIGHT)
            c.drawCentredString(self._w / 2, 8, self._sub[:28])

    def wrap(self, aw, ah):
        return self._w, self._h


# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
def build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=10, leading=15, textColor=BLACK,
            spaceAfter=5, alignment=TA_JUSTIFY, fontName="Helvetica",
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"],
            fontSize=10, leading=14, textColor=BLACK,
            leftIndent=14, spaceAfter=5, fontName="Helvetica",
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"],
            fontSize=8, leading=11, textColor=GRAY_TEXT,
            alignment=TA_CENTER, fontName="Helvetica-Oblique", spaceAfter=8,
        ),
        "th": ParagraphStyle(
            "th", parent=base["Normal"],
            fontSize=8.5, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "tc": ParagraphStyle(
            "tc", parent=base["Normal"],
            fontSize=8.5, textColor=BLACK,
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "tc_bold": ParagraphStyle(
            "tc_bold", parent=base["Normal"],
            fontSize=8.5, textColor=BLACK,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "disc": ParagraphStyle(
            "disc", parent=base["Normal"],
            fontSize=8, textColor=GRAY_TEXT,
            alignment=TA_CENTER, fontName="Helvetica-Oblique",
        ),
        "best_label": ParagraphStyle(
            "best_label", parent=base["Normal"],
            fontSize=8.5, textColor=SUCCESS,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "worst_label": ParagraphStyle(
            "worst_label", parent=base["Normal"],
            fontSize=8.5, textColor=DANGER,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PAGE TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
def make_on_page(report_title: str):
    def fn(canvas, doc):
        canvas.saveState()
        W, H = A4
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, W, 20, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(
            22, 6, f"{report_title}  |  {datetime.now().strftime('%d %b %Y')}")
        canvas.drawRightString(W - 22, 6, f"Page {doc.page}")
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(1.5)
        canvas.line(22, H - 22, W - 22, H - 22)
        canvas.restoreState()
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# PDF ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
def _section(header_flowable, *content):
    return KeepTogether([header_flowable] + list(content))


def build_pdf(
    df: pd.DataFrame,
    analytics: dict,
    narrative: dict,
    chart_paths: dict,
    output_path: str,
):
    W, H = A4
    MARGIN = 22 * mm
    CW = W - 2 * MARGIN

    best_combo = analytics["best_combination"]
    best_label = best_combo["run_label"]
    best_score = best_combo["overall_score"]
    n_exp = analytics["num_experiments"]
    best_row = df[df["run_label"] == best_label].iloc[0]

    REPORT_TITLE = "RAG Pipeline Evaluation Report"

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=30 * mm, bottomMargin=24 * mm,
        title=REPORT_TITLE, author="RAG Experiment Pipeline",
    )

    styles = build_styles()
    story = []
    on_page = make_on_page(REPORT_TITLE)

    # ── 1. HEADER ─────────────────────────────────────────────────────────────
    subtitle = (
        f"{n_exp} experiments  |  "
        f"Chunking × Retrieval × Reranker  |  "
        f"Best: {best_label}"
    )
    story.append(HeaderBanner(
        title=REPORT_TITLE, subtitle=subtitle,
        date_str=datetime.now().strftime("%d %B %Y"), width=CW,
    ))
    story.append(Spacer(1, 14))

    # ── 2. EXECUTIVE SUMMARY ──────────────────────────────────────────────────
    story.append(SectionHeader("Executive Summary", CW))
    story.append(Spacer(1, 6))
    exec_tbl = Table(
        [[Paragraph(narrative["executive_summary"], styles["body"])]],
        colWidths=[CW],
    )
    exec_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), LIGHT_BG),
        ("BOX",          (0, 0), (-1, -1), 1, BORDER),
        ("LINEAFTER",    (0, 0), (0, -1),  3, ACCENT),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(exec_tbl)
    story.append(Spacer(1, 12))

    # ── 3. METRIC BADGES ──────────────────────────────────────────────────────
    badge_defs = [
        ("Best Config",    best_label[:22],
         TEAL,      f"Score {best_score:.4f}"),
        ("Overall Score",  f"{best_score:.4f}",
         ACCENT,    best_combo["chunking"].title()),
        ("Faithfulness",
         f"{best_row['faithfulness']:.4f}", DEEP_BLUE, best_label[:22]),
        ("RAG Latency",    f"{analytics['rag_latency']['best']['value']:.2f}s",
         NAVY,      analytics['rag_latency']['best']['strategy'][:22]),
        ("Lowest Cost",    f"${analytics['total_cost_usd']['best']['value']:.4f}",
         GOLD,      analytics['total_cost_usd']['best']['strategy'][:22]),
        ("Context Recall", f"{analytics['context_recall']['best']['value']:.4f}",
         HexColor("#7C3AED"),
         analytics['context_recall']['best']['strategy'][:22]),
    ]
    bw3 = CW / 3
    for badge_row in [badge_defs[:3], badge_defs[3:]]:
        row_data = [[MetricBadge(l, v, c, s) for l, v, c, s in badge_row]]
        badge_tbl = Table(row_data, colWidths=[bw3] * 3)
        badge_tbl.setStyle(TableStyle([
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(badge_tbl)
        story.append(Spacer(1, 6))

    # ── 4. OVERALL LEADERBOARD ────────────────────────────────────────────────
    i1, c1 = _img(chart_paths["leaderboard"], CW,
                  "Fig 1 — Overall score per experiment run (best highlighted in green)", styles)
    story.append(_section(
        SectionHeader("Overall Score Leaderboard", CW),
        Spacer(1, 8), i1, c1, Spacer(1, 8),
    ))

    # ── 5. DIMENSION COMPARISON ───────────────────────────────────────────────
    if chart_paths.get("dimension_comparison"):
        i_dim, c_dim = _img(
            chart_paths["dimension_comparison"], CW,
            "Fig 2 — Overall score distribution by experiment dimension (chunking / retrieval / reranker)",
            styles,
        )
        story.append(_section(
            SectionHeader("Impact by Experiment Dimension", CW),
            Spacer(1, 8), i_dim, c_dim, Spacer(1, 8),
        ))

    # ── 6. QUALITY METRICS ────────────────────────────────────────────────────
    i2, c2 = _img(chart_paths["radar"], CW,
                  "Fig 3 — Multi-metric radar comparison across all experiment runs", styles)
    i3, c3 = _img(chart_paths["quality_bar"], CW,
                  "Fig 4 — Grouped quality metric comparison across all runs", styles)

    story.append(_section(
        SectionHeader("RAG Quality Metrics", CW),
        Spacer(1, 8), i2, c2,
    ))
    story.append(Spacer(1, 8))
    story.append(KeepTogether([i3, c3]))
    story.append(Spacer(1, 12))

    # ── 7. BEST / WORST TABLE ─────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader("Best & Worst Configuration per Metric", CW))
    story.append(Spacer(1, 8))

    bw_headers = ["Metric", "Best Config",
                  "Best Value", "Worst Config", "Worst Value"]
    bw_col_w = [110, 110, 68, 110, 68]

    bw_data = [[Paragraph(h, styles["th"]) for h in bw_headers]]
    metric_labels = {
        "overall_score":      "Overall Score",
        "answer_correctness": "Answer Correctness",
        "answer_similarity":  "Answer Similarity",
        "faithfulness":       "Faithfulness",
        "answer_relevancy":   "Answer Relevancy",
        "context_precision":  "Context Precision",
        "context_recall":     "Context Recall",
        "rag_latency":        "RAG Latency (s)  ↓",
        "evaluation_latency": "Eval Latency (s)  ↓",
        "total_tokens":       "Total Tokens  ↓",
        "total_cost_usd":     "Total Cost USD  ↓",
    }

    for key, display in metric_labels.items():
        if key not in analytics:
            continue
        entry = analytics[key]
        b_label = entry["best"]["strategy"]
        b_val = entry["best"]["value"]
        w_label = entry["worst"]["strategy"]
        w_val = entry["worst"]["value"]
        fmt = ".4f" if isinstance(b_val, float) and b_val < 10 else ".2f"
        bw_data.append([
            Paragraph(display,          styles["tc"]),
            Paragraph(b_label,          styles["best_label"]),
            Paragraph(f"{b_val:{fmt}}", styles["best_label"]),
            Paragraph(w_label,          styles["worst_label"]),
            Paragraph(f"{w_val:{fmt}}", styles["worst_label"]),
        ])

    bw_row_styles = [
        ("BACKGROUND", (0, i), (-1, i), LIGHT_BG)
        for i in range(1, len(bw_data)) if i % 2 == 0
    ]
    bw_table = Table(bw_data, colWidths=bw_col_w, repeatRows=1)
    bw_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        *bw_row_styles,
    ]))
    story.append(bw_table)
    story.append(Spacer(1, 16))

    # ── 8. FULL RESULTS TABLE ─────────────────────────────────────────────────
    display_cols = [
        ("run_label",          "Configuration"),
        ("overall_score",      "Overall"),
        ("answer_correctness", "Correctness"),
        ("faithfulness",       "Faithfulness"),
        ("answer_relevancy",   "Relevancy"),
        ("context_precision",  "Ctx Precision"),
        ("context_recall",     "Ctx Recall"),
        ("rag_latency",        "RAG Lat (s)"),
        ("total_cost_usd",     "Cost (USD)"),
    ]
    col_keys = [c[0] for c in display_cols]
    col_labels = [c[1] for c in display_cols]
    col_w = [90, 42, 52, 52, 48, 58, 48, 48, 48]

    full_data = [[Paragraph(h, styles["th"]) for h in col_labels]]
    for _, row in df.iterrows():
        is_best = row["run_label"] == best_label
        st = styles["tc_bold"] if is_best else styles["tc"]
        r = []
        for k in col_keys:
            v = row[k]
            if isinstance(v, float):
                r.append(Paragraph(f"{v:.4f}" if v < 10 else f"{v:.2f}", st))
            else:
                r.append(Paragraph(str(v), st))
        full_data.append(r)

    best_row_idx = list(df["run_label"]).index(best_label) + 1
    full_tbl = Table(full_data, colWidths=col_w, repeatRows=1)
    full_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0),            (-1, 0),             NAVY),
        ("TEXTCOLOR",     (0, 0),            (-1, 0),             WHITE),
        ("BACKGROUND",    (0, best_row_idx),
         (-1, best_row_idx),  HexColor("#D1FAE5")),
        ("ALIGN",         (0, 0),            (-1, -1),            "CENTER"),
        ("VALIGN",        (0, 0),            (-1, -1),            "MIDDLE"),
        ("GRID",          (0, 0),            (-1, -1),            0.4, BORDER),
        ("TOPPADDING",    (0, 0),            (-1, -1),            7),
        ("BOTTOMPADDING", (0, 0),            (-1, -1),            7),
        *[("BACKGROUND",  (0, i),            (-1, i),             LIGHT_BG)
          for i in range(1, len(full_data)) if i % 2 == 0 and i != best_row_idx],
    ]))

    story.append(_section(
        SectionHeader("Full Experiment Results Table", CW),
        Spacer(1, 8), full_tbl, Spacer(1, 6),
        Paragraph(
            f"Best configuration (<b>{best_label}</b>) highlighted in green.",
            styles["caption"],
        ),
    ))
    story.append(Spacer(1, 12))

    # ── 9. LATENCY CHARTS ─────────────────────────────────────────────────────
    story.append(PageBreak())
    i4, c4 = _img(chart_paths["latency_stacked"], CW,
                  "Fig 5 — Latency breakdown by experiment run", styles)
    story.append(_section(
        SectionHeader("Latency Analysis", CW),
        Spacer(1, 8), i4, c4,
    ))
    story.append(Spacer(1, 10))

    i5, c5 = _img(chart_paths["rag_latency"], CW,
                  "Fig 6 — RAG inference latency (fastest run highlighted)", styles)
    story.append(KeepTogether([i5, c5]))
    story.append(Spacer(1, 14))

    # ── 10. COST & TOKEN CHARTS ───────────────────────────────────────────────
    i6, c6 = _img(chart_paths["cost_quality"], CW,
                  "Fig 7 — Cost vs quality trade-off  (bubble size = total tokens)", styles)
    story.append(_section(
        SectionHeader("Cost & Token Analysis", CW),
        Spacer(1, 8), i6, c6,
    ))
    story.append(Spacer(1, 10))

    i7, c7 = _img(chart_paths["token_usage"], CW,
                  "Fig 8 — Token usage breakdown (prompt vs completion) by run", styles)
    story.append(KeepTogether([i7, c7]))
    story.append(Spacer(1, 16))

    # ── 11. KEY FINDINGS ──────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(SectionHeader("Key Findings", CW))
    story.append(Spacer(1, 8))
    for line in narrative["key_findings"].splitlines():
        line = line.strip()
        if line:
            story.append(Paragraph(line, styles["bullet"]))
            story.append(Spacer(1, 4))
    story.append(Spacer(1, 12))

    # ── 12. NEXT STEPS ────────────────────────────────────────────────────────
    story.append(SectionHeader("Recommended Next Steps", CW))
    story.append(Spacer(1, 8))

    ns_rows = []
    for line in narrative["next_steps"].splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and ". " in line:
            num, text = line.split(". ", 1)
        else:
            num, text = "•", line
        ns_rows.append([
            Paragraph(f"<b>{num}</b>",
                      ParagraphStyle("n", fontSize=14, textColor=ACCENT,
                                     fontName="Helvetica-Bold", alignment=TA_CENTER)),
            Paragraph(text, styles["body"]),
        ])

    if ns_rows:
        ns_tbl = Table(ns_rows, colWidths=[30, CW - 30])
        ns_tbl.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",   (1, 0), (1, -1),  10),
            ("TOPPADDING",    (0, 0), (-1, -1),   7),
            ("BOTTOMPADDING", (0, 0), (-1, -1),   7),
            ("LINEBELOW",     (0, 0), (-1, -2),  0.4, BORDER),
        ]))
        story.append(ns_tbl)

    story.append(Spacer(1, 24))

    # ── FOOTER DISCLAIMER ─────────────────────────────────────────────────────
    story.append(HRFlowable(width=CW, thickness=1, color=BORDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report was automatically generated by the RAG Experiment Pipeline. "
        "Narratives are LLM-generated — cross-reference with raw data before "
        "making production decisions.",
        styles["disc"],
    ))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    logger.info(f"PDF saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def generate_report():
    _ensure_plots_dir()

    logger.info(f"Loading CSV: {COMPILED_CSV_PATH}")
    df = pd.read_csv(COMPILED_CSV_PATH)

    # Add run_label combining all experiment dimensions
    df = build_run_labels(df)

    # Compute analytics from CSV data
    logger.info("Computing analytics...")
    analytics = compute_analytics(df)

    # Save analytics JSON
    with open(COMPILED_ANALYTICS_JSON, "w") as f:
        json.dump(analytics, f, indent=2)
    logger.info(f"Analytics saved → {COMPILED_ANALYTICS_JSON}")

    # Generate charts
    logger.info("Rendering charts...")
    dim_chart = chart_dimension_comparison(df)
    chart_paths = {
        "leaderboard":          chart_overall_leaderboard(df),
        "quality_bar":          chart_quality_grouped_bar(df),
        "radar":                chart_radar(df),
        "latency_stacked":      chart_latency_stacked(df),
        "rag_latency":          chart_rag_latency_bar(df),
        "cost_quality":         chart_cost_vs_quality(df),
        "token_usage":          chart_token_usage(df),
        "dimension_comparison": dim_chart,
    }
    logger.info(f"Charts saved → {PLOTS_DIR}")

    # Generate LLM narrative
    narrative = generate_narrative(df, analytics)

    # Build PDF
    logger.info(f"Building PDF → {EXPERIMENT_SUMMARY_PDF}")
    build_pdf(df, analytics, narrative, chart_paths, EXPERIMENT_SUMMARY_PDF)
    logger.info("Done.")


if __name__ == "__main__":
    generate_report()
