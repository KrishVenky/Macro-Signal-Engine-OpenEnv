"""
test.py — LLM Benchmark Runner for Macro Signal Engine
=======================================================
Runs inference across one or more model configurations, records per-task scores,
and writes assets/llm_comparison.png (horizontal bar chart, sorted by avg score).

All models use the HuggingFace router (free tier) by default.

Usage
-----
# Run the default free HF model list, then plot
HF_TOKEN=hf_xxx python test.py

# Run specific models
python test.py --models Qwen/Qwen2.5-72B-Instruct,meta-llama/Llama-3.3-70B-Instruct

# Skip running; just regenerate the chart from saved results
python test.py --plot-only

Environment variables
---------------------------------------------
  HF_TOKEN      HuggingFace API token (required for running models)
  API_BASE_URL  LLM endpoint (default: https://router.huggingface.co/v1)
  ENV_URL       Space URL    (default: https://krishvenky-macro-signal-env.hf.space)

Results are accumulated in assets/results.json so you can run models
incrementally without losing earlier data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
ASSETS_DIR = ROOT / "assets"
RESULTS_FILE = ASSETS_DIR / "results.json"
CHART_FILE = ASSETS_DIR / "llm_comparison.png"
INFERENCE_SCRIPT = ROOT / "inference.py"

ASSETS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
TASKS = ["single_event", "regime_shift", "causal_chain"]

# ---------------------------------------------------------------------------
# Free HF router models to benchmark by default
# ---------------------------------------------------------------------------
DEFAULT_MODELS = [
    # --- Large frontier (same tier as that guy's benchmark) ---
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "moonshotai/Kimi-K2-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "google/gemma-3-27b-it",
    # --- Smaller / faster ---
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-4-multimodal-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    # --- Finance-specialized ---
    "AdaptLLM/finance-chat",                    # LLaMA fine-tuned on finance corpus
    "TheFinAI/finma-7b-nlp",                    # FinMA — FLARE benchmark finance LLM
    "INTERNLM/internlm2_5-7b-finance",          # InternLM finance variant
]

# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def clamp_open_score(value: float, low: float = 0.01, high: float = 0.99, default: float = 0.5) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    if not math.isfinite(v):
        v = default
    return max(low, min(high, v))


def avg_score(task_scores: Dict[str, float]) -> float:
    vals = [clamp_open_score(task_scores.get(t, 0.5)) for t in TASKS]
    return sum(vals) / len(vals)


def parse_scores_from_output(stdout: str) -> Dict[str, float]:
    """Extract {task: score} from the stdout of inference.py."""
    scores: Dict[str, float] = {}
    current_task: str | None = None
    for line in stdout.splitlines():
        start = re.search(r"\[START\]\s+task=(\S+)", line)
        if start:
            current_task = start.group(1)

        end = re.search(r"\[END\].*\btask=(\S+).*\bscore=([\d.]+)", line)
        if end:
            scores[end.group(1)] = clamp_open_score(float(end.group(2)))
            continue

        end2 = re.search(r"\[END\].*\bscore=([\d.]+)", line)
        if end2 and current_task:
            scores[current_task] = clamp_open_score(float(end2.group(1)))

    return scores


# ---------------------------------------------------------------------------
# Running inference
# ---------------------------------------------------------------------------

def run_model(model_name: str, env_url: str, hf_token: str, api_base: str) -> Dict[str, float]:
    """Run inference.py for a single model; return {task: score}."""
    env = {
        **os.environ,
        "MODEL_NAME": model_name,
        "HF_TOKEN": hf_token,
        "API_BASE_URL": api_base,
        "ENV_URL": env_url,
    }
    print(f"\n{'='*60}")
    print(f"Running model: {model_name}")
    print(f"{'='*60}")

    # Stream output to terminal
    subprocess.run([sys.executable, str(INFERENCE_SCRIPT)], env=env, capture_output=False, text=True)

    # Re-run silently to capture and parse scores
    proc2 = subprocess.run(
        [sys.executable, str(INFERENCE_SCRIPT)],
        env=env, capture_output=True, text=True,
    )
    scores = parse_scores_from_output(proc2.stdout + proc2.stderr)

    for task in TASKS:
        if task not in scores:
            scores[task] = 0.5
    return scores


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_results() -> Dict[str, Dict[str, float]]:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: Dict[str, Dict[str, float]]) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Chart generation — horizontal bars sorted by avg score
# ---------------------------------------------------------------------------

def generate_chart(results: Dict[str, Dict[str, float]]) -> None:
    if not results:
        print("No results to chart yet. Run with HF_TOKEN set to benchmark models.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping chart. Run: pip install matplotlib numpy")
        return

    # Categorise models for colour coding
    FINANCE_MODELS = {"AdaptLLM/finance-chat", "TheFinAI/finma-7b-nlp", "INTERNLM/internlm2_5-7b-finance"}
    SMALL_MODELS   = {"meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
                      "microsoft/Phi-4-multimodal-instruct"}
    REFERENCE      = {"gpt-4o (measured)"}

    def model_color(name: str) -> str:
        if name in FINANCE_MODELS:   return "#f0a500"   # gold  — finance-tuned
        if name in REFERENCE:        return "#888888"   # grey  — proprietary reference
        if name in SMALL_MODELS:     return "#5f9ea0"   # muted blue — small/fast
        return "#4C9BE8"                                # bright blue — large open

    # Sort by avg score ascending so best is at top of horizontal chart
    sorted_models = sorted(results.items(), key=lambda kv: avg_score(kv[1]))
    labels = [m for m, _ in sorted_models]
    avgs   = [avg_score(s) for _, s in sorted_models]
    task_scores = {
        task: [clamp_open_score(s.get(task, 0.5)) for _, s in sorted_models]
        for task in TASKS
    }

    n = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, n * 0.70)),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    fig.patch.set_facecolor("#1a1a1a")

    # ---- LEFT: overall avg horizontal bar chart ----
    ax = axes[0]
    ax.set_facecolor("#1a1a1a")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.tick_params(colors="#cccccc")

    y = np.arange(n)
    colors = [model_color(m) for m in labels]
    bars = ax.barh(y, avgs, color=colors, edgecolor="#1a1a1a", linewidth=0.5, height=0.62)

    for bar, score in zip(bars, avgs):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", ha="left", fontsize=9, color="#dddddd")

    ax.axvline(0.5, color="#ffd700", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_yticks(y)
    short_labels = [lbl.split("/")[-1] for lbl in labels]
    ax.set_yticklabels(short_labels, fontsize=9.5, color="#cccccc")
    ax.set_xlabel("Avg Episode Score  (0, 1)", fontsize=10, color="#cccccc", labelpad=6)
    ax.set_title("Overall Average", fontsize=11, color="#00d4ff", pad=8, fontweight="bold")
    ax.set_xlim(0, 1.13)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="x", color="#2a2a2a", linewidth=0.6, linestyle="--")

    # Legend: categories
    legend_patches = [
        mpatches.Patch(color="#4C9BE8", label="Large open-source"),
        mpatches.Patch(color="#5f9ea0", label="Small / fast"),
        mpatches.Patch(color="#f0a500", label="Finance-specialized"),
        mpatches.Patch(color="#888888", label="Proprietary reference"),
        plt.Line2D([0], [0], color="#ffd700", linestyle="--", linewidth=1.2, label="0.5 baseline"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.25,
              facecolor="#111", edgecolor="#444", labelcolor="#cccccc", fontsize=8)

    # ---- RIGHT: per-task score heat-style bar chart ----
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a1a")
    for spine in ax2.spines.values():
        spine.set_color("#333")
    ax2.tick_params(colors="#cccccc")

    task_colors = {"single_event": "#4CAF50", "regime_shift": "#2196F3", "causal_chain": "#F44336"}
    task_display = {"single_event": "Single Event (Easy)", "regime_shift": "Regime Shift (Med)", "causal_chain": "Causal Chain (Hard)"}
    bar_h = 0.20
    offsets = [-bar_h, 0, bar_h]

    for i, (task, offset) in enumerate(zip(TASKS, offsets)):
        ax2.barh(y + offset, task_scores[task],
                 height=bar_h * 0.88,
                 color=task_colors[task],
                 edgecolor="#1a1a1a", linewidth=0.4,
                 label=task_display[task], alpha=0.88)

    ax2.axvline(0.5, color="#ffd700", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(short_labels, fontsize=9.5, color="#cccccc")
    ax2.set_xlabel("Score per Task", fontsize=10, color="#cccccc", labelpad=6)
    ax2.set_title("Per-Task Breakdown", fontsize=11, color="#00d4ff", pad=8, fontweight="bold")
    ax2.set_xlim(0, 1.05)
    ax2.grid(axis="x", color="#2a2a2a", linewidth=0.6, linestyle="--")
    ax2.legend(loc="lower right", framealpha=0.25, facecolor="#111",
               edgecolor="#444", labelcolor="#cccccc", fontsize=8)

    fig.suptitle(
        "Macro Signal Engine — LLM Benchmark",
        fontsize=14, color="#00d4ff", fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()
    print(f"Chart saved -> {CHART_FILE}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLMs on Macro Signal Engine")
    parser.add_argument(
        "--models", default="",
        help="Comma-separated model names to run. Leave empty to use DEFAULT_MODELS list.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip inference runs; regenerate chart from saved results.json",
    )
    args = parser.parse_args()

    results = load_results()

    if not args.plot_only:
        hf_token = os.getenv("HF_TOKEN", "")
        if not hf_token:
            print("HF_TOKEN not set — skipping inference runs, plotting existing data.")
        else:
            api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
            env_url = os.getenv("ENV_URL", "https://krishvenky-macro-signal-env.hf.space")
            models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
            if not models_to_run:
                models_to_run = DEFAULT_MODELS

            for model in models_to_run:
                scores = run_model(model, env_url, hf_token, api_base)
                results[model] = scores
                print(f"  {model}: avg={avg_score(scores):.3f}  {scores}")

            save_results(results)

    generate_chart(results)


if __name__ == "__main__":
    main()
