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
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-3-27b-it",
    "microsoft/Phi-4-multimodal-instruct",
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
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping chart. Run: pip install matplotlib numpy")
        return

    # Sort models by avg score ascending (so best is at top)
    sorted_models = sorted(results.items(), key=lambda kv: avg_score(kv[1]))
    labels = [m for m, _ in sorted_models]
    avgs = [avg_score(s) for _, s in sorted_models]

    # Colour palette — distinct per bar
    COLORS = [
        "#4C72B0", "#2e8b57", "#3d9970", "#4682b4", "#8B6914",
        "#C44E52", "#8172B3", "#DA8BC3", "#cc5500", "#4e9a9a",
        "#888888", "#b5651d",
    ]

    fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.72)))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="#cccccc")
    ax.xaxis.label.set_color("#cccccc")

    y = np.arange(len(labels))
    bars = ax.barh(
        y, avgs,
        color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
        edgecolor="#1a1a1a",
        linewidth=0.6,
        height=0.65,
    )

    # Score labels on bars
    for bar, score in zip(bars, avgs):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center", ha="left",
            fontsize=9, color="#cccccc",
        )

    ax.axvline(0.5, color="#ffd700", linestyle="--", linewidth=0.9, alpha=0.55, label="0.5 baseline")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color="#cccccc")
    ax.set_xlabel("Average Episode Score  (0, 1)  — higher is better", fontsize=11, color="#cccccc")
    ax.set_title(
        "Macro Signal Engine — LLM Benchmark (avg across 3 tasks)",
        fontsize=13, color="#00d4ff", pad=12, fontweight="bold",
    )
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="x", color="#333", linewidth=0.5, linestyle="--")

    ax.legend(loc="lower right", framealpha=0.3, facecolor="#1e3a5f",
              edgecolor="#00d4ff", labelcolor="#cccccc", fontsize=9)

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
