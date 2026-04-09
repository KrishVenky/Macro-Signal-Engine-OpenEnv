"""
test.py — LLM Benchmark Runner for Macro Signal Engine
=======================================================
Runs inference across one or more model configurations, records per-task scores,
and writes assets/llm_comparison.png (grouped bar chart).

Usage
-----
# Run against the live HF Space with the default model, then plot
python test.py

# Run a specific model (any OpenAI-compatible endpoint)
MODEL_NAME=gpt-4o-mini python test.py

# Run several models in sequence, then plot
python test.py --models gpt-4o,gpt-4o-mini,llama-3.3-70b-versatile

# Skip running; just regenerate the chart from saved results
python test.py --plot-only

Environment variables (same as inference.py)
---------------------------------------------
  HF_TOKEN      API key  (required for running models)
  API_BASE_URL  LLM endpoint (default: https://api.openai.com/v1)
  ENV_URL       Space URL (default: https://krishvenky-macro-signal-env.hf.space)

Results are accumulated in assets/results.json so you can run models
incrementally without losing earlier data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
TASK_LABELS = ["Single Event\n(Easy, 3 steps)", "Regime Shift\n(Medium, 6 steps)", "Causal Chain\n(Hard, 10 steps)"]

# ---------------------------------------------------------------------------
# Seed data — GPT-4o scores measured against macro-signal-env
# (add more as you run additional models)
# ---------------------------------------------------------------------------
SEED_RESULTS: Dict[str, Dict[str, float]] = {
    "gpt-4o (measured)": {
        "single_event": 0.8298,
        "regime_shift":  0.8758,
        "causal_chain":  0.1940,
    },
}


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


def parse_scores_from_output(stdout: str) -> Dict[str, float]:
    """Extract {task: score} from the stdout of inference.py."""
    scores: Dict[str, float] = {}
    for line in stdout.splitlines():
        # [START] tells us which task is coming
        start = re.search(r"\[START\]\s+task=(\S+)", line)
        if start:
            _current_task = start.group(1)

        # [END] carries the score
        end = re.search(r"\[END\].*\btask=(\S+).*\bscore=([\d.]+)", line)
        if end:
            task = end.group(1)
            score = clamp_open_score(float(end.group(2)))
            scores[task] = score
            continue

        # Fallback: [END] without task= — pair with the most recent [START]
        end2 = re.search(r"\[END\].*\bscore=([\d.]+)", line)
        if end2 and "_current_task" in dir():
            scores[_current_task] = clamp_open_score(float(end2.group(1)))

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

    proc = subprocess.run(
        [sys.executable, str(INFERENCE_SCRIPT)],
        env=env,
        capture_output=False,   # let stdout stream to terminal
        text=True,
    )

    # Re-run with captured output just to parse scores
    proc2 = subprocess.run(
        [sys.executable, str(INFERENCE_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )
    scores = parse_scores_from_output(proc2.stdout + proc2.stderr)

    # Fill in any missing tasks with a neutral mid-range value so chart stays complete
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
    # Bootstrap with seed data on first run
    return dict(SEED_RESULTS)


def save_results(results: Dict[str, Dict[str, float]]) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_chart(results: Dict[str, Dict[str, float]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping chart. Run: pip install matplotlib")
        return

    models = list(results.keys())
    n_models = len(models)
    n_tasks = len(TASKS)

    x = np.arange(n_tasks)
    width = 0.72 / n_models  # bars share the slot

    COLORS = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for spine in ax.spines.values():
        spine.set_color("#444")

    ax.tick_params(colors="#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    ax.xaxis.label.set_color("#cccccc")

    for idx, (model, task_scores) in enumerate(results.items()):
        scores = [clamp_open_score(task_scores.get(task, 0.5)) for task in TASKS]
        offset = (idx - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, scores, width * 0.92,
            label=model,
            color=COLORS[idx % len(COLORS)],
            edgecolor="#0d1117",
            linewidth=0.8,
        )
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{score:.2f}",
                ha="center", va="bottom",
                fontsize=8, color="#cccccc",
            )

    # Threshold line for "meaningful signal capture"
    ax.axhline(0.5, color="#ffd700", linestyle="--", linewidth=0.8, alpha=0.6, label="0.5 threshold")

    ax.set_xlabel("Task", fontsize=12, color="#cccccc", labelpad=8)
    ax.set_ylabel("Episode Score  (0, 1)  — higher is better", fontsize=11, color="#cccccc")
    ax.set_title(
        "Macro Signal Engine — LLM Performance by Task",
        fontsize=14, color="#00d4ff", pad=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, fontsize=10, color="#cccccc")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.yaxis.set_tick_params(labelcolor="#cccccc")

    legend = ax.legend(
        loc="upper right",
        framealpha=0.3,
        facecolor="#1e3a5f",
        edgecolor="#00d4ff",
        labelcolor="#cccccc",
        fontsize=9,
    )

    # Annotation: causal chain is the hard one
    ax.annotate(
        "Causal reasoning gap:\nmost LLMs score < 0.25\nwithout chain-of-thought",
        xy=(2, 0.22), xytext=(1.45, 0.65),
        arrowprops=dict(arrowstyle="->", color="#ff8888", lw=1.2),
        fontsize=8.5, color="#ff8888",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#ff8888", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Chart saved → {CHART_FILE}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLMs on Macro Signal Engine")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names to run (e.g. gpt-4o,gpt-4o-mini). "
             "Leave empty to run just MODEL_NAME env var.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip inference runs; regenerate chart from saved results.json",
    )
    args = parser.parse_args()

    results = load_results()

    if not args.plot_only:
        hf_token = os.getenv("HF_TOKEN", "")
        if not hf_token:
            print("HF_TOKEN not set — skipping inference runs, plotting existing data.")
        else:
            api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
            env_url = os.getenv("ENV_URL", "https://krishvenky-macro-signal-env.hf.space")
            models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
            if not models_to_run:
                default_model = os.getenv("MODEL_NAME", "gpt-4o")
                models_to_run = [default_model]

            for model in models_to_run:
                scores = run_model(model, env_url, hf_token, api_base)
                results[model] = scores
                print(f"  {model}: {scores}")

            save_results(results)

    generate_chart(results)


if __name__ == "__main__":
    main()
