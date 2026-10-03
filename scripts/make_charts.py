#!/usr/bin/env python3
"""Generate the README charts from results/consolidated.csv.

Reads the tracked rollup and writes PNGs to charts/. Re-run after any
`--consolidate` to refresh the figures. Requires matplotlib (not part of the
benchmark harness itself, which stays stdlib + PyYAML).

For each (cell, backend) the latest run_id wins, so the early 1-cell smoke
(run-…-1719, ROCm) is superseded by the full ROCm sweep (run-…-1751) — this
matches the headline table in the README.
"""
import collections
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = f"{REPO}/results/consolidated.csv"
OUT = f"{REPO}/charts"

# Order and short labels match the README headline table.
MODEL_ORDER = [
    "gpt-oss-120b-mxfp4",
    "qwen3-coder-30b-a3b-q6_k",
    "qwen3.5-35b-a3b-q6_k",
    "qwen3.8-27b-q5_k_xl",
    "qwen3.8-27b-q6_k",
]
SHORT = {
    "gpt-oss-120b-mxfp4": "gpt-oss-120b\n(MoE · 120B · MXFP4)",
    "qwen3-coder-30b-a3b-q6_k": "Qwen3-Coder-30B-A3B\n(MoE · ~3B active)",
    "qwen3.5-35b-a3b-q6_k": "Qwen3.5-35B-A3B\n(MoE · ~3B active)",
    "qwen3.8-27b-q5_k_xl": "Qwen3.8-27B\n(dense · Q5_K_XL)",
    "qwen3.8-27b-q6_k": "Qwen3.8-27B\n(dense · Q6_K)",
}
BACKENDS = ["vulkan", "rocm"]
COLORS = {"vulkan": "#4C72B0", "rocm": "#DD8452"}
LABELS = {"vulkan": "Vulkan", "rocm": "ROCm"}


def load_latest():
    """(cell, backend) -> metric -> value, preferring the latest run_id."""
    vals = collections.defaultdict(dict)
    run_of = {}
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["cell"], r["backend"])
            if run_of.get(key, "") <= r["run_id"]:
                run_of[key] = r["run_id"]
                vals[key][r["metric"]] = float(r["value"])
    return vals


def grouped_hbar(vals, metric, title, xlabel, fmt):
    models = [m for m in MODEL_ORDER if (m, "vulkan") in vals]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(models))
    height = 0.38
    for i, backend in enumerate(BACKENDS):
        values = [vals[(m, backend)].get(metric, 0.0) for m in models]
        offset = (i - (len(BACKENDS) - 1) / 2) * height
        bars = ax.barh(y + offset, values, height=height,
                       label=LABELS[backend], color=COLORS[backend], zorder=3)
        ax.bar_label(bars, fmt=fmt, padding=2, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in models], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="x", color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    vals = load_latest()

    charts = [
        ("generation_tps.png", "tg512_bench_tps",
         "Generation throughput (llama-bench tg512)", "tok/s", "%.1f"),
        ("prefill_tps.png", "pp2048_bench_tps",
         "Prompt processing (llama-bench pp2048)", "tok/s", "%.0f"),
        ("tokens_per_watt.png", "tok_per_w",
         "Generation efficiency (tokens per watt)", "tok/s per W", "%.2f"),
    ]
    for filename, metric, title, xlabel, fmt in charts:
        fig = grouped_hbar(vals, metric, title, xlabel, fmt)
        path = os.path.join(OUT, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
