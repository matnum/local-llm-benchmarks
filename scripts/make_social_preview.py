#!/usr/bin/env python3
"""Render the 1280x640 GitHub social-preview card.

GitHub wants exactly 2:1 (1280x640) for the repository social preview; the
benchmark charts are tall (one row per model), so they crop badly. This draws a
dedicated card from the same tracked rollup as the README, so the headline
numbers on the card move with the data.

Usage:
    /usr/bin/python3 scripts/make_social_preview.py     # writes .github/social-preview.png
"""
import csv
import collections
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLUP = os.path.join(REPO, "results", "consolidated.csv")
OUT = os.path.join(REPO, ".github", "social-preview.png")

VULKAN = "#4C72B0"
ROCM = "#DD8452"
INK = "#1f2328"
MUTED = "#57606a"


def load():
    vals = collections.defaultdict(dict)
    with open(ROLLUP, newline="") as f:
        for r in csv.DictReader(f):
            vals[(r["cell"], r["backend"])][r["metric"]] = float(r["value"])
    return vals


def pct(vals, cell, metric):
    """Backend gap as a percentage of Vulkan, as the README states it."""
    v = vals[(cell, "vulkan")][metric]
    r = vals[(cell, "rocm")][metric]
    return abs((v - r) / v * 100)


def mem_peak(m):
    """Residency (VRAM + GTT) when the run recorded GTT, else the VRAM pool."""
    return m.get("mem_peak_gib") or m.get("vram_peak_gib", 0.0)


# The README's noise floor for three repeats; a gap below this is a tie.
TIE_PCT = 2.0


def main():
    vals = load()
    cells = sorted({c for c, _ in vals})
    # Range and count over the cells that backend actually wins: a cell inside the
    # noise floor is a tie, not a member of either leading set.
    tg_wins = [c for c in cells
               if vals[(c, "vulkan")]["tg512_bench_tps"] > vals[(c, "rocm")]["tg512_bench_tps"]]
    pp_wins = [c for c in cells
               if vals[(c, "rocm")]["pp2048_bench_tps"] > vals[(c, "vulkan")]["pp2048_bench_tps"]]
    # The range quoted on the card is over decisive wins only, the same convention
    # the README and tests/test_readme_ranges.py use: a 1.3% "win" is a tie and
    # must not set the floor of a headline range.
    tg = [g for g in (pct(vals, c, "tg512_bench_tps") for c in tg_wins) if g >= TIE_PCT]
    pp = [g for g in (pct(vals, c, "pp2048_bench_tps") for c in pp_wins) if g >= TIE_PCT]
    big = max(cells, key=lambda c: mem_peak(vals[(c, "vulkan")]))
    big_tg = vals[(big, "vulkan")]["tg512_bench_tps"]

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Title block
    ax.text(0.06, 0.83, "Local LLM Benchmark Suite", fontsize=34, fontweight="bold",
            color=INK, va="center")
    ax.text(0.06, 0.72, "llama.cpp on a 128 GB AMD Ryzen AI Max+ 395 (Strix Halo)",
            fontsize=16, color=MUTED, va="center")
    ax.plot([0.06, 0.94], [0.655, 0.655], color="#d0d7de", lw=1.2)

    # Three headline numbers
    stats = [
        (f"{big_tg:.0f} tok/s", "gpt-oss-120b (59 GiB)\ngeneration, Vulkan"),
        (f"{len(cells)} × 2", "model/quant cells\n× Vulkan and ROCm"),
        (f"{len(tg_wins)}/{len(cells)} · {len(pp_wins)}/{len(cells)}", "Vulkan wins generation\nROCm wins prefill"),
    ]
    x = 0.06
    for value, label in stats:
        ax.text(x, 0.50, value, fontsize=30, fontweight="bold", color=INK, va="center")
        ax.text(x, 0.34, label, fontsize=12.5, color=MUTED, va="top", linespacing=1.35)
        x += 0.30

    # Range line, straight from the rollup
    ax.text(0.06, 0.12,
            f"Vulkan faster on generation by {min(tg):.1f}–{max(tg):.1f}%  ·  "
            f"ROCm faster on prefill by {min(pp):.1f}–{max(pp):.1f}%",
            fontsize=13.5, color=INK, va="center")

    # Legend swatches
    ax.scatter([0.78], [0.12], s=110, color=VULKAN, zorder=3, edgecolor="white")
    ax.text(0.795, 0.12, "Vulkan", fontsize=11.5, color=MUTED, va="center")
    ax.scatter([0.87], [0.12], s=110, color=ROCM, zorder=3, edgecolor="white")
    ax.text(0.885, 0.12, "ROCm", fontsize=11.5, color=MUTED, va="center")

    ax.text(0.06, 0.035, "Pinned llama.cpp 5b59b83f4 · tracked rollups · numbers checked by the test suite",
            fontsize=10.5, color=MUTED, va="center")
    ax.text(0.94, 0.035, "github.com/matnum/local-llm-benchmarks",
            fontsize=10.5, color=MUTED, va="center", ha="right")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT} (1280x640)")


if __name__ == "__main__":
    main()
