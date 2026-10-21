#!/usr/bin/env python3
"""Generate the README charts from results/consolidated.csv.

Reads the tracked rollup and writes PNGs to charts/. Re-run after any
`--consolidate` to refresh the figures. Requires matplotlib and PyYAML (neither
is part of the benchmark harness itself, which stays stdlib + PyYAML).

For each (cell, backend) the latest run_id wins, so the early 1-cell smoke
(run-…-1719, ROCm) is superseded by the full ROCm sweep (run-…-1751).

Design notes — why dumbbells rather than grouped bars:
  * one row per model, with the Vulkan and ROCm results joined by a line, so the
    pair *and its gap* read directly instead of being reconstructed from two bars;
  * the model list, its order and its labels all come from the matrix and the
    rollup (never a hardcoded list), so a newly swept cell appears without editing
    this file — ordering by model size also shows the throughput/size relationship;
  * generation is plotted against the dense-model bandwidth ceiling
    (256 GB/s ÷ model size, README "ceiling rule of thumb"), which is what makes
    the MoE cells' sub-ceiling behaviour visible rather than just "bigger number".
"""
import collections
import csv
import hashlib
import os
import struct

import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = f"{REPO}/results/consolidated.csv"
MATRIX = f"{REPO}/matrix.yaml"
OUT = f"{REPO}/charts"

BACKENDS = ["vulkan", "rocm"]
COLORS = {"vulkan": "#4C72B0", "rocm": "#DD8452"}
LABELS = {"vulkan": "Vulkan", "rocm": "ROCm"}
CEILING_COLOR = "#666666"
CEILING_BW = 256.0          # GB/s, README: memory bandwidth of the box
METRIC_COLORS = {"tg512_bench_tps": "#8172B2", "pp2048_bench_tps": "#64B5CD"}
METRIC_LABELS = {"tg512_bench_tps": "generation (tg512)",
                 "pp2048_bench_tps": "prefill (pp2048)"}

# Each figure carries a fingerprint of what it was drawn from, as a PNG tEXt
# chunk. Regenerating the rollup without re-running this script changes the
# fingerprint and `tests/test_charts.py` fails — the data-derived list test
# alone cannot tell a stale committed PNG from a fresh one.
STAMP_KEY = "chart-stamp"


def chart_stamp(cells, vals, metrics):
    """Fingerprint of a figure's data: every plotted cell and both values.

    Order-independent on purpose: the efficiency chart plots a different row
    order from the size-ordered ones, and row order already follows the values.
    """
    h = hashlib.sha256()
    for cell in sorted(cells):
        for metric in metrics:
            h.update(f"{cell}\t{metric}\t{vals[(cell, 'vulkan')][metric]:.6f}\t"
                     f"{vals[(cell, 'rocm')][metric]:.6f}\n".encode())
    return f"cells={len(cells)};sha256={h.hexdigest()[:16]}"


def png_stamp(path):
    """Read back the chart-stamp written into a PNG, or None if it has none."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 8  # past the 8-byte PNG signature
    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"tEXt":
            key, _, value = body.partition(b"\x00")
            if key.decode("latin-1") == STAMP_KEY:
                return value.decode("latin-1")
        pos += 12 + length
    return None


def load_matrix():
    """cell id -> {model, quant, size_gib, arch} from the frozen matrix."""
    with open(MATRIX) as f:
        m = yaml.safe_load(f)
    return {c["id"]: c for c in m["cells"]}


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


def comparable_cells(vals):
    """Cells measured on *every* backend, ordered by model size then cell id.

    A cell missing a backend is skipped rather than plotted against a guess —
    plotting one-sided rows would silently imply a comparison that was never run.
    """
    by_cell = collections.defaultdict(set)
    for cell, backend in vals:
        by_cell[cell].add(backend)
    complete = [c for c, bs in by_cell.items() if set(BACKENDS) <= bs]
    meta = load_matrix()
    return sorted(complete, key=lambda c: (meta.get(c, {}).get("size_gib", 0), c))


def row_label(cell, meta):
    m = meta.get(cell, {})
    name = m.get("model", cell)
    bits = [b for b in (m.get("quant"), (f"{m['size_gib']:g} GiB" if "size_gib" in m else None)) if b]
    tier = m.get("arch")
    return f"{name}\n{' · '.join(bits)}" + (f"\n{tier}" if tier in ("moe", "dense") else "")


# The two backends sit slightly above/below the row centre. Without the offset a
# coincident pair (Poro 2 8B is 0.31 tok/s/W on both backends; Qwen3-32B 0.11 on
# both) is drawn one dot on top of the other and the lower series disappears —
# which reads as "only one backend was measured".
ROW_OFFSET = 0.13


def dumbbell(ax, vals, cells, meta, metric, fmt):
    """One row per model: Vulkan and ROCm dots joined, values printed per row."""
    y = np.arange(len(cells))
    for i, cell in enumerate(cells):
        pair = [(b, vals[(cell, b)][metric]) for b in BACKENDS if metric in vals.get((cell, b), {})]
        if len(pair) < 2:
            continue
        pts = [(v, i + (ROW_OFFSET if b == "vulkan" else -ROW_OFFSET)) for b, v in pair]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#999999", lw=2.0,
                zorder=2, solid_capstyle="round")
        for (v, yy), (b, _) in zip(pts, pair):
            ax.scatter([v], [yy], s=50, color=COLORS[b], zorder=3,
                       label=LABELS[b] if i == 0 else None, edgecolor="white", linewidth=0.8)
        xs = [v for _, v in pair]
        ax.annotate(f"V {fmt % pair[0][1]} · R {fmt % pair[1][1]}", (max(xs), i),
                    textcoords="offset points", xytext=(7, -3), fontsize=7.5,
                    color="#444444")

    # Dense cells: mark the bandwidth ceiling they are physically bounded by.
    first = True
    for i, cell in enumerate(cells):
        if meta.get(cell, {}).get("arch") != "dense":
            continue
        size = meta[cell].get("size_gib")
        if not size:
            continue
        ax.scatter([CEILING_BW / size], [i + ROW_OFFSET], marker="x", s=46,
                   color=CEILING_COLOR, zorder=4, linewidths=1.8,
                   label="dense ceiling (256 GB/s ÷ size)" if first else None)
        first = False

    ax.set_yticks(y)
    ax.set_yticklabels([row_label(c, meta) for c in cells], fontsize=8)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.22)
    ax.invert_yaxis()


def finish(ax, title, xlabel, legend_loc="lower right"):
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="x", color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)


def chart_metric(vals, cells, meta, metric, title, xlabel, fmt, filename):
    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(cells) + 1.5))
    dumbbell(ax, vals, cells, meta, metric, fmt)
    finish(ax, title, xlabel)
    path = os.path.join(OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                metadata={STAMP_KEY: chart_stamp(cells, vals, (metric,))})
    plt.close(fig)
    print(f"wrote {path} ({len(cells)} cells)")
    return path


def chart_delta(vals, cells, meta, filename):
    """ROCm vs Vulkan, per model, for both metrics — the split in one picture."""
    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(cells) + 1.5))
    y = np.arange(len(cells))
    height = 0.36
    for i, metric in enumerate(("tg512_bench_tps", "pp2048_bench_tps")):
        deltas = []
        for cell in cells:
            v = vals[(cell, "vulkan")][metric]
            r = vals[(cell, "rocm")][metric]
            deltas.append((r - v) / v * 100)
        offset = (i - 0.5) * height
        bars = ax.barh(y + offset, deltas, height=height,
                       color=METRIC_COLORS[metric], label=METRIC_LABELS[metric], zorder=3)
        for bar, d in zip(bars, deltas):
            ax.annotate(f"{d:+.1f}%", (d, bar.get_y() + bar.get_height() / 2),
                        textcoords="offset points",
                        xytext=(5 if d >= 0 else -5, -3),
                        ha="left" if d >= 0 else "right", fontsize=7.5, color="#444444")

    ax.axvline(0, color="#333333", lw=1.1, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([row_label(c, meta) for c in cells], fontsize=8)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 2, hi + 2)
    ax.invert_yaxis()
    finish(ax, "Backend difference per model", "ROCm relative to Vulkan (%)\n"
                                               "positive = ROCm faster · negative = Vulkan faster",
           legend_loc="lower right")
    path = os.path.join(OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                metadata={STAMP_KEY: chart_stamp(
                    cells, vals, ("tg512_bench_tps", "pp2048_bench_tps"))})
    plt.close(fig)
    print(f"wrote {path} ({len(cells)} cells)")
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    vals = load_latest()
    meta = load_matrix()
    cells = comparable_cells(vals)
    if not cells:
        raise SystemExit("no cell has results on both backends — nothing to plot")

    # Two orderings: size (shows the throughput/size relationship) and efficiency.
    eff_order = sorted(cells, key=lambda c: -vals[(c, "vulkan")].get("tok_per_w", 0))

    chart_metric(vals, cells, meta, "tg512_bench_tps",
                 "Generation throughput (llama-bench tg512)", "tok/s", "%.1f",
                 "generation_tps.png")
    chart_metric(vals, cells, meta, "pp2048_bench_tps",
                 "Prompt processing (llama-bench pp2048)", "tok/s", "%.0f",
                 "prefill_tps.png")
    chart_metric(vals, eff_order, meta, "tok_per_w",
                 "Generation efficiency (tokens per watt)", "tok/s per W", "%.2f",
                 "tokens_per_watt.png")
    chart_delta(vals, cells, meta, "backend_delta.png")


if __name__ == "__main__":
    main()
