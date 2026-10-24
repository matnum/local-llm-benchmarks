#!/usr/bin/env python3
"""Regenerate the tracked extracts behind the published columns, or verify them.

Raw run dirs are gitignored, so the repo ships three small extracts so a reader can
check a published number without them (independent review, 2026-10-24, item 5.3):

  results/ttft-per-request.csv   per-request TTFT for the cold passes behind the
                                 published column, the repeat passes used for the
                                 run-to-run band, and the warm column it replaced.
  results/bench-logs/            verbatim llama-bench output for the sweeps that own
                                 the published bench columns (`llama-bench -r 3`
                                 keeps no per-repeat values, so the log is the
                                 finest artifact that exists).
  results/sampler-support.csv    per window, the 5 Hz sample count/duration plus the
                                 power/temperature/memory aggregate the rollup copies.
                                 The traces are ~12k rows per sweep; too large to ship.

Usage:
  make_extracts.py            # rewrite the three extracts from results/run-*/
  make_extracts.py --check    # regenerate in memory and diff; exit 1 on drift
  make_extracts.py --stats    # print the numbers docs/RUNNING.md §3c quotes

The run lists below are explicit on purpose: they name the passes the docs cite, and
`tests/test_ttft_extract.py` / `tests/test_bench_extract.py` refuse anything else.
"""
import argparse
import csv
import glob
import io
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cold passes: the two published pairs (2026-10-24 first pass, then the repeat that
# measures run-to-run agreement). Warm: the column replaced on 2026-10-24.
TTFT_RUNS = [
    ("run-20261024-0910", "vulkan"),
    ("run-20261024-0930", "rocm"),
    ("run-20261024-1749", "rocm"),   # repeat pass
    ("run-20261024-1812", "vulkan"),  # repeat pass
    ("run-20261023-1140", "vulkan"),  # warm column
    ("run-20261023-1159", "rocm"),    # warm column
]
# (backend, earlier pass, later pass) — the pairs the run-to-run band compares.
REPEAT_PAIRS = [
    ("vulkan", "run-20261024-0910", "run-20261024-1812"),
    ("rocm", "run-20261024-0930", "run-20261024-1749"),
]
BENCH_RUNS = [("run-20261022-2042", "vulkan"), ("run-20261022-2124", "rocm")]

TTFT_FIELDS = ["run_id", "backend", "cell", "request", "ttft_ms", "gen_tps",
               "erased_tokens", "protocol"]
SUPPORT_FIELDS = ["run_id", "backend", "cell", "window", "n_samples", "duration_s",
                  "power_w_mean", "power_w_max", "temp_c_max", "vram_gib_max",
                  "gtt_gib_max", "mem_gib_max"]


def _csv_text(fields, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def ttft_rows(repo=REPO):
    rows = []
    for run_id, backend in TTFT_RUNS:
        for path in sorted(glob.glob(f"{repo}/results/{run_id}/{backend}/*/ttft.json")):
            with open(path) as f:
                data = json.load(f)
            cell = os.path.basename(os.path.dirname(path))
            for i, r in enumerate(data["requests"], 1):
                rows.append({"run_id": run_id, "backend": backend, "cell": cell,
                             "request": i, "ttft_ms": r["ttft_ms"],
                             "gen_tps": r.get("gen_tps", ""),
                             "erased_tokens": r.get("erased_tokens", ""),
                             "protocol": data.get("protocol") or "warm"})
    return rows


def ttft_text(repo=REPO):
    return _csv_text(TTFT_FIELDS, ttft_rows(repo))


def bench_log_files(repo=REPO):
    """[(run_id, backend, cell, source_path)] for the published bench sweeps."""
    out = []
    for run_id, backend in BENCH_RUNS:
        for path in sorted(glob.glob(f"{repo}/results/{run_id}/{backend}/*/bench.log")):
            out.append((run_id, backend, os.path.basename(os.path.dirname(path)), path))
    return out


def support_rows(repo=REPO):
    rows = []
    for run_id, backend in BENCH_RUNS:
        for cj in sorted(glob.glob(f"{repo}/results/{run_id}/{backend}/*/cell.json")):
            cdir = os.path.dirname(cj)
            with open(cj) as f:
                cell = json.load(f)
            with open(f"{cdir}/phases.json") as f:
                ph = json.load(f)
            with open(f"{cdir}/resources.csv", newline="") as f:
                ts = [float(r["ts"]) for r in csv.DictReader(f)]
            for name, start, end in (("bench_window", ph["bench_start"], ph["bench_end"]),
                                     ("probe_window", ph["probe_start"], ph["probe_end"]),
                                     ("cell_window", ph["bench_start"], ph["done"])):
                w = cell["resources"][name]
                rows.append({"run_id": cell["run"], "backend": cell["backend"],
                             "cell": cell["cell"], "window": name,
                             "n_samples": sum(1 for t in ts if start <= t <= end),
                             "duration_s": round(end - start, 1),
                             "power_w_mean": w.get("power_w_mean"),
                             "power_w_max": w.get("power_w_max"),
                             "temp_c_max": w.get("temp_c_max"),
                             "vram_gib_max": w.get("vram_gib_max"),
                             "gtt_gib_max": w.get("gtt_gib_max", ""),
                             "mem_gib_max": w.get("mem_gib_max", "")})
    return rows


def support_text(repo=REPO):
    return _csv_text(SUPPORT_FIELDS, support_rows(repo))


def _p50_by_cell(rows, run_id, backend, protocol):
    out = {}
    for r in rows:
        if (r["run_id"], r["backend"], r["protocol"]) == (run_id, backend, protocol):
            out.setdefault(r["cell"], []).append(float(r["ttft_ms"]))
    return {c: statistics.median(v) for c, v in out.items()}


def stats(repo=REPO):
    """The numbers §3c cites, recomputed from the extract (not from the run dirs)."""
    rows = ttft_rows(repo)
    print(f"ttft extract: {len(rows)} rows over {len({r['run_id'] for r in rows})} runs")
    for backend in ("vulkan", "rocm"):
        # warm -> cold, on the published (latest) cold pass per backend
        cold_run = [p for b, _, p in REPEAT_PAIRS if b == backend][0]
        warm_run = [r for r, be in TTFT_RUNS if be == backend and r.startswith("run-20261023")][0]
        cold = _p50_by_cell(rows, cold_run, backend, "cold")
        warm = _p50_by_cell(rows, warm_run, backend, "warm")
        deltas = [(cold[c] / warm[c] - 1) * 100 for c in cold if c in warm]
        print(f"{backend}: warm->cold p50 {min(deltas):+.0f}..{max(deltas):+.0f}% "
              f"(published pass {cold_run})")
    for backend, a, b in REPEAT_PAIRS:
        pa = _p50_by_cell(rows, a, backend, "cold")
        pb = _p50_by_cell(rows, b, backend, "cold")
        d = {c: (pb[c] / pa[c] - 1) * 100 for c in pa if c in pb}
        worst = max(d, key=lambda c: abs(d[c]))
        print(f"{backend}: repeat band {min(d.values()):+.1f}..{max(d.values()):+.1f}% "
              f"| median |d| {statistics.median(abs(x) for x in d.values()):.1f}% "
              f"| worst {worst} {d[worst]:+.1f}%")
    # mean/p50 and per-request spread, per protocol, over the published passes
    for protocol in ("warm", "cold"):
        ratios, spreads = [], []
        for run_id, backend in TTFT_RUNS:
            groups = {}
            for r in rows:
                if (r["run_id"], r["protocol"]) == (run_id, protocol):
                    groups.setdefault((backend, r["cell"]), []).append(float(r["ttft_ms"]))
            for vals in groups.values():
                ratios.append(statistics.fmean(vals) / statistics.median(vals))
                spreads.append(max(vals) - min(vals))
        print(f"{protocol}: mean/p50 max {max(ratios):.2f}x | spread median "
              f"{statistics.median(spreads):.0f} ms, max {max(spreads):.0f} ms")


def main():
    ap = argparse.ArgumentParser(
        description="Regenerate the tracked extracts behind the published columns.")
    ap.add_argument("--check", action="store_true",
                    help="diff the tracked extracts against a regeneration; exit 1 on drift")
    ap.add_argument("--stats", action="store_true",
                    help="print the per-protocol statistics docs/RUNNING.md §3c quotes")
    args = ap.parse_args()

    if not glob.glob(f"{REPO}/results/run-*"):
        raise SystemExit("no run dirs here — the extracts can only be built on the machine "
                         "that measured the runs")

    if args.stats:
        stats()
        return

    targets = {
        f"{REPO}/results/ttft-per-request.csv": ttft_text,
        f"{REPO}/results/sampler-support.csv": support_text,
    }
    if args.check:
        problems = []
        for path, build in targets.items():
            try:
                # newline="" so the CRLF the csv writer emits is not translated away
                with open(path, newline="") as f:
                    tracked = f.read()
            except OSError:
                problems.append(f"missing: {path}")
                continue
            if tracked != build():
                problems.append(f"stale: {path}")
        for run_id, backend, cell, path in bench_log_files():
            dst = f"{REPO}/results/bench-logs/{run_id}/{backend}/{cell}.log"
            try:
                with open(dst, newline="") as f:
                    have = f.read()
            except OSError:
                problems.append(f"missing: {dst}")
                continue
            with open(path, newline="") as f:
                if have != f.read():
                    problems.append(f"stale: {dst}")
        for p in problems:
            print(p)
        print("extracts up to date" if not problems else f"{len(problems)} problem(s)")
        sys.exit(1 if problems else 0)

    for path, build in targets.items():
        with open(path, "w", newline="") as f:
            f.write(build())
        print("wrote", os.path.relpath(path, REPO))
    for run_id, backend, cell, src in bench_log_files():
        dst_dir = f"{REPO}/results/bench-logs/{run_id}/{backend}"
        os.makedirs(dst_dir, exist_ok=True)
        with open(src) as f:
            text = f.read()
        with open(f"{dst_dir}/{cell}.log", "w") as f:
            f.write(text)
    print(f"wrote results/bench-logs/ ({len(bench_log_files())} logs)")


if __name__ == "__main__":
    main()
