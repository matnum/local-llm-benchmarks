#!/usr/bin/env python3
"""One-command benchmark sweep driver.

Usage:
  run_sweep.py --backend {vulkan,rocm} [--cells id1,id2] [--repeats 3]
               [--pp-tokens 512,2048] [--tg-tokens 512] [--port 11500]
               [--ctx 8192] [--threads 16] [--ngl 99] [--warmup 1]
               [--power-hz 5] [--force] [--rocm-override]
  run_sweep.py --consolidate        # rebuild the rollups from existing run dirs

Run-parameter defaults come from matrix.yaml `defaults` (single source of truth);
the flags above override them per run.

A sweep writes <run>/<backend>/sweep.log (stdout, argv and any crash traceback),
so an interrupted run explains itself instead of leaving run.json listing cells
that have no artifacts.

Per cell: idle-check GPU -> start resource sampler -> llama-bench (pp/tg)
-> llama-server + TTFT probe -> aggregate to results/run-*/<cell>/cell.json.
After all cells: update results/consolidated.csv and render results/table.md.

Contamination rule: the GPU must be idle (busy < 5%, VRAM < 4 GiB) unless
--force. Stop any running llama-server before real sweeps.
stdlib + PyYAML only.
"""
import argparse
import csv
import glob
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback

try:
    import yaml
except ImportError as _e:  # M4: PyYAML is required; the shebang's python3 may lack it.
    sys.stderr.write(
        "run_sweep.py: PyYAML is required but not importable "
        f"({_e}).\n"
        "  Run with a python that has it (e.g. /usr/bin/python3), or:\n"
        "    pip install PyYAML   # or: pip install -r requirements.txt\n"
    )
    raise

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CPP = os.path.expanduser("~/.local/opt/llama.cpp")
BUILDS = {"vulkan": f"{LLAMA_CPP}/build-vulkan/bin", "rocm": f"{LLAMA_CPP}/build/bin"}
GIB = 1024 ** 3
IDLE_BUSY_PCT = 5
IDLE_MEM_GIB = 4.0      # resident GPU memory budget (VRAM + GTT): see gpu_mem()
IDLE_SETTLE_S = 90      # how long to let the GPU settle before declaring it busy


LOG_F = None


def log(msg):
    """Print to stdout and tee into the sweep log, when one is open."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_F is not None:
        LOG_F.write(line + "\n")
        LOG_F.flush()


def open_sweep_log(path):
    """Tee log() into <run dir>/sweep.log (apples to the same run on retry)."""
    global LOG_F
    LOG_F = open(path, "a")
    log(f"sweep log -> {path}")


def _excepthook(exc_type, exc, tb):
    # A crash mid-sweep must land in sweep.log, not only on a terminal that may
    # be long gone (run.json is written up-front and lists cells that never ran).
    log("ABORTED with unhandled exception:\n  " + "".join(
        traceback.format_exception(exc_type, exc, tb)).rstrip().replace("\n", "\n  "))
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _excepthook


def rd(path, default=0):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return default


def read_text(path, default="unknown"):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return default


def cpu_power_state():
    """The CPU power state a run was measured under.

    A measurement condition, not a detail: the reference machine runs
    amd_pstate=active with
    governor=powersave and a firmware-chosen EPP, and nothing pins them. Decision
    2026-10-02 was to leave the default state rather than force `performance`, so
    each run has to state what it measured under — otherwise a sweep taken after
    a power-profile change is silently incomparable with one taken before.
    """
    return {
        "governor": read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "epp": read_text("/sys/devices/system/cpu/cpu0/cpufreq/"
                         "energy_performance_preference"),
        "pstate_driver": read_text("/sys/devices/system/cpu/amd_pstate/status"),
    }


def gpu_mem():
    """Resident GPU memory in GiB: (vram, gtt, total).

    M-1: on a small UMA carve-out the weights live in GTT, so a foreign
    `llama-server` shows up in `mem_info_gtt_used` while `mem_info_vram_used`
    stays at its idle floor. A gate that reads VRAM alone would wave a loaded GPU
    through and silently contaminate every cell of the sweep.
    """
    vram = rd("/sys/class/drm/card0/device/mem_info_vram_used") / GIB
    gtt_raw = rd("/sys/class/drm/card0/device/mem_info_gtt_used")
    gtt = 0.0 if gtt_raw is None else gtt_raw / GIB
    return vram, gtt, vram + gtt


def gpu_idle():
    busy = rd("/sys/class/drm/card0/device/gpu_busy_percent")
    _vram, _gtt, total = gpu_mem()
    return busy < IDLE_BUSY_PCT and total < IDLE_MEM_GIB, busy, total


def wait_idle(tag, timeout=IDLE_SETTLE_S, poll_s=2.0):
    """Poll until the GPU is idle; return (ok, busy, mem_gib).

    A single instantaneous read is not enough. Tearing down a large model leaves
    gpu_busy near 95% for several seconds *after* its memory has been released —
    the teardown is itself GPU work and the busy counter lags it. Judging on the
    instant right after `probe_cell` kills a 59 GiB server hard-failed the next
    cell of a sweep (run-20261002-1731 died between cells 1 and 2 with
    busy=93%, vram=0.1 GiB). Waiting for a settled GPU keeps the contamination
    guarantee — it never proceeds while the GPU is busy — without mistaking a
    cell's own teardown for foreign load.
    """
    t0 = time.time()
    while True:
        ok, busy, mem = gpu_idle()
        if ok:
            waited = time.time() - t0
            if waited >= poll_s:
                log(f"[{tag}] GPU settled after {waited:.1f}s "
                    f"(busy={busy}%, vram+gtt={mem:.1f} GiB)")
            return True, busy, mem
        if time.time() - t0 >= timeout:
            return False, busy, mem
        time.sleep(poll_s)


def killpg(proc):
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


def parse_bench(stdout):
    """Parse llama-bench markdown-table stdout -> {test: {"mean": float, "stddev": float|None}}.

    Extracted from run_bench so it can be tested against a captured golden table
    without invoking the binary. The pinned build prints one row per test as
    "mean ± stddev" (both fields present only when -r > 1); a bare number yields
    stddev None.
    """
    tests = {}
    for line in stdout.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        for i, c in enumerate(cells):
            if re.fullmatch(r"(pp|tg)\d+", c) and i + 1 < len(cells):
                # llama-bench -r N prints ONE row as "mean ± stddev" (it does the
                # N runs itself). Keep both numbers — taking only the first threw
                # away the run's sole dispersion measurement.
                m = re.match(r"([\d.]+)(?:\s*±\s*([\d.]+))?", cells[i + 1])
                if m:
                    tests[c] = {"mean": float(m.group(1)),
                                "stddev": float(m.group(2)) if m.group(2) else None}
    if not tests:
        raise RuntimeError("llama-bench produced no parseable rows")
    return tests


def run_bench(binpath, model, repeats, p, n, logf, env=None, threads=16, ngl=99):
    """One llama-bench invocation; returns {test: {"mean", "stddev"}}.

    NOTE (B1): the pinned llama.cpp build (5b59b83f4) llama-bench has NO -c/--ctx
    flag, so the bench runs at the model's default context (4096), NOT the sweep's
    requested --ctx. The effective context is recorded separately in run.json
    (bench_ctx) so the data is never silently mislabeled.
    """
    cmd = [f"{binpath}/llama-bench", "-m", model,
           "-ngl", str(ngl), "-t", str(threads), "-p", str(p), "-n", str(n),
           "-r", str(repeats)]
    log("  ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    logf.write(out.stdout + "\n--- stderr ---\n" + out.stderr + "\n")
    logf.flush()
    return parse_bench(out.stdout)


def wait_health(base, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            import urllib.request
            with urllib.request.urlopen(base + "/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"server not healthy within {timeout}s")


def bench_cell(binpath, cell, repeats, pp_tokens, tg_tokens, cdir, env=None,
               threads=16, ngl=99):
    logf = open(f"{cdir}/bench.log", "w")
    try:
        results = {}
        for p in pp_tokens:
            results.update(run_bench(binpath, cell["file"], repeats, p, 0, logf,
                                     env, threads, ngl))
        results.update(run_bench(binpath, cell["file"], repeats, 0, tg_tokens, logf,
                                 env, threads, ngl))
    finally:
        logf.close()
    return {k: {"mean": round(v["mean"], 2), "stddev": v["stddev"], "repeats": repeats}
            for k, v in results.items()}


def probe_cell(binpath, cell, ctx, port, cdir, repeats, tg_tokens, env=None,
               threads=16, ngl=99, warmup=1, kv="q8_0"):
    srv = subprocess.Popen(
        [f"{binpath}/llama-server", "-m", cell["file"], "--host", "127.0.0.1",
         "--port", str(port), "-c", str(ctx), "-ngl", str(ngl), "-t", str(threads),
         "-fa", "on", "-ctk", kv, "-ctv", kv, "-np", "1",
         "--alias", cell["id"]],
        stdout=open(f"{cdir}/server.log", "w"), stderr=subprocess.STDOUT,
        start_new_session=True, env=env)
    try:
        base = f"http://127.0.0.1:{port}"
        # M7: server startup (load + weight load) scales with model size; give the
        # 59 GB gpt-oss-120b cell more room than the 19–27 GB ones.
        timeout = 900 + int(cell["size_gib"]) * 15
        wait_health(base, timeout=timeout)
        ts = time.time()
        subprocess.run([sys.executable, f"{REPO}/scripts/ttft_probe.py", base,
                        cell["id"], str(warmup), str(repeats), str(tg_tokens),
                        f"{cdir}/ttft.json"], check=True, timeout=3600)
        ts_end = time.time()
        with open(f"{cdir}/ttft.json") as f:
            ttft = json.load(f)
        return ttft, ts, ts_end
    finally:
        killpg(srv)


def window_stats(rows, t0, t1):
    sel = [r for r in rows if t0 <= float(r["ts"]) <= t1]
    if not sel:
        return None
    stats = {
        "power_w_mean": round(statistics.fmean(float(r["power_w"]) for r in sel), 1),
        "power_w_max": round(max(float(r["power_w"]) for r in sel), 1),
        "temp_c_max": round(max(float(r["temp_c"]) for r in sel), 1),
        "vram_gib_max": round(max(float(r["vram_used_gib"]) for r in sel), 2),
        "sys_avail_gib_min": round(min(float(r["sys_avail_gib"]) for r in sel), 2),
    }
    # M-1: resident GPU memory is the SUM of the two pools (VRAM carve-out + GTT).
    # Only emitted when the trace actually recorded GTT: filling it in from VRAM
    # alone is what published a 59 GiB cell as "0.87 GiB peak". A resources.csv
    # without the gtt_used_gib column is a pre-2026-10-22 trace and gets no
    # mem_gib_max rather than a wrong one.
    if all(r.get("gtt_used_gib") for r in sel):
        stats["gtt_gib_max"] = round(max(float(r["gtt_used_gib"]) for r in sel), 2)
        stats["mem_gib_max"] = round(max(float(r["vram_used_gib"]) + float(r["gtt_used_gib"])
                                         for r in sel), 2)
    return stats


def load_matrix():
    with open(f"{REPO}/matrix.yaml") as f:
        return yaml.safe_load(f)


def _ngl(gpu_layers):
    """Translate matrix.yaml `gpu_layers` into a llama.cpp -ngl value ("all" -> 99)."""
    return 99 if gpu_layers == "all" else int(gpu_layers)


def build_parser(defaults):
    """ArgumentParser whose defaults derive from matrix.yaml `defaults` (I-1).

    The matrix is the single source of truth for run parameters; argparse defaults
    are computed from it rather than duplicated as literals, so editing a value in
    matrix.yaml actually changes the sweep instead of being silently ignored.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=defaults.get("backends", ["vulkan", "rocm"]),
                    help="backend to sweep (required unless --consolidate)")
    ap.add_argument("--cells", default="", help="comma-separated cell ids (default: all)")
    ap.add_argument("--repeats", type=int, default=int(defaults.get("repeats", 3)))
    ap.add_argument("--pp-tokens",
                    default=",".join(str(x) for x in defaults.get("pp_tokens", [512, 2048])))
    ap.add_argument("--tg-tokens", type=int, default=int(defaults.get("tg_tokens", 512)))
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--ctx", type=int, default=int(defaults.get("ctx_size", 8192)))
    ap.add_argument("--threads", type=int, default=int(defaults.get("threads_cpu", 16)),
                    help="CPU threads for llama-bench/llama-server (threads_cpu)")
    ap.add_argument("--ngl", type=int, default=_ngl(defaults.get("gpu_layers", "all")),
                    help="GPU layers (-ngl); matrix gpu_layers 'all' -> 99")
    ap.add_argument("--warmup", type=int, default=int(defaults.get("warmup_requests", 1)),
                    help="untimed warmup requests before the timed TTFT probe")
    ap.add_argument("--power-hz", type=int, default=int(defaults.get("power_sample_hz", 5)),
                    help="resource sampler rate (power_sample_hz)")
    ap.add_argument("--kv", choices=("q8_0", "f16"), default="q8_0",
                    help="KV cache type for the server phase (-ctk/-ctv); q8_0 is the "
                         "historical default, f16 avoids the quantised-KV bookkeeping "
                         "that inflates TTFT on some Vulkan cells (RUNNING §7)")
    ap.add_argument("--force", action="store_true", help="skip idle check")
    ap.add_argument("--rocm-override", action="store_true",
                    help="set HSA_OVERRIDE_GFX_VERSION=11.5.0")
    ap.add_argument("--consolidate", action="store_true",
                    help="rebuild results/consolidated.csv + results/table.md from "
                         "existing run dirs, then exit (no GPU work)")
    return ap


def main():
    # Load the matrix first: its `defaults` block is the single source of truth for
    # run parameters (I-1), so argparse defaults derive from it rather than being
    # duplicated as literals elsewhere.
    matrix = load_matrix()
    ap = build_parser(matrix.get("defaults") or {})
    args = ap.parse_args()

    if args.consolidate:
        # Standalone rollup: usable after an aborted/partial sweep, which is
        # exactly when consolidate() never got to run before.
        consolidate()
        log("consolidated -> results/consolidated.csv + results/table.md")
        return
    if not args.backend:
        ap.error("--backend is required unless --consolidate is given")

    env = dict(os.environ)
    if args.rocm_override:
        env["HSA_OVERRIDE_GFX_VERSION"] = "11.5.0"

    pp_tokens = [int(x) for x in args.pp_tokens.split(",")]
    cells = [c for c in matrix["cells"]
             if not args.cells or c["id"] in args.cells.split(",")]
    # matrix.yaml stores model paths as ~/-relative (no home-directory username in
    # the committed file); expand them so os.path.isfile() and llama-bench/server
    # resolve them on this machine.
    for c in cells:
        c["file"] = os.path.expanduser(c["file"])

    run_id = time.strftime("run-%Y%m%d-%H%M")
    rdir = f"{REPO}/results/{run_id}/{args.backend}"
    os.makedirs(rdir, exist_ok=True)
    try:
        commit = subprocess.run(["git", "-C", LLAMA_CPP, "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    try:
        harness_commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                        capture_output=True, text=True).stdout.strip()
    except Exception:
        harness_commit = "unknown"
    # B1: the pinned llama-bench has NO -c/--ctx flag, so bench runs at the model's
    # default context (4096). The server (llama-server) DOES accept -c and runs at
    # args.ctx. Record both so the data is never silently mislabeled.
    meta = {"run_id": run_id, "ts": time.time(), "backend": args.backend,
            "llama_cpp_commit": commit, "harness_commit": harness_commit,
            "cpu_power": cpu_power_state(),
            "cells": [c["id"] for c in cells],
            "repeats": args.repeats, "tg_tokens": args.tg_tokens,
            "pp_tokens": pp_tokens,
            "ctx_server": args.ctx,
            "ctx_bench": "model-default (4096)",
            "kv_cache": args.kv,
            "kernel": platform.release(),
            "mesa": mesa_version(),
            "note_bench_ctx": "llama-bench in pinned build 5b59b83f4 has no -c/--ctx; "
                              "bench tok/s measured at model default ctx, not ctx_server",
            # M-2: a --cells selection that does not cover the matrix is a probe.
            # Stamp it so a later full sweep's consolidate() cannot fold its rows
            # into the published rollup as the "latest" value for those cells.
            "partial": not tracks_rollup([c["id"] for c in cells],
                                         [c["id"] for c in matrix["cells"]]),
            "forced": args.force}
    json.dump(meta, open(f"{rdir}/run.json", "w"), indent=1)
    open_sweep_log(f"{rdir}/sweep.log")
    log(f"argv: {' '.join(sys.argv)}")
    log(f"harness {harness_commit[:12]} | llama.cpp {commit[:12]} | "
        f"backend {args.backend} | cells {[c['id'] for c in cells]} "
        f"(bench ctx = model default; server ctx = {args.ctx})")
    ps = meta["cpu_power"]
    log(f"cpu: governor={ps['governor']} epp={ps['epp']} "
        f"pstate={ps['pstate_driver']}")

    def check_idle(tag):
        """Idle gate (I3): GPU must be idle before each cell, not just once.

        Waits for the GPU to *settle* (see wait_idle) rather than sampling once.
        Returns True to proceed; False means hard-fail (caller exits). --force
        downgrades to a contamination warning rather than a stop.
        """
        ok, busy, mem = wait_idle(tag)
        if not ok:
            if not args.force:
                log(f"[{tag}] GPU NOT idle after {IDLE_SETTLE_S}s "
                    f"(busy={busy}%, vram+gtt={mem:.1f} GiB) — stop the running "
                    "llama-server and rerun, or pass --force")
                return False
            log(f"[{tag}] GPU not idle (busy={busy}%, vram+gtt={mem:.1f} GiB) but --force: "
                "continuing, this cell's results are CONTAMINATED")
        return True

    # M6: fail fast on a missing/renamed GGUF instead of dying minutes deep in the sweep.
    for cell in cells:
        if not os.path.isfile(cell["file"]):
            log(f"ERROR: model file missing for {cell['id']}: {cell['file']}")
            sys.exit(3)

    if not check_idle("start"):
        sys.exit(2)

    for cell in cells:
        # I3: re-check per cell — a 10-cell sweep spans hours; a background process
        # that starts mid-sweep must not silently contaminate later cells.
        if not check_idle(f"cell {cell['id']}"):
            sys.exit(2)
        cdir = f"{rdir}/{cell['id']}"
        os.makedirs(cdir, exist_ok=True)
        log(f"=== {cell['id']} ({cell['quant']}, {cell['size_gib']} GiB) ===")
        sampler = subprocess.Popen([sys.executable, f"{REPO}/scripts/sampler.py",
                                    f"{cdir}/resources.csv", str(args.power_hz)],
                                   start_new_session=True, env=env)
        phases = {}
        try:
            phases["bench_start"] = time.time()
            bench = bench_cell(env_binpath(args), cell, args.repeats,
                               pp_tokens, args.tg_tokens, cdir, env,
                               args.threads, args.ngl)
            phases["bench_end"] = time.time()
            log(f"  bench: { {k: v['mean'] for k, v in bench.items()} }")
            ttft, p0, p1 = probe_cell(env_binpath(args), cell, args.ctx, args.port,
                                      cdir, args.repeats, args.tg_tokens, env,
                                      args.threads, args.ngl, args.warmup, args.kv)
            phases["probe_start"], phases["probe_end"] = p0, p1
            log(f"  probe: ttft={ttft['ttft_ms_mean']}ms gen={ttft['gen_tps_mean']} t/s")
        finally:
            phases["done"] = time.time()
            killpg(sampler)
        json.dump(phases, open(f"{cdir}/phases.json", "w"), indent=1)

        rows = list(csv.DictReader(open(f"{cdir}/resources.csv")))
        res = {
            "bench_window": window_stats(rows, phases["bench_start"], phases["bench_end"]),
            "probe_window": window_stats(rows, phases["probe_start"], phases["probe_end"]),
            "cell_window": window_stats(rows, phases["bench_start"], phases["done"]),
        }
        tg_srv = ttft["gen_tps_mean"]
        pw = (res["probe_window"] or {}).get("power_w_mean")
        summary = {
            "cell": cell["id"], "backend": args.backend, "run": run_id, "kv": args.kv,
            "model": cell["model"], "quant": cell["quant"], "arch": cell["arch"],
            "size_gib": cell["size_gib"], "bench": bench, "ttft": ttft,
            "resources": res,
            "tok_per_w": round(tg_srv / pw, 2) if pw else None,
        }
        json.dump(summary, open(f"{cdir}/cell.json", "w"), indent=1)

    if tracks_rollup([c["id"] for c in cells], [c["id"] for c in matrix["cells"]]):
        consolidate()
    else:
        # A one-cell probe must not rewrite the published rollup: consolidate()
        # scans every run dir, so a partial run's rows would silently become the
        # "latest" value for that cell in consolidated.csv, table.md, the charts
        # and every README number pinned to them.
        log(f"partial sweep ({len(cells)}/{len(matrix['cells'])} cells): tracked rollup, "
            f"table and charts left untouched (cell.json in the run dir has the numbers)")
    # M8 (tracked, not yet implemented): PLAN §3 says every sweep is "git commit +
    # tag run-YYYYMMDD-N" with llama.cpp commit + env fingerprint, and fixes the CPU
    # governor. The driver does NOT do this yet. Until implemented, run manually after
    # a sweep:
    #   git add results/ && git commit -m "run: <id>" \
    #     && git tag run-$(date +%Y%m%d)-$(n)
    # and fix the governor (e.g. `cpupower frequency-set -g performance`).
    log(f"done -> {rdir} (see results/table.md)")


def env_binpath(args):
    return BUILDS[args.backend]


def git_head(repo):
    """Return the git HEAD of `repo`, or "unknown" if it is not a git checkout."""
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


DRIVERS = ("scripts/run_sweep.py", "scripts/sampler.py", "scripts/ttft_probe.py")


def harness_commit(repo):
    """The commit that last changed the code producing this rollup — not HEAD.

    Stamping HEAD made the stamp stale on the very next commit, and because
    --consolidate rewrites the stamp, every subsequent landing dirtied the tracked
    rollup again (finding M-3 of the independent review recurred on each commit). The numbers are fixed by
    the driver code, so record that code's last change: unrelated commits then
    leave the rollup byte-identical.
    """
    try:
        out = subprocess.run(["git", "-C", repo, "log", "-1", "--format=%H", "--", *DRIVERS],
                             capture_output=True, text=True).stdout.strip()
        return out or git_head(repo) or "unknown"
    except Exception:
        return "unknown"


def mesa_version():
    """Best-effort RADV/Mesa version for provenance (D7: numbers are stack-stamped)."""
    try:
        out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True,
                             timeout=20).stdout
        m = re.search(r"driverInfo\s*=\s*(Mesa[^\n]+)", out)
        return m.group(1).strip() if m else "unknown"
    except Exception:
        return "unknown"


def tracks_rollup(selected_ids, all_ids):
    """Only a sweep covering the whole matrix may rewrite the tracked rollup."""
    return set(selected_ids) == set(all_ids)


def consolidate(repo=REPO):
    """Rebuild results/consolidated.csv + results/table.md from all run dirs."""
    rows = []
    for rj in sorted(glob.glob(f"{repo}/results/run-*/**/run.json", recursive=True)):
        rd_ = os.path.dirname(rj)
        # M-2: skip a partial sweep (--cells ...). The driver refuses to
        # consolidate one itself, but every later full sweep calls consolidate(),
        # and without this the probe's rows come back as the "latest" value for
        # the cells it touched (run-20261022-0917 did exactly that).
        try:
            with open(rj) as f:
                if json.load(f).get("partial"):
                    continue
        except (OSError, ValueError):
            pass
        for cj in sorted(glob.glob(f"{rd_}/*/cell.json")):
            with open(cj) as f:
                s = json.load(f)
            res = s["resources"]
            base = {"run_id": s["run"], "backend": s["backend"], "cell": s["cell"],
                    "model": s["model"], "quant": s["quant"], "size_gib": s["size_gib"]}
            for test, stats in s["bench"].items():
                reps = stats.get("repeats")
                if isinstance(reps, list):      # legacy cell.json: parsed rows list
                    reps = len(reps)
                sd = stats.get("stddev")
                rows.append({**base, "metric": f"{test}_bench_tps", "unit": "t/s",
                             "value": stats["mean"],
                             "note": f"llama-bench mean of {reps}"
                                     + (f", sd ±{sd}" if sd is not None else "")})
            # M3: N is the actual request count (len of ttft.requests), not a literal.
            nreq = len(s["ttft"].get("requests", []))
            rows.append({**base, "metric": "tg_server_tps", "unit": "t/s",
                         "value": s["ttft"]["gen_tps_mean"],
                         "note": f"streaming, mean of {nreq}"})
            rows.append({**base, "metric": "ttft_ms", "unit": "ms",
                         "value": s["ttft"]["ttft_ms_mean"],
                         "note": f"mean of {nreq}"})
            # Mean alone misleads when the per-request spread is wide: on Vulkan
            # the first timed request hits the prefill prediction and requests
            # 2-3 take 2-12x longer to first token, while ROCm runs flat. Ship the
            # median alongside it so the difference is visible in the rollup.
            if s["ttft"].get("ttft_ms_p50") is not None:
                rows.append({**base, "metric": "ttft_ms_p50", "unit": "ms",
                             "value": s["ttft"]["ttft_ms_p50"],
                             "note": f"median of {nreq}; compare with ttft_ms to "
                                     "gauge request-to-request spread"})
            for wname in ("probe_window", "bench_window"):
                w = res.get(wname) or {}
                if w:
                    rows.append({**base, "metric": f"power_w_{wname[:5]}", "unit": "W",
                                 "value": w["power_w_mean"], "note": f"max {w['power_w_max']}"})
            cw = res.get("cell_window") or {}
            if cw:
                rows.append({**base, "metric": "temp_c_max", "unit": "C",
                             "value": cw["temp_c_max"], "note": ""})
                # M-1: vram_peak_gib is the VRAM pool alone. On a small UMA
                # carve-out the weights live in GTT and this column sits at the
                # idle floor, so mem_peak_gib carries residency; the plain-VRAM
                # figure stays for the runs that predate GTT tracking.
                rows.append({**base, "metric": "vram_peak_gib", "unit": "GiB",
                             "value": cw["vram_gib_max"],
                             "note": "VRAM pool only; residency = mem_peak_gib"
                                     if "mem_gib_max" in cw else ""})
                if "mem_gib_max" in cw:
                    rows.append({**base, "metric": "mem_peak_gib", "unit": "GiB",
                                 "value": cw["mem_gib_max"],
                                 "note": "peak resident GPU memory (VRAM + GTT)"})
            if s.get("tok_per_w"):
                rows.append({**base, "metric": "tok_per_w", "unit": "t/s/W",
                             "value": s["tok_per_w"], "note": "server tg / probe W mean"})

    path = f"{repo}/results/consolidated.csv"
    fields = ["run_id", "backend", "cell", "model", "quant", "size_gib",
              "metric", "unit", "value", "note"]
    # B3: rebuild the whole file ("w") — rows are already re-derived from ALL run
    # dirs above. Append mode re-emitted every run's rows on each call, so a Vulkan
    # pass followed by ROCm would double every Vulkan row.
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # render table.md (all runs)
    # M7: record which harness build produced the rollup. run.json stamps the
    # capture-time harness; this stamps the driver code that generated the numbers,
    # so the tracked rollup carries its own provenance without going stale on every
    # unrelated commit (stamping HEAD did; review finding M-3 kept recurring).
    rollup_commit = harness_commit(repo) or "unknown"
    md = ["# Results", "",
          f"> Rollup built by harness `{rollup_commit[:12]}`. Sustained = llama-server "
          "streaming 512-token generations. bench = llama-bench (mean of repeats). "
          "probe W = GPU power mean during server phase. TTFT mean is inflated where "
          "the per-request spread is wide (Vulkan) — compare with p50. Peak GPU mem = "
          "peak (VRAM + GTT), the residency figure; runs that predate GTT sampling "
          "show the VRAM pool only.", ""]
    runs = {}
    for r in rows:
        runs.setdefault((r["run_id"], r["backend"]), []).append(r)
    for (rid, be), rs in sorted(runs.items()):
        md.append(f"## {rid} — {be}")
        md.append("")
        md.append("| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        bycell = {}
        for r in rs:
            bycell.setdefault(r["cell"], {})[r["metric"]] = r["value"]
        for cell, m in bycell.items():
            quant = next(r["quant"] for r in rs if r["cell"] == cell)
            md.append("| {c} | {q} | {p} | {b} | {s} | {t} | {t50} | {w} | {e} | {v} |".format(
                c=cell, q=quant,
                p=m.get("pp2048_bench_tps", "–"), b=m.get("tg512_bench_tps", "–"),
                s=m.get("tg_server_tps", "–"), t=m.get("ttft_ms", "–"),
                t50=m.get("ttft_ms_p50", "–"),
                w=m.get("power_w_probe", "–"), e=m.get("tok_per_w", "–"),
                # M-1: publish residency (VRAM + GTT) when the run recorded GTT,
                # and fall back to the VRAM pool for runs that predate it.
                v=m.get("mem_peak_gib") or m.get("vram_peak_gib", "–")))
        md.append("")
    with open(f"{repo}/results/table.md", "w") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
