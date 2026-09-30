# Running the benchmark suite

Operator guide for `scripts/run_sweep.py` — the one-command sweep driver.
Companion docs: [README](../README.md) (what + hardware + matrix), [PLAN](../PLAN.md)
(locked decisions + methodology + phases), [env.md](env.md) (verified machine fingerprint).

> **Current state (2026-09-30):** the harness is implemented, independently reviewed
> ([REVIEW.md](../REVIEW.md)), and the blocking/important fixes (B1–B3, I1–I4) are
> committed on branch `fix/harness-b1-b3-i1-i4`. A **clean end-to-end smoke run has not
> yet been captured** — the first real smoke is what exercises the `llama-bench` table
> parser and the SSE probe against this pinned build. Until then, treat every number in
> `results/` as unverified.

---

## 1. Prerequisites

| Need | Detail |
|---|---|
| This machine | The pinned llama.cpp builds and all 5 GGUF models are already on disk (see [env.md](env.md)). No downloads. |
| Python for the driver | Must import **PyYAML** (only `run_sweep.py` needs it — to parse `matrix.yaml`). On this box use **`/usr/bin/python3`** (system, has PyYAML). The `python3` on a shell PATH that points at a venv has no PyYAML and the driver refuses with an install hint. The two helpers (`sampler.py`, `ttft_probe.py`) are stdlib-only. |
| Idle GPU | The dev `llama-server` on `127.0.0.1:11436` (serves `opencode`, holds ~22 GiB VRAM) **must be stopped** before a real sweep. |

Install the one dependency if you ever need a different python:

```
pip install -r requirements.txt     # PyYAML>=6.0
```

## 2. Stop the dev server (do this first)

```
# find it (it is the llama-server on :11436)
pgrep -af 'llama-server.*11436'
kill <pid>                          # SIGTERM; it should release cleanly

# wait ~10s, then confirm it released VRAM + settled:
awk '{printf "vram: %.2f GiB\n", $1/1024^3}' /sys/class/drm/card0/device/mem_info_vram_used
awk '{printf "busy: %s%%\n", $1}' /sys/class/drm/card0/device/gpu_busy_percent
```

The driver's idle gate requires **busy < 5% and VRAM < 4 GiB**. It is checked **at sweep
start and again before every cell** (a 10-cell sweep spans hours; a background process that
starts mid-sweep must not silently contaminate later cells). If the GPU is not idle the
driver **hard-fails (exit 2)** — unless you pass `--force`, which downgrades the failure to
a "CONTAMINATED" warning and stamps `forced: true` into `run.json` (that's how the
`run-20260930-1851-SMOKE-FAILED` run came to exist — do not run a real sweep with `--force`).

## 3. Run a sweep

```
cd ~/projects/local-llm-benchmarks

# Full sweep of all 5 cells, one backend each:
/usr/bin/python3 scripts/run_sweep.py --backend vulkan
/usr/bin/python3 scripts/run_sweep.py --backend rocm

# Smoke — the single smallest cell (19.4 GiB), ~5–10 min:
/usr/bin/python3 scripts/run_sweep.py --backend rocm --cells qwen3.8-27b-q5_k_xl
```

Options (from `--help`):

| Flag | Default | Meaning |
|---|---|---|
| `--backend {vulkan,rocm}` | required | `vulkan` = `build-vulkan` (reference, A); `rocm` = `build` (HIP, B) |
| `--cells id1,id2` | all | comma-separated cell ids from `matrix.yaml` |
| `--repeats N` | `3` | `llama-bench` repeats per test |
| `--pp-tokens a,b` | `512,2048` | prompt-processing sweep points |
| `--tg-tokens N` | `512` | generation length (both bench and probe) |
| `--port N` | `11500` | local `llama-server` port (binds 127.0.0.1) |
| `--ctx N` | `8192` | server context size (see caveat in §7) |
| `--force` | off | skip the idle gate (results flagged contaminated) |
| `--rocm-override` | off | set `HSA_OVERRIDE_GFX_VERSION=11.5.0` (gfx1151 fallback, see §7) |

## 4. What the driver does per cell

1. **Model pre-check** — `os.path.isfile(cell.file)` for every selected cell *before* the
   loop; a missing/renamed GGUF fails fast (exit 3) instead of minutes into the sweep.
2. **Idle gate** — as above.
3. **Resource sampler** — spawns `sampler.py` writing `resources.csv` at 5 Hz (sysfs only).
4. **`llama-bench`** — for each pp point (`-n 0`) plus one tg run (`-p 0 -n 512`), `-ngl 99
   -t 16 -r <repeats>`; output → `bench.log`. Runs at the **model-default context** (see §7).
5. **`llama-server` + probe** — starts the server (`-c <ctx> -ngl 99 -fa on -ctk q8_0 -ctv
   q8_0 -np 1 --alias <cell>`), waits on `/health` (timeout scales with model size:
   `900 + size_gib*15` s), then runs `ttft_probe.py`: 1 untimed warmup request, then `N`
   timed **streaming** 512-token completions (temperature 0) → `ttft.json`. TTFT = time to
   first streamed token; gen_tps = tokens / (last − first).
6. **Aggregate** — kills server + sampler, slices `resources.csv` into bench / probe / cell
   windows, writes `cell.json` (includes `tok_per_w = server tg ÷ probe-window mean W`).

After all cells: **rebuild** `results/consolidated.csv` and `results/table.md` from *every*
`run-*` directory (full rewrite, not append — so a Vulkan pass then a ROCm pass does not
duplicate rows).

## 5. Where results land

```
results/
  run-YYYYMMDD-HHMM/
    <backend>/                      # vulkan | rocm
      run.json                      # run metadata: llama.cpp commit, cells, ctx_bench/ctx_server, forced
      <cell_id>/
        bench.log                   # raw llama-bench stdout+stderr
        server.log                  # raw llama-server stdout
        resources.csv               # 5 Hz power/temp/busy/vram/sys-avail trace
        ttft.json                   # per-request TTFT + gen_tps (mean + p50)
        phases.json                 # bench/probe window timestamps
        cell.json                   # THE result: bench + ttft + resource windows + tok_per_w
  consolidated.csv                  # tracked rollup across all runs
  table.md                          # tracked rendered markdown across all runs
```

Raw run dirs (`results/run-*/`) are **gitignored**; only `consolidated.csv` + `table.md` are
tracked. The failed smoke dir is named `...-SMOKE-FAILED` so it is visually distinct, and it
contributes no data either way: `consolidate()` emits rows only from cells with a
`cell.json`, and the failed run has none — it stays on disk as evidence (its `run.json`
records `forced: true`).

## 6. Reading the results

- **`consolidated.csv`** — one row per (run, cell, metric). Columns:
  `run_id, backend, cell, model, quant, size_gib, metric, unit, value, note`.
  `metric` values: `pp512_bench_tps`, `pp2048_bench_tps`, `tg512_bench_tps` (llama-bench
  means), `tg_server_tps`, `ttft_ms` (streaming, from the probe), `power_w_probe` /
  `power_w_bench` (W means in each window), `temp_c_max`, `vram_peak_gib`, `tok_per_w`.
  The `note` carries the actual repeat/request count (e.g. "streaming, mean of 3").
- **`table.md`** — one table per (run, backend); the headline comparison view.

## 7. Caveats (read before trusting a number)

- **Two different contexts, recorded separately.** The pinned `llama-bench`
  (`5b59b83f4`) has **no `-c/--ctx` flag**, so bench tok/s are measured at the
  **model-default context (4096)** — *not* at `--ctx`. `llama-server` *does* take `-c` and
  runs at `--ctx` (default 8192). `run.json` records **both** `ctx_bench` and `ctx_server`
  so the data is never silently mislabeled. Don't compare a bench number to a server number
  as if they were the same context.
- **Power is GPU-only.** The sampler reads the amdgpu hwmon (`power1_average`, microwatts→W;
  `temp1_input`, millidegrees→°C), not full SoC draw. Under CPU-heavy prompt processing the
  true draw is higher; `amd-power-tune` (platform level) is not installed. `tok_per_w`
  therefore **understates** the energy cost of prompt processing.
- **`tok_per_w = server tg ÷ probe-window mean W`** — generation-phase efficiency, not a
  whole-run figure.
- **ROCm gfx1151** is seen natively (no override needed for a clean load). If a MoE/MXFP4 op
  misbehaves, re-run that backend with `--rocm-override` (sets
  `HSA_OVERRIDE_GFX_VERSION=11.5.0`); if it still fails, treat ROCm as a bonus axis per PLAN.
- **Throttling is data, not a failure.** Sustained 59 GB loads can heat the box; `temp_c_max`
  and the power trace are logged so a throttled run is still a *reported* run — just read the
  sustained (not burst) tok/s.

## 8. Recording a run (reproducibility — manual for now)

The driver does **not** yet auto-commit/tag (tracked as M8 in REVIEW.md; PLAN §3 calls for
"git commit + tag `run-YYYYMMDD-N`" per sweep). Until that lands, do it by hand after a
successful sweep:

```
git add results/ && git commit -m "run: $(ls -1t results | head -1)" \
  && git tag run-$(date +%Y%m%d)-<n>
```

Also, for controlled conditions: pin the CPU governor before the sweep
(`cpupower frequency-set -g performance`), and record the llama.cpp commit (it's already in
`run.json` → `llama_cpp_commit`) and the [env.md](env.md) fingerprint.
