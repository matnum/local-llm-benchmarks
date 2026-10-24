# Running the benchmark suite

Operator guide for `scripts/run_sweep.py` — the one-command sweep driver.
Companion docs: [README](../README.md) (what + hardware + matrix), [PLAN](../PLAN.md)
(design decisions + methodology), [env.md](env.md) (reference-machine fingerprint).

> **State:** the full sweep is done and re-measured — ten cells × two backends. The
> published throughput/energy/memory columns come from the 2026-10-22 pass on kernel 7.2.10 /
> Mesa 26.2.4, and the published TTFT/`server tg` columns from the 2026-10-24 cold-request
> re-measure (`run-20261024-0910` Vulkan, `run-20261024-0930` ROCm) on the same stack; the
> 2026-10-02/03 rows stay in the rollup under their own ids, and the 2026-10-22 morning
> pair (measured before the GPU-memory fix) is renamed `superseded-pre-mem-fix-*` and
> excluded from the rollup — see
> [results/table.md](../results/table.md). The harness is independently reviewed
> ([the independent review](reviews/2026-10-04-independent-review.md)). This guide
> documents how to run it.

---

## 1. Prerequisites

| Need | Detail |
|---|---|
| Models & builds | The pinned llama.cpp builds and all ten cells' GGUF models are already on disk (see [env.md](env.md)). No downloads. |
| Python for the driver | Must import **PyYAML** (only `run_sweep.py` needs it — to parse `matrix.yaml`). On the reference machine use **`/usr/bin/python3`** (system python, has PyYAML). A `python3` on `PATH` that points at a venv without PyYAML will be refused with an install hint. The two helpers (`sampler.py`, `ttft_probe.py`) are stdlib-only. |
| Idle GPU | Any running `llama-server` holding VRAM **must be stopped** before a real sweep (see §2). |

Install the one dependency if you ever need a different python:

```
pip install -r requirements.txt     # PyYAML>=6.0
```

## 2. Free the GPU (do this first)

```
# find any running llama-server
pgrep -af llama-server
kill <pid>                          # SIGTERM; it should release cleanly

# wait ~10s, then confirm it released memory + settled:
awk '{printf "vram: %.2f GiB\n", $1/1024^3}' /sys/class/drm/card0/device/mem_info_vram_used
awk '{printf "gtt:  %.2f GiB\n", $1/1024^3}' /sys/class/drm/card0/device/mem_info_gtt_used
awk '{printf "busy: %s%%\n", $1}' /sys/class/drm/card0/device/gpu_busy_percent
```

The driver's idle gate requires **busy < 5% and resident GPU memory < 4 GiB**, where
resident memory is **VRAM + GTT**: with the UMA carve-out at 1 GiB (see
[env.md](env.md)) a loaded `llama-server` holds its weights in GTT and leaves the
VRAM counter at its idle floor, so a VRAM-only gate would wave a loaded GPU through.
It is checked **at sweep start and again before every cell** (a 10-cell sweep spans
hours; a background process that starts mid-sweep must not silently contaminate later
cells). The check **polls for up to 90 s until the GPU settles** rather than sampling once:
tearing down a large model leaves
`gpu_busy` near 95% for several seconds after VRAM has already been released (the teardown is
itself GPU work and the counter lags it) — sampling once hard-failed the cell following a
59 GiB cell in `run-20261002-1731`. If the GPU still is not idle the driver
**hard-fails (exit 2)** — unless you pass `--force`, which downgrades the failure to
a "CONTAMINATED" warning and stamps `forced: true` into `run.json` (that's how the
`run-20260930-1851-SMOKE-FAILED` run came to exist — do not run a real sweep with `--force`).

## 3. Run a sweep

```
cd local-llm-benchmarks

# Full sweep of all 10 cells, one backend each:
/usr/bin/python3 scripts/run_sweep.py --backend vulkan
/usr/bin/python3 scripts/run_sweep.py --backend rocm

# Smoke — the single smallest cell (19.4 GiB), ~5–10 min:
/usr/bin/python3 scripts/run_sweep.py --backend rocm --cells qwen3.8-27b-q5_k_xl

# Roll up whatever exists, without touching the GPU (use after a partial/aborted
# sweep — before this flag existed, an interrupted sweep produced no table at all):
/usr/bin/python3 scripts/run_sweep.py --consolidate
```

Options (from `--help`). Run-parameter defaults come from `matrix.yaml` `defaults`
(single source of truth); the flags below override them per run.

| Flag | Default | Meaning |
|---|---|---|
| `--backend {vulkan,rocm}` | required | `vulkan` = `build-vulkan` (reference, A); `rocm` = `build` (HIP, B) |
| `--cells id1,id2` | all | comma-separated cell ids from `matrix.yaml` |
| `--repeats N` | `3` | `llama-bench` repeats per test |
| `--pp-tokens a,b` | `512,2048` | prompt-processing sweep points |
| `--tg-tokens N` | `512` | generation length (both bench and probe) |
| `--port N` | `11500` | local `llama-server` port (binds 127.0.0.1) |
| `--ctx N` | `8192` | server context size (see caveat in §7) |
| `--threads N` | `16` (`threads_cpu`) | CPU threads for `llama-bench` / `llama-server` |
| `--ngl N` | `99` (`gpu_layers: all`) | GPU layers (`-ngl`); matrix `all` → 99 |
| `--warmup N` | `1` (`warmup_requests`) | untimed warmup requests before the timed TTFT probe |
| `--power-hz N` | `5` (`power_sample_hz`) | resource sampler rate (Hz) |
| `--force` | off | skip the idle gate (results flagged contaminated) |
| `--rocm-override` | off | set `HSA_OVERRIDE_GFX_VERSION=11.5.0` (gfx1151 fallback, see §7) |
| `--consolidate` | off | rebuild `consolidated.csv` + `table.md` from existing run dirs and exit; no GPU, no `--backend` needed |
| `--check` | off | verify that the tracked rollups match what the local run dirs would rebuild, and exit 1 on drift; read-only, local-only (raw run dirs are gitignored, so CI has none) — see §3c |
| `--probe-only` | off | re-run **only** the `llama-server` + TTFT probe phase (no `llama-bench`), for re-measuring TTFT with a different KV type; see §3b |
| `--ttft-protocol {cold,warm}` | `cold` (`matrix.yaml` `defaults.ttft_protocol`) | what a timed TTFT request starts from: `cold` erases the slot's KV cache first (published since 2026-10-24 — full prompt prefill + first token, no KV bookkeeping in the number); `warm` leaves the prompt cache alone and is the pre-2026-10-24 protocol, kept to reproduce those runs; see §3c and §7 |
| `--tag` | off | after a full sweep, create the annotated run tag — but only when the tracked results already match HEAD, otherwise the exact commands are printed |

## 3b. Re-measuring one phase (`--probe-only`)

`--probe-only` runs the per-cell sequence **without** `llama-bench`: idle gate, sampler,
`llama-server` + TTFT probe, aggregate. It exists so a protocol change can be re-measured
without paying for the bench phase again — the KV-type comparison in §7 is the motivating
case:

```
/usr/bin/python3 scripts/run_sweep.py --backend vulkan --kv f16 --probe-only
/usr/bin/python3 scripts/run_sweep.py --backend rocm   --kv f16 --probe-only
```

What a probe-only run publishes, and why it is not a full sweep:

- `cell.json` carries `"probe_only": true`, an empty `bench` block and a `null`
  `bench_window`; `phases.json` has no bench phase.
- `consolidate()` emits **only** `tg_server_tps`, `ttft_ms` and `ttft_ms_p50` from it, with
  `[<kv> KV, <warm|cold> TTFT, probe-only run]` appended to each note.
- It deliberately does **not** emit `temp_c_max`, `vram_peak_gib`/`mem_peak_gib`,
  `power_w_*` or `tok_per_w`: those describe a whole cell, and a probe-only cell window has
  no bench heat soak, so its peak temperature and residency understate the full-sweep cell
  they are published from. Those columns stay attributed to the last full sweep.
- Because latest-wins is per metric, the streaming columns move to the new run while the
  bench, heat and memory columns keep pointing at the full sweep. Each rollup row carries
  its own `run_id`, so the provenance of every column is visible.
- A probe-only pass that covers the matrix is published; a `--cells` subset is stamped
  `partial` and stays out (M-2). `run.json` is written with `partial: true` **before the
  first cell** and cleared only when the last selected cell has written its `cell.json`, so
  an interrupted sweep publishes nothing at all — the hole that let a crashed run's cells
  become the "latest" values for seven columns (I2, 2026-10-24).

**First use (2026-10-23):** the ten-cell f16 re-probe —
`results/run-20261023-1140` (Vulkan) and `-1159` (ROCm), both `--kv f16 --probe-only` —
answered the TTFT protocol question and doubled as the run-to-run noise measurement:
`tg_server_tps` within −0.2…+1.4%, `ttft_ms` −4.4…+6.5%, `ttft_ms_p50` −5.9…+8.1% over
20 cell/backend pairs (§7). Those two runs carried the streaming columns until the cold
protocol replaced them on 2026-10-24 (§3c), which is what the README table publishes now;
their per-request data is still tracked in `results/ttft-per-request.csv` as the warm side
of that comparison. The bench, power and memory columns stay attributed to the 2026-10-22
full sweeps.

## 3c. The cold-request TTFT protocol

`--ttft-protocol cold` (the default since 2026-10-24) makes every **timed** request start
from an empty KV cache: `ttft_probe.py` calls `POST /slots/0?action=erase` before each
one. The erase is untimed — it is setup, not latency — and `ttft.json` records
`protocol` plus each request's `erased_tokens`, so the data shows what was dropped.

The server must be started with `--slot-save-path` for the erasing
`POST /slots/:id?action=erase` action to exist — that flag is what gates the action.
`GET /slots` is gated separately on `--slots`, which this harness never passes, so a
missing `--slots` is not the error you will see. `probe_cell()` adds
`--slot-save-path` (pointed at `<cell_dir>/slots`) only in cold mode. Nothing is saved
there — the flag only unlocks the action. `--kv` and the protocol are independent:
the published runs are `f16` KV *and* cold.

Why it exists, measured on three cells (3×512-token requests, f16 KV; the full ten-cell
per-request table is in §7). Both sides of the comparison are tracked per request in
[`results/ttft-per-request.csv`](../results/ttft-per-request.csv) — warm rows from
`run-20261023-1140`/`-1159` (the column published until 2026-10-24), cold rows from the
published passes `run-20261024-1812` (Vulkan) / `-1749` (ROCm), with the earlier passes
`-0910` / `-0930` kept for the run-to-run band below — and `tests/test_ttft_extract.py`
checks this table against that file, so it cannot be retyped from memory again:

| cell | backend | warm per request (ms) | warm spread | cold per request (ms) | cold spread |
|---|---|---|---|---|---|
| `qwen3-4b-q8_0` | vulkan | 22.5 / 80.0 / 27.6 | 57.5 | 53.8 / 54.8 / 55.4 | 1.6 |
| `qwen3-4b-q8_0` | rocm | 23.4 / 31.2 / 24.8 | 7.8 | 42.2 / 42.2 / 42.7 | 0.5 |
| `qwen3-32b-q5_k_m` | vulkan | 104.9 / 186.7 / 185.7 | 81.8 | 401.0 / 431.7 / 428.3 | 30.7 |
| `qwen3.8-27b-q5_k_xl` | vulkan | 143.3 / 422.0 / 427.7 | 284.4 | 481.6 / 509.7 / 514.0 | 32.4 |

`gen_tps` is unchanged by the protocol — on the four published cold passes `server tg` sits
within ±0.4% per cell of the warm runs on ROCm, and within ±0.3% on Vulkan apart from one
cell (Qwen3-Coder-30B-A3B, −3.2%) — so the protocol changes only what TTFT measures. The
cold number is **larger** than a warm
first request because it includes the full prompt prefill: it is "time to first token for a
fresh request", which is what a user sees, rather than "time to first token when the cache
happens to hold the previous turn". `cache_prompt: false` was tried first and rejected: it
re-prefills the prompt but still pays the previous generation's KV eviction, so it was both
slower and just as noisy (27B: 480 / 511 / 702 ms).

**Repeatability.** Each backend now has two clean cold passes on the same stack, and the
band between them is the honest noise floor for a per-cell streaming figure (recomputed
from the extract by `make_extracts.py --stats`, pinned by `tests/test_ttft_extract.py`):

| backend | passes compared | `TTFT p50` per cell | median \|Δ\| | worst cell | `server tg` |
|---|---|---|---|---|---|
| vulkan | `run-20261024-0910` → `run-20261024-1812` | −6.2…+1.5% | 0.8% | `poro2-8b-q8_0` −6.2% | −3.1…+0.2% |
| rocm | `run-20261024-0930` → `run-20261024-1749` | −7.9…+0.2% | 0.4% | `gpt-oss-120b-mxfp4` −7.9% | −0.1…+2.4% |

The medians reproduce, the individual cells do not: 3 of 20 cell/backend pairs move
3–8% between two identical quiet passes, so a per-cell cold median is good to roughly
±8% on the worst cell and ~1% typically — not the ±2% the bench repeats support.

Cold is stable, not identical across models: the 27B Q6_K cell still takes 1.21 s to first
token on Vulkan (0.36 s on ROCm) because prefilling the 46-token prompt on a 20 GiB dense
model costs that much. And a contended pass shows it in `gen_tps` before TTFT — the 08:43
Vulkan pass (`superseded-contended-20261024-0843`) read the 4B cell at 39.85 t/s against
48.15 in the clean 09:10 pass, while its p50 moved 0.5%.

Reproduce a cold pass without re-running the bench phase:

```
/usr/bin/python3 scripts/run_sweep.py --backend vulkan --kv f16 --ttft-protocol cold --probe-only
/usr/bin/python3 scripts/run_sweep.py --backend rocm   --kv f16 --ttft-protocol cold --probe-only
```

`--ttft-protocol warm` re-creates the pre-2026-10-24 column; the two protocols are
different quantities and the rollup note says which one a row carries. The per-request
values behind the published cold column are tracked in
[`results/ttft-per-request.csv`](../results/ttft-per-request.csv), and
`tests/test_ttft_extract.py` re-derives the rollup's mean/median from it.

## 3d. Run the tests

The harness has a small stdlib-`unittest` suite (no pytest dependency) covering the
`llama-bench` table parser, `consolidate()` idempotency (no duplicate rows) and the
`matrix.yaml`-defaults wiring. Run it with the same python as the driver:

```
/usr/bin/python3 -m unittest discover -s tests -v
```

## 4. What the driver does per cell

1. **Model pre-check** — `os.path.isfile(cell.file)` for every selected cell *before* the
   loop; a missing/renamed GGUF fails fast (exit 3) instead of minutes into the sweep.
2. **Idle gate** — as above.
3. **Resource sampler** — spawns `sampler.py` writing `resources.csv` at 5 Hz (sysfs only).
4. **`llama-bench`** — for each pp point (`-n 0`) plus one tg run (`-p 0 -n 512`), `-ngl 99
   -t 16 -r <repeats>`; output → `bench.log`. Runs at the **model-default context** (see §7).
5. **`llama-server` + probe** — starts the server (`-c <ctx> -ngl 99 -fa on -ctk <kv> -ctv <kv> -np 1 --alias <cell>`,
   `--kv` defaulting to `q8_0`; the published runs use `f16`), adding
   `--slot-save-path <cell>/slots` for the cold TTFT protocol (§3c), waits on `/health` (timeout scales with model size:
   `900 + size_gib*15` s), then runs `ttft_probe.py`: 1 untimed warmup request, then `N`
   timed **streaming** 512-token completions (temperature 0, **`ignore_eos: true`**) →
   `ttft.json`. TTFT = time to first streamed token; gen_tps = tokens / (t_end − t_first),
   both absolute clock readings. `ignore_eos` matters: without it a model that ends its
   turn on a raw completion prompt (Qwen3.8 does) returns one empty chunk and the probe
   cannot measure anything. Each request also records `finish_reason`,
   `tokens_requested` and `complete_decodes`, so a partial decode is visible in the JSON
   instead of being averaged in silently.
6. **Aggregate** — kills server + sampler, slices `resources.csv` into bench / probe / cell
   windows, writes `cell.json` (includes `tok_per_w = server tg ÷ probe-window mean W`).

Every `log()` line is also written to `<run>/<backend>/sweep.log`, together with `argv`, the
harness commit and any crash traceback — so an interrupted sweep explains itself instead of
leaving a `run.json` that merely lists cells with no artifacts.

After all cells: **rebuild** `results/consolidated.csv` and `results/table.md` from *every*
`run-*` directory (full rewrite, not append — so a Vulkan pass then a ROCm pass does not
duplicate rows). The same rebuild is available standalone via `--consolidate`, for when a
sweep is aborted or only some cells were selected.

The driver cannot tell whether the tracked rollup in the worktree is the one those run dirs
produce, so before committing run numbers check it:

```
/usr/bin/python3 scripts/run_sweep.py --check      # exit 1 on drift, names the rows
```

`--check` is deliberately local-only: it rebuilds the rollup in memory and diffs it against
the tracked files, and raw `run-*` dirs are gitignored, so a CI checkout has nothing to
compare (it says so and passes). This is the drift guard the README once claimed the test
suite provided (review I5, 2026-10-24): the suite can only diff the README against the
tracked CSV, which fires *after* a rebuild has already invalidated the README.

## 5. Where results land

```
results/
  run-YYYYMMDD-HHMM/
    <backend>/                      # vulkan | rocm
      run.json                      # run metadata: llama.cpp + harness commit, cpu_power, cells, ctx_bench/ctx_server, forced
      sweep.log                     # driver stdout + crash traceback (why an interrupted run stopped)
      <cell_id>/
        bench.log                   # raw llama-bench stdout+stderr
        server.log                  # raw llama-server stdout
        resources.csv               # 5 Hz power/temp/busy/vram/sys-avail trace
        ttft.json                   # per-request TTFT + gen_tps (mean + p50)
        phases.json                 # bench/probe window timestamps
        cell.json                   # THE result: bench + ttft + resource windows + tok_per_w
  consolidated.csv                  # tracked rollup across all runs
  table.md                          # tracked rendered markdown across all runs
  ttft-per-request.csv              # tracked: per-request TTFT, both protocols (§3c)
  bench-logs/<run>/<backend>/<cell>.log  # tracked: verbatim llama-bench output for the
                                    # sweeps that own the published bench columns
  sampler-support.csv               # tracked: per-window sample count/duration + the
                                    # power/temp/memory aggregate the rollup copies
```

Raw run dirs (`results/run-*/`) are **gitignored**; the rollups plus the three extracts
above are tracked. The extracts exist so a reader can check a published number without the
run dirs (review 5.3): `ttft-per-request.csv` re-derives `ttft_ms`/`ttft_ms_p50`,
`bench-logs/` re-derives the three llama-bench means and their ±sd through the driver's own
`parse_bench()`, and `sampler-support.csv` shows how many 5 Hz samples (and how many
seconds) back each published power/temperature/memory figure — the traces themselves are
12k rows per sweep, too large to ship. `tests/test_ttft_extract.py` and
`tests/test_bench_extract.py` pin both directions.

`llama-bench -r 3` prints one `mean ± sd` row and does **not** keep the three repeats, so a
per-repeat bench extract cannot be built from the stored data — the log is the finest
artifact that exists.

The failed smoke dir is named `...-SMOKE-FAILED` so it is visually distinct, and it
contributes no data either way: `consolidate()` emits rows only from cells with a
`cell.json`, and the failed run has none — it stays on disk as evidence (its `run.json`
records `forced: true`).

A run that is superseded — replaced by a better pass on the same stack, or measured with a
harness whose metadata was wrong — is renamed `results/superseded-<reason>-<run-id>/` and
falls **outside** `consolidate()`'s `results/run-*` glob, so it can never re-enter the rollup
or a published table. Example: `superseded-pre-mem-fix-20261022-0919` and `-1001`, the
morning pair measured before the VRAM+GTT fix, replaced by the evening pair after it.

## 6. Reading the results

- **`consolidated.csv`** — one row per (run, cell, metric). Columns:
  `run_id, backend, cell, model, quant, size_gib, metric, unit, value, note`.
  `metric` values: `pp512_bench_tps`, `pp2048_bench_tps`, `tg512_bench_tps` (llama-bench
  means), `tg_server_tps`, `ttft_ms` / `ttft_ms_p50` (streaming, from the probe),
  `power_w_probe` / `power_w_bench` (W means in each window), `temp_c_max`, `vram_peak_gib`
  (the VRAM pool alone), `mem_peak_gib` (residency = VRAM + GTT, only for runs whose trace
  records GTT), `tok_per_w`.
  The `note` carries the actual repeat/request count (e.g. "streaming, mean of 3").
- **`table.md`** — one table per (run, backend); the headline comparison view.

**Charts are a separate step.** The driver never redraws them: after any `--consolidate`
that changes `consolidated.csv`, run `/usr/bin/python3 scripts/make_charts.py` and commit
`charts/`. Each PNG carries a fingerprint of its own inputs (PNG tEXt chunk
`chart-stamp`: cell count + a digest of every plotted Vulkan/ROCm value), and
`tests/test_charts.py` fails when a committed PNG's stamp no longer matches the rollup —
so a stale figure cannot ship with fresh numbers.

## 7. Caveats (read before trusting a number)

- **CPU power state is a measurement condition, and it is deliberately not pinned.** The
  reference machine runs `amd_pstate=active` with governor `powersave`, EPP `balance_performance` and no power
  daemon (decision 2026-10-02: leave the default, record it — see
  [env.md](env.md#cpu-power-state-measurement-condition)). `run.json.cpu_power` stamps what
  each run measured under, so a run taken after a profile change is spotted rather than
  silently averaged in with the rest.
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
- **GPU memory is measured as two pools, and the published figure is their sum.** The
  sampler records `mem_info_vram_used` **and** `mem_info_gtt_used` into `resources.csv`
  (`vram_used_gib`, `gtt_used_gib`), and the rollup carries
  `mem_peak_gib = peak(VRAM + GTT)`. This matters on an APU: with a large UMA carve-out the
  weights sit in the VRAM pool; with a small one (this box: `mem_info_vram_total` = 1 GiB as
  of 2026-10-22) the same weights sit in GTT and `mem_info_vram_used` stays at its idle floor
  — a VRAM-only column read a 59 GiB cell as "0.87 GiB peak". Runs taken before the fix have
  no `gtt_used_gib` column and therefore **no** `mem_peak_gib`: their `vram_peak_gib` is the
  residency figure only because those runs predate the carve-out change. `vram_peak_gib` is
  still emitted (it is what the counter said) but it is **not** residency on the current box.
  The README table and `results/table.md` publish `mem_peak_gib` and fall back to
  `vram_peak_gib` only for runs without GTT sampling; `tests/test_readme_tables.py` fails if a
  published cell has no residency row, so the dead counter cannot come back as a headline.
- **`tok_per_w = server tg ÷ probe-window mean W`** — generation-phase efficiency, not a
  whole-run figure.
- **TTFT is now the cold-request protocol, and the number means "prefill a fresh request".**
  Since 2026-10-24 both backends are measured with `--ttft-protocol cold`: the probe erases the
  slot's KV cache before every timed request (§3c), so each request pays full prompt prefill +
  first token from an empty cache and nothing else. The published column comes from
  `run-20261024-1812` (Vulkan) and `run-20261024-1749` (ROCm) — the second quiet pass on
  each backend — and every streaming row in the
  rollup says which protocol it carries (`[<kv> KV, <warm|cold> TTFT]`).

  Per-request values, clean Vulkan pass (3 × 512-token requests) — the same data,
  both backends, is tracked in
  [`results/ttft-per-request.csv`](../results/ttft-per-request.csv)
  so a reader can check it without the gitignored run dirs:

  | cell | warm p50 (pre-2026-10-24) | cold per request (ms) | cold p50 |
  |---|---|---|---|
  | qwen3-4b-q8_0 | 27.6 | 53.8 / 54.8 / 55.4 | 54.8 |
  | poro2-8b-q8_0 | 42.3 | 118.4 / 120.0 / 120.3 | 120.0 |
  | eurollm-9b-q8_0 | 48.3 | 141.9 / 143.5 / 143.0 | 143.0 |
  | qwen3.8-27b-q5_k_xl | 422.0 | 481.6 / 509.7 / 514.0 | 509.7 |
  | qwen3.8-27b-q6_k | 818.5 | 1206.6 / 1210.6 / 1207.3 | 1207.3 |
  | qwen3-32b-q5_k_m | 185.7 | 401.0 / 431.7 / 428.3 | 428.3 |
  | qwen3-coder-30b-a3b-q6_k | 44.0 | 181.3 / 179.3 / 181.2 | 181.2 |
  | qwen3.5-35b-a3b-q6_k | 48.3 | 188.5 / 188.2 / 187.6 | 188.2 |
  | poro2-70b-q4_k_m | 199.0 | 1164.7 / 1183.6 / 1186.1 | 1183.6 |
  | gpt-oss-120b-mxfp4 | 116.5 | 365.3 / 366.8 / 393.8 | 366.8 |

  It is a **larger** number than the warm column it replaced (+21…+495% on Vulkan, +58…+386% on
  ROCm) because a warm request re-used the prompt cache and evaluated one token, while a cold
  request prefills all 46. That is the trade for stability: across the 20 cell/backend pairs the
  warm column's median per-request spread was 53 ms (max 823 ms, `ttft_ms`/`p50` up to 2.11),
  cold is 3 ms median (max 51 ms, ratio ≤ 1.04). The protocol does not move `server tg`: across
  the four published cold passes it sits within ±0.4% per cell of the warm runs on ROCm and
  within ±0.3% on Vulkan apart from one cell (−3.2%, Qwen3-Coder-30B-A3B) — the erase changes
  what TTFT measures and nothing else. What it costs is reproducibility at the cell level: two
  quiet passes per backend move `p50` by up to −6.2% (Vulkan, Poro 2 8B) and −7.9% (ROCm,
  gpt-oss) on the worst cell, and by a 0.4–0.8% median (§3c table).

  What it does not fix is model-specific prefill cost: cold, the 27B Q6_K cell still takes
  **1.21 s** on Vulkan (0.36 s on ROCm) and the 70B 1.18 s (0.55 s), because prefilling 46 tokens
  on a 20–40 GiB dense model is genuinely that slow. Contention shows up in `server tg` first: the
  08:43 Vulkan pass (kept as `superseded-contended-20261024-0843`) read the 4B cell at 39.85 t/s
  and Q6_K at 8.91 t/s against 48.09 and 9.91 in the clean pass, while their p50 moved by 0.5%
  and 0.3%. The driver cannot detect another process taking the GPU — check `temp_c_max` and the
  per-request decode rates in `ttft.json` when a cell looks slow.

  **History — why the warm column was unusable.** Measured 2026-10-22 on the Qwen3.8-27B q5 cell
  (Vulkan, three 512-token requests, q8 KV): the server logs a **1-token** prompt eval of ~785 ms
  and TTFT is 787 ms on the first request, then 1707 / 1706 ms — an extra ~905 ms that the server
  attributes to neither prefill nor decode. Two single-variable runs isolate the trigger (values
  are TTFT / logged prompt-eval / difference):

  | condition | TTFT (3 requests) | prompt eval | TTFT − prompt eval |
  |---|---|---|---|
  | 512-token generations, q8 KV | 787 / 1707 / 1706 ms | 785 / 802 / 802 ms | 2 / 905 / 903 ms |
  | 32-token generations, q8 KV | 813 / 804 / 785 ms | 811 / 802 / 783 ms | 2 / 2 / 2 ms |
  | 512-token generations, f16 KV | 141 / 417 / 417 ms | 139 / 134 / 133 ms | 2 / 284 / 284 ms |

  So (a) the stall needed a **long preceding generation** — the prompt itself is not the trigger, it
  is the KV trim/rollback that follows 512 generated tokens; and (b) **quantised KV was ~85% of the
  cost** on this cell (base 790 → 135 ms, stall 905 → 284 ms), while generation rate was identical
  in every arm (10.46–10.52 tok/s). The same q8→f16 swap on the 32B cell changed nothing (overhead
  73 → 76 ms), so it was a **model-specific** cost, not a general Vulkan penalty. Those runs produced
  the `warm` column, which stays in the rollup under its own run ids (`run-20261022-*`,
  `run-20261023-*`) and can be reproduced with `--ttft-protocol warm`; it is not published. This
  supersedes the 2026-10-02 note that described "first request fast, next two slow": the
  discriminator is the preceding generation, not the prompt.
- **Numbers are stack-stamped, and the published tables were re-measured.** The tracked
  tables' bench, power and memory columns come from the 2026-10-22 sweeps (their streaming
  columns moved to the 2026-10-24 cold runs, §3c), all on kernel 7.2.10 / Mesa 26.2.4
  (`docs/env.md`).
  The 2026-10-02/03 rows were measured on kernel 7.2.8 / Mesa 26.2.3 and stay in the rollup
  under their own run ids. Measured over **all 20 cell/backend pairs** (not the two-cell
  pilot this bullet used to quote), the newer stack is **pp2048 2.6–31.3% lower** — the
  outlier is Poro 2 70B on Vulkan, −31.3% — and **tg512 inside −2.8…+0.9%**. Do not mix rows
  from two stacks into one comparison, and stamp the stack in any new `run.json`
  (the stamp reads `"unknown"` on a machine without `vulkaninfo`, e.g. a CI runner —
  that is the honest value, and the parsing that produces it is pinned without a GPU).

- **ROCm gfx1151** is seen natively (no override needed for a clean load). If a MoE/MXFP4 op
  misbehaves, re-run that backend with `--rocm-override` (sets
  `HSA_OVERRIDE_GFX_VERSION=11.5.0`); if it still fails, treat ROCm as a bonus axis per PLAN.
- **Throttling is data, not a failure.** Sustained 59 GB loads can heat the APU; `temp_c_max`
  and the power trace are logged so a throttled run is still a *reported* run — just read the
  sustained (not burst) tok/s.

## 8. Recording a run (tag is derived; the commit stays yours)

The driver never commits — an unattended sweep must not write to git — but it no longer
leaves the tag to memory either. At the end of a sweep that covers the whole matrix it:

1. derives the next free tag for the day (`next_run_tag()`: `run-YYYYMMDD-N`, continuing
   from the highest existing `N` for that date),
2. builds the message from the run (`tag_message()`: backend, cell count, harness 7-char,
   llama.cpp 9-char, kernel / Mesa, KV type, and whether it was probe-only),
3. stamps the name into `run.json` → `tag`, and
4. logs the two commands to run:

```
next tag: run-20261023-1
record it: git add results/ && git commit -m "results: run-20261023-1015 (vulkan)" \
  && git tag -a run-20261023-1 -m "Vulkan probe-only sweep — 10 cells, ..."
```

`--tag` creates the annotated tag directly **only when the tracked `results/` and `charts/`
already match HEAD** (`results_match_head()`); otherwise it refuses and prints the same
commands, because tagging HEAD with a rollup that is still uncommitted would point at a
commit that does not contain the numbers. That is not a theoretical worry: the 2026-10-22
sweeps shipped untagged for exactly that reason (hand-typed step, forgotten at the end of a
long run) and were tagged retroactively on 2026-10-23.

Before the fix, the message was whatever the operator typed; now every run's tag says which
harness, stack and KV type produced it.

CPU power state: the governor is deliberately **not** pinned — the runs are meant to describe
the box as it actually behaves, and `run.json` → `cpu_power` records the governor/EPP each run
measured under (see [env.md](env.md#cpu-power-state-measurement-condition)). The llama.cpp
commit is in `run.json` → `llama_cpp_commit`, the harness revision in `harness_commit`, and the
[env.md](env.md) fingerprint holds the rest.

## 9. Finnish capability axis — FIN-bench-v2 (`cf` / `p0`)

The sweeps above measure throughput; they say nothing about whether a model writes good
Finnish. That is measured separately with the official FIN-bench-v2 suite, driven by
`scripts/finbench.py`.

**Why it needs its own scorer.** Every multiple-choice task in FIN-bench-v2 is scored by the
loglikelihood of the *full choice text as a continuation* (`doc_to_choice: "{{ choices.text }}"`,
metrics `acc` + `acc_norm`). The pinned llama-server cannot produce those numbers: it refuses
`echo` outright (`tools/server/server-common.cpp`: *"Only no echo is supported"*), the native
`/completion` returns next-token probabilities only (measured: 1 `completion_probabilities`
entry for 22 evaluated prompt tokens), and the OpenAI-format `logprobs` caps at 20 alternatives.
The harness fork's `gguf` model backend is no help either — it expects the older
`text_offset` / `token_logprobs` response shape. So `tools/score_ll.cpp` links against the
pinned build's own `libllama` and computes the loglikelihoods directly.

Build (no new dependencies — the pinned build already ships `libllama.so` and `include/llama.h`):

```
L=~/.local/opt/llama.cpp
g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include tools/score_ll.cpp \
  -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o tools/score_ll
```

The scorer's loop — one context pass, a short pass per choice, and the KV rollback that
lets the next choice reuse the context — is pinned by `tests/test_score_ll.py`:
choice-order invariance, repeated-choice identity (the rollback's failure mode), and a
CPU golden. It skips unless the binary and `Qwen3-4B-Q4_K_M.gguf` are present:

```
/usr/bin/python3 -m unittest discover -s tests -p 'test_score_ll.py' -v
```

Run (`scripts/finbench.py` is stdlib + PyYAML + Jinja2; data comes from the HF datasets-server
REST API and is cached resumably under `~/.cache/finbench/`):

```
/usr/bin/python3 scripts/finbench.py --list
/usr/bin/python3 scripts/finbench.py --model poro2-8b-q8_0 --pilot 150   # fast pipeline check
/usr/bin/python3 scripts/finbench.py --model <cell> --run-id <id>
/usr/bin/python3 scripts/finbench.py --consolidate-only --run-id <id>
```

**Scope and grid.** Upstream publishes the five `multiple_choice` families at two formulations
(`cf`, `mcf`) × five prompt variants (`p0`–`p4`) = 50 task configs; the shipped table is
`cf`/`p0` for all five. One invocation runs exactly **one** (formulation, variant) pair:

```
/usr/bin/python3 scripts/finbench.py --formulation mcf --variant p0 --model <cell> --run-id <id>
```

The prompts are not paraphrased: `scripts/fetch_finbench_upstream.py` vendors the upstream
`doc_to_text` (and `description`) strings into `docs/finbench/upstream-<sha>.json`, and the
driver renders them with Jinja2. `mcf` is not just a different prompt — the scored choice set
changes for some families (belebele_fin scores the letters A–D, arc_c keeps the full option
text); the choice set follows the upstream base yaml's `doc_to_choice`.
`tests/test_finbench_upstream.py` pins the whole 5×2×5 grid, including two upstream quirks that
are deliberately replicated (scandisent/mcf/p3 spells "positivinen"; belebele/mcf numbers the
options 1–4 while scoring A–D).

The five families: `arc_c`, `belebele_fin`, `goldenswag`, `scandisent`, `sib200` — 7,296 items
per model at full split. `goldenswag` is scored on its `train` split (its test labels are
withheld upstream), while the other four use `test`. `squad_fi` and `truthfulqa` exist upstream
only as `gen`/`mc1`/`mc2`, so they are out of scope for this axis by construction rather than
by omission. Per-task detail lands in `results/finbench-<run>/<cell>/<task>.json` (gitignored);
`results/finbench-consolidated.csv` and `results/finbench-table.md` are tracked, and carry
`formulation`/`variant` columns.

**Pilots never touch the tracked rollup.** A `--pilot N` run writes `pilot-consolidated.csv`
and `pilot-table.md` inside its own run directory; only a full run rewrites the tracked files.

Metrics are the harness's own (`lm_eval/api/task.py`, `OUTPUT_TYPE == "multiple_choice"`):
`acc` = argmax of the summed choice loglikelihood; `acc_norm` = argmax of the loglikelihood
divided by the choice's **character** length.

Two caveats before quoting a number:

- **GPU and CPU numerics differ slightly.** Validated against the server's own next-token
  logprobs on CPU: agreement to ~3e-4 (` Su` -0.157315 vs -0.1571; ` Helsinki` -3.524064 vs
  -3.5238). On GPU the two implementations diverge by up to ~0.12 in logprob, because batch
  shapes and kernels differ — harmless for a task score, but a near-tie between two choices
  could flip. The whole axis runs one way (GPU, no `--fa`), so the numbers are internally
  consistent.
- **Not a drop-in lm-eval number.** Prompts, datasets and metric definitions match the fork;
  the scoring engine does not. Comparable-with-a-caveat, not leaderboard-submittable.

## 10. Embeddings axis — Finnish retrieval (`fi-2026-10`)

The set is committed, so this axis needs no network: 263 passages from 24 Finnish
Wikipedia articles (CC BY-SA 4.0, per-article sha256 in
`docs/retrieval/fi-2026-10/manifest.json`) and 30 hand-authored queries whose
qrels are derived from an answer string (`tests/test_retrieval_set.py` re-derives
them, so a label cannot drift).

```bash
# one model per run; the server is started and stopped by the driver
/usr/bin/python3 scripts/embed_eval.py \
    --model ~/.local/share/models/bge-m3-f16.gguf --name bge-m3 --backend vulkan
/usr/bin/python3 scripts/embed_eval.py \
    --model ~/.local/share/models/nomic-embed-text-v1.5.f16.gguf \
    --name nomic-embed-text-v1.5 --backend rocm

/usr/bin/python3 scripts/embed_eval.py --table       # rebuild results/embeddings-table.md
/usr/bin/python3 scripts/embed_eval.py --compare bge-m3@vulkan nomic-embed-text-v1.5@vulkan
```

Per-run detail lands in `results/embed-<stamp>-<name>/summary.json` (gitignored):
model sha256, backend, prefixes, corpus/query hashes, timings, metrics and the
rank of the first relevant passage per query. `results/embeddings-table.md` is
tracked and is what the README section reproduces.

Protocol notes:

- **Task prefixes are per model.** nomic-embed-text is trained with
  `search_query:` / `search_document:`; bge-m3 is not. The driver applies the
  right pair per model name (`--query-prefix` / `--doc-prefix` override) and
  records it — scoring nomic bare would turn a model comparison into a protocol
  difference.
- **Pooling is the model's own.** `--embedding` without `--pooling` lets llama.cpp
  use the model default (CLS for bge-m3, mean for nomic).
- **Both backends should agree on quality.** They did (identical metrics on
  Vulkan and ROCm for both models); a backend that changes hit@1 is a bug, not a
  speed difference.
- **30 queries is a small sample.** One question moving changes hit@1 by 0.033;
  quote the paired count (`--compare`) alongside the averages.
