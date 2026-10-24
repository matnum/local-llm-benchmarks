# Design & methodology — Local LLM Benchmark Suite

Decisions locked 2026-09-30. This documents the design decisions and measurement
methodology behind the benchmark. Results live in the [README](../README.md); the
operator guide is [docs/RUNNING.md](RUNNING.md); the machine fingerprint is
[docs/env.md](env.md).

## 1. Design decisions (locked 2026-09-30)

| # | Decision | Consequence |
|---|---|---|
| D1 | **On-disk models only — download nothing.** | The matrix is 5 GGUF files already in `~/.local/share/models/` (`matrix.yaml`). The wider matrix is demoted to "future cells". No network model pulls during a run. |
| D2 | **Pin the existing llama.cpp** (`~/.local/opt/llama.cpp` @ `5b59b83f4`). | No update/rebuild mid-project. `build-vulkan` = Backend A (reference), `build` = Backend B (ROCm/HIP). |
| D3 | **Document the environment as verified, never assumed.** | OS, matrix, and hardware are stated as actually measured on the reference machine (CachyOS, on-disk matrix), not from the vendor sheet. |
| D4 | **No HF token at the moment.** | Gated-repo cells stay in "future cells". Embedding models (bge-m3, nomic-embed-text) are ungated but still blocked by D1 → embeddings axis deferred. |
| D5 | **FIN-bench prompts come from the upstream templates verbatim** (2026-10-21). `scripts/fetch_finbench_upstream.py` vendors the `doc_to_text`/`description` strings from `LumiOpen/lm-evaluation-harness` @ `finbench_v2` into `docs/finbench/upstream-<sha>.json`; the driver renders them with Jinja2 instead of hand-copying prose. | A prompt can never drift from upstream by transcription, and the whole 5×2×5 grid is reachable (`--formulation`, `--variant`). One pair per run; `summary.json` stamps formulation, variant and upstream commit. Tests pin the grid, including upstream quirks (scandisent/mcf/p3 "positivinen"; belebele/mcf scores A–D while numbering 1–4). A `--pilot` run writes its rollup inside the run directory so a sample can never replace the tracked table. |
| D6 | **TTFT is reported as measured, with its mechanism documented, until the protocol is fixed** (2026-10-22; fixed by **D10** on 2026-10-24). Four controlled runs (27B and 32B, Vulkan) showed the inflation is KV-cache work after a long generation, and quantisation-dependent on the 27B: q8 KV → ~0.79 s base + ~0.9 s stall per request, f16 → ~0.14 s + ~0.28 s; the 32B is unaffected by KV type. | The `ttft_ms`/`ttft_ms_p50` columns stay as measured but are explicitly not comparable across cells or backends (README + RUNNING §7). A future TTFT axis must state its KV type; options are f16 KV or a cold-request protocol, followed by a ten-cell re-probe. |
| D7 | **Published numbers are stack-stamped** (2026-10-22). The tracked tables were measured on kernel 7.2.8 / Mesa 26.2.3; the machine now runs 7.2.10 / 26.2.4, where re-checked cells are 8–11% lower on `pp2048` and unchanged on `tg512`. | New runs must not be mixed into the published tables without re-measuring the rest, and `run.json` should carry the stack versions alongside the llama.cpp commit. |
| D8 | **The published tables are re-measured after the stack change, and the prose moves with the data** (2026-10-22/23). All ten cells ran on both backends on kernel 7.2.10 / Mesa 26.2.4 after the GPU-memory fix. Measured over the full grid the newer stack is `pp2048` **2.6–31.3% lower** (Poro 2 70B Vulkan is the −31.3% outlier) and `tg512` inside **−2.8…+0.9%** — D7's two-cell pilot figure understated it. The morning pair measured before the memory fix is excluded as `superseded-pre-mem-fix-*`. | The rollup keeps both stacks under separate run ids; README tables, chart stamps, `table.md` and the hand-written prose are regenerated from the new rollup in one commit, so no table mixes stacks. The published comparisons now carry **direction counts and ties**: `tests/test_readme_ranges.py` pins "N of 10" against the rollup and treats a gap under 2% as a tie — which is what the Qwen3.5-35B prefill cell (0.7%) became. |
| D9 | **The TTFT column's protocol is f16 KV, and the ten-cell re-probe turned out to be a reproducibility measurement, not a protocol change** (2026-10-23). Both 2026-10-22 sweeps already ran `--kv f16` — `sweep.log` records the argv, `run.json` stamps `kv_cache` — so D6's q8 arm was a single-variable experiment, never the published protocol. The probe-only re-probe (`run-20261023-1140` Vulkan, `-1159` ROCm, same kernel/Mesa/llama.cpp) reproduced the column: `tg_server_tps` within **−0.2…+1.4%** (mean abs 0.18%), `ttft_ms` **−4.4…+6.5%**, `ttft_ms_p50` **−5.9…+8.1%** (mean abs 1.7% / 2.0%) over 20 cell/backend pairs. | The column is labelled **f16 KV** wherever it is published, and its `server tg`/TTFT cells now carry the probe run ids while bench, power and memory stay with the full sweeps. The residual stall is protocol-inherent: f16 cuts the 27B's post-generation KV bookkeeping from ~0.9 s to ~0.3–0.8 s but does not remove it (Q6_K: 818/1641/818 ms), so the median stays the quoted figure and cross-cell comparison stays invalid. The re-probe also converts the README's "~2% is noise" assertion into a measured band. **Superseded by D10** for the protocol itself: the KV type was never what made the column unstable. |
| D10 | **TTFT is measured cold: every timed request starts from an erased KV cache, on both backends** (2026-10-24). `--ttft-protocol cold` makes `ttft_probe.py` POST `/slots/0?action=erase` before each timed request (untimed, needs `--slot-save-path`), so the number is full 46-token prompt prefill + first token and nothing else. Measured on ten cells × two backends, twice per backend (`run-20261024-0910` → `-1812` Vulkan, `-0930` → `-1749` ROCm): per-request spread collapses from a median 53 ms (max 823, `ttft_ms`/`p50` up to 2.11) to 3 ms (max 51, ratio ≤ 1.04); `server tg` is unchanged by the protocol (±0.4% per cell on ROCm, ±0.3% on Vulkan apart from one cell at −3.2%); the cold column reads **+21…+495%** (Vulkan) / **+58…+386%** (ROCm) above the warm one; and the two passes per backend put the run-to-run band at −6.2…+1.5% (Vulkan) / −7.9…+0.2% (ROCm) on cell `TTFT p50`, median |Δ| 0.8% / 0.4%. | The published column is cold and labelled `[f16 KV, cold TTFT]` on every streaming row; `matrix.yaml` defaults now carry `ttft_protocol: cold` **and** `kv: f16`, so an all-default sweep reproduces the published protocol instead of drifting from it (the D9 lesson). `cache_prompt: false` was rejected — it re-prefills but still pays the previous generation's KV eviction, so it is both slower and as noisy. Cold does **not** equalise the cells: the 27B Q6_K cell stays at 1.21 s on Vulkan (0.36 s ROCm) and the 70B at 1.18 s (0.55 s), because prefill cost is model-specific; and contention still corrupts a pass (one pass was renamed `superseded-contended-*` after reading 4B `tg` 17% low). The warm column stays in the rollup under its run ids for reproducibility (`--ttft-protocol warm`), and the run-to-run latency band is measured on both backends (§3c table), with three cells moving 3–8% between two quiet passes — a per-cell cold median is good to ~8% on the worst cell, not the ~2% the bench repeats support. |

## 2. Reference environment

Full fingerprint in [docs/env.md](env.md). Summary: AMD Ryzen AI Max+ 395
(Strix Halo, 16C/32T), Radeon 8060S iGPU (1 GiB VRAM carve-out + 104 GiB GTT pool
since 2026-10-22), 128 GB unified LPDDR5X-8000 (~256 GB/s), CachyOS (Arch-based,
kernel 7.2.10), Mesa 26.2.4 RADV (Vulkan) and ROCm 7.2.4 (`gfx1151` natively),
llama.cpp pinned at `5b59b83f4`.

## 3. Methodology

**Per cell (5 cells × 2 backends at planning time; the matrix now covers ten):**

- **`llama-bench`** (matching build dir): `-p {512, 2048}` prompt processing, `-n 512`
  generation, `-r 3` repeats, `-ngl 99` → pp/tg tok/s + peak VRAM. Context note: the
  pinned `llama-bench` has **no `-c/--ctx` flag**, so bench runs at the model-default
  context (4096); `llama-server` (the TTFT/sustained probe) takes `-c` and runs at
  `--ctx` (8192). `run.json` records both `ctx_bench` and `ctx_server`.
- **TTFT + sustained tok/s**: `llama-server` (OpenAI-compatible) + Python stdlib client —
  1 untimed warmup request, then 3 timed 512-token streaming generations; TTFT per request.
- **Power**: a 5 Hz sampler reads the amdgpu hwmon via sysfs (`power1_average` µW→W,
  `temp1_input` m°C→°C); `rocm-smi` is a per-cell cross-check. GPU-only power (no
  `amd-power-tune` for the SoC level) — documented caveat.
- **Memory**: `mem_info_vram_used` polled by the sampler (peak) + the llama-bench peak
  report + system `free`.
- **Controlled conditions**: the GPU must be idle before each cell (the driver refuses
  otherwise); the CPU power state is recorded, not pinned, and stamped into `run.json`;
  temperatures are logged (throttling is data).

**Quality eval set (planned)** — `evals/`, ~50–100 items, JSONL:

- EN: strict-JSON extraction, function-calling correctness, summarization, GSM8K-style math.
- FI: Finnish summarization, invoice-field extraction, grammar/translation, chat Q&A.
- Embeddings (bge-m3 vs nomic-embed-text on a Finnish retrieval set): **deferred** (D1/D4).

**Reproducibility:** one command per sweep (`python3 scripts/run_sweep.py --backend
{vulkan,rocm}`). The driver verifies the GPU is idle → starts the power sampler →
iterates cells (llama-bench + server TTFT probe) → writes `results/run-YYYYMMDD-HHMM/` →
rebuilds `results/consolidated.csv` + `results/table.md`. Each run records the llama.cpp
commit and the environment fingerprint; committing/tagging the run is done by hand for
now (see `docs/RUNNING.md` §8).

## 4. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Sustained-load thermals (throttling mid-run) | log temps/power per run; report sustained (not burst) tok/s |
| gpt-oss-120b (59 GB) + KV @ 8192 ctx vs 96 GiB | fits with margin; if not, drop ctx to 4K for that cell and note it |
| ROCm gfx1151 instability (MoE/MXFP4 ops) | `--rocm-override` (`HSA_OVERRIDE_GFX_VERSION=11.5.0`), else demote ROCm to a bonus axis |
| A running `llama-server` holding VRAM | stop it before sweeps; the driver hard-fails if the GPU is not idle |
| GPU-only power understates SoC draw during pp | documented as a methodology caveat; optionally add `amd-power-tune` later |
| Matrix drift (new model releases) | matrix frozen in `matrix.yaml`; file sizes + model names recorded in run metadata |
