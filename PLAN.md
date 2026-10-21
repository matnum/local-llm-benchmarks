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

## 2. Reference environment

Full fingerprint in [docs/env.md](env.md). Summary: AMD Ryzen AI Max+ 395
(Strix Halo, 16C/32T), Radeon 8060S iGPU with 96 GiB VRAM, 128 GB unified
LPDDR5X-8000 (~256 GB/s), CachyOS (Arch-based, kernel 7.2.8), Mesa 26.2.3 RADV
(Vulkan) and ROCm 7.2.4 (`gfx1151` natively), llama.cpp pinned at `5b59b83f4`.

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
