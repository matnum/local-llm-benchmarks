# Plan — Local LLM Benchmark Suite (Strix Halo)

Date: 2026-09-30 · Status: decisions locked (D1–D4), Phase 0 in progress

## 1. Decisions (locked 2026-09-30)

| # | Decision | Consequence |
|---|---|---|
| D1 | **Update matrix to on-disk models; download nothing** | Matrix = 5 GGUF files already in `~/.local/share/models/` (see `matrix.yaml`). Original wider matrix demoted to "future cells" in README. No network model pulls in this round. |
| D2 | **Pin existing llama.cpp** (`~/.local/opt/llama.cpp` @ `5b59b83f4`) | No update/rebuild. `build-vulkan` = Backend A (reference), `build` = Backend B (ROCm/HIP). Do not disturb the source tree or the running server's build. |
| D3 | **Correct README** | Done: OS line now says CachyOS (verified), matrix replaced, status updated. |
| D4 | **No HF token at the moment** | All gated-repo cells stay in "future cells". Embedding models (bge-m3, nomic-embed-text) are ungated but still blocked by D1 (no downloads) → embeddings axis deferred. |

## 2. Environment snapshot (verified on this machine)

The README's target hardware **is this box**:

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI Max+ 395, 16C/32T |
| iGPU | Radeon 8060S, **96 GiB VRAM** (`mem_info_vram_total` = 103,079,215,104 B), ~30 GiB system RAM |
| Storage | 2 TB NVMe, ~1.6 TB free |
| OS | **CachyOS** (Arch-based), kernel 7.2.8 |
| Vulkan | Mesa 26.2.3 RADV `STRIX_HALO` — working |
| ROCm | **7.2.4**, `rocminfo` sees `gfx1151` natively — no override env needed |
| llama.cpp | `build-vulkan` (A, proven — currently serving) + `build` (B, HIP) |

Full fingerprint: `docs/env.md`.

## 3. Methodology

**Per cell (5 cells × 2 backends = 10 runs):**
- `llama-bench` (matching build dir): `-p {512, 2048}` prompt processing, `-n 512` generation, `-r 3` repeats, `-ngl 99`, ctx 8192. Yields pp/tg tok/s + peak VRAM.
- TTFT + sustained tok/s: `llama-server` (OpenAI-compatible) + Python stdlib client — warmup request, then 3 timed 512-token generations, TTFT per request.
- **Power**: 5 Hz sampler → `rocm-smi --showpower --showtemp` CSV; compute avg W and tok/s/W. Caveat documented: GPU-only power (no `amd-power-tune` for SoC level).
- **Memory**: `mem_info_vram_used` polled by sampler (peak) + llama-bench peak report + system `free`.
- **Controlled conditions**: GPU must be idle before each cell (the running `llama-server` on :11436 serving `opencode` must be stopped — coordinate with user); CPU governor fixed; temps logged (throttling is a finding).

**Fixed eval set (quality axis)** — `evals/`, ~50–100 items, JSONL:
- EN: strict-JSON extraction, function-calling correctness, summarization, GSM8K-style math.
- FI: Finnish summarization, invoice-field extraction, grammar/translation, chat Q&A.
- Run against the 5 local models (no extra downloads needed).
- Embeddings (bge-m3 vs nomic-embed-text, FI retrieval): **deferred** (D1/D4).

**Reproducibility:** one command per sweep:
```
./scripts/run_sweep.sh --backend vulkan --matrix matrix.yaml
```
Driver: verify GPU idle → start power sampler → iterate cells (llama-bench + server TTFT probe) → write `results/run-YYYYMMDD-HHMM/` → consolidate `results/consolidated.csv` + `results/table.md`. Every sweep: git commit + tag `run-YYYYMMDD-N` with llama.cpp commit + env fingerprint.

## 4. Phased execution (revised for no-downloads)

| Phase | Work | Est. | Exit criteria |
|---|---|---|---|
| **0. Scaffold + README** (½ h) | git init, layout, `.gitignore`, `docs/env.md`, README corrections (D1/D3), freeze `matrix.yaml` | 30 min | repo initialized, README corrected ✅ (this phase) |
| **1. Verify backends** (1 h) | Smoke: `llama-simple` on smallest cell (Qwen3.8-27B Q5_K_XL) via **ROCm build** (Vulkan already proven by the running server); confirm rocm-smi power sampling; record idle baseline | 1 h | both backends load + generate a model |
| **2. Harness v1 + smoke** (¼ d) | `scripts/run_sweep.sh`, power sampler, TTFT client, CSV/markdown renderers; 1 cell × both backends end-to-end | 3–4 h | one clean run producing valid CSV: tok/s, TTFT, peak VRAM, W |
| **3. Full Vulkan sweep** (evening) | 5 cells, backend A — **needs :11436 server stopped first** | 2–3 h | complete `results/run-*` (Vulkan) |
| **4. Full ROCm sweep** (evening) | 5 cells, backend B; gfx1151 instability → `HSA_OVERRIDE_GFX_VERSION=11.5.0` retry, else demote ROCm to bonus axis | 2–3 h | Vulkan-vs-ROCm comparison table |
| **5. Quality evals** (1 d, parallel track) | Build EN/FI eval set + scoring scripts; run all 5 models. (Embeddings deferred.) | 1 d | quality table per model × category |
| **6. Publish** (½ d) | Consolidated CSV + `table.md`; README checkboxes; post drafts | 4–8 h | both post drafts committed |

Critical path: 0 → 1 → 2 → 3/4 (overnights, after stopping :11436) → 6. Phase 5 is independent.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Sustained-load thermals (throttling mid-run) | log temps/power per run; report sustained (not burst) tok/s; throttling is data |
| gpt-oss-120b (59 GB) + KV @ 8192 ctx vs 96 GiB | fits with margin; if not, drop ctx to 4K for that cell only and note it |
| ROCm gfx1151 instability (MoE/MXFP4 ops) | Phase 1 smoke catches it early; fallback: override env var → demote to bonus axis (README already allows) |
| Running llama-server (:11436, ~23.5 GB VRAM, serves `opencode`) | coordinate stop before sweeps; driver hard-fails if GPU not idle |
| GPU-only power understates SoC draw during pp | document as methodology caveat; optional later: `pacman -S amd-power-tune` |
| Matrix drift (new model releases) | matrix frozen in `matrix.yaml`; file sizes + model names recorded in results metadata |

## 6. Deliverable mapping (README checkboxes)

- [ ] Benchmark script, one-command run → Phases 2–4 (`scripts/run_sweep.sh`)
- [ ] Vulkan vs ROCm comparison → Phase 4
- [ ] Results table CSV + markdown, repo others can run → Phases 2–4 (+ `docs/RUNNING.md`)
- [ ] Post: hardware + numbers → Phase 6
- [ ] Headline: "A 120B-class model on a 128 GB APU" → Phase 6 (gpt-oss-120b cell already on disk)
- (deferred) bge-m3 vs nomic-embed-text → needs download decision + possibly token
