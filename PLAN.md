# Plan — Local LLM Benchmark Suite (Strix Halo)

Date: 2026-09-30 · Status: draft for review

## 1. Environment snapshot (verified on this machine, 2026-09-30)

The README's target hardware **is this box**. Verified facts:

| Item | Value | Note |
|---|---|---|
| CPU | AMD Ryzen AI Max+ 395, 16C/32T | as expected |
| iGPU | Radeon 8060S (RDNA 3.5), 40 CUs | `RADV STRIX_HALO` |
| iGPU VRAM | **96 GiB** (103,079,215,104 B) via `mem_info_vram_total` | within README's 64–96 GB target |
| System RAM | ~30 GiB (rest reserved for iGPU) | "keep 32 GB+ for system" roughly met |
| Storage | 2 TB NVMe, **~1.6 TB free** | fits full matrix (~215 GB models) |
| OS | **CachyOS** (Arch-based), kernel 7.2.8-1-cachyos | ⚠️ README says Ubuntu 24.04 — README needs correcting |
| Vulkan | Mesa 26.2.3 (RADV), `vulkaninfo` OK | backend A ready |
| ROCm | **7.2.4** at `/opt/rocm`, `rocminfo` sees `gfx1151` natively | backend B ready; ≥6.3 as README requires; `HSA_OVERRIDE_GFX_VERSION` likely unnecessary (keep as fallback) |
| Toolchain | git, cmake, gcc/g++, clang, hipcc/amdclang, python 3.14, pacman | no cargo (not needed) |

## 2. Assets that already exist (README says "Not started" — it's wrong)

1. **llama.cpp source tree** at `~/.local/opt/llama.cpp` (commit `5b59b83f4`), with three builds:
   - `build-vulkan/` — `GGML_VULKAN=ON` (Backend A, reference) — **working**: currently serving a model.
   - `build/` — `GGML_HIP=ON` (Backend B, ROCm), with `GGML_HIP_ROCWMMA_FATTN=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_NO_VMM=ON`.
   - `build-probe/` — second HIP build (probe; superseded by `build/`).
2. **Models on disk** (`~/.local/share/models/`, ~152 GB):
   - `gpt-oss-120b-MXFP4.gguf` (61 GB) — the headline "Big" cell ✓
   - `Qwen3.5-35B-A3B-Q6_K.gguf` (27 GB)
   - `Qwen3.8-27B-UD-Q5_K_XL.gguf` (20 GB) + `Qwen3.8-27B-UD-Q6_K.gguf` (21 GB)
   - `qwen3-coder-30b-a3b-instruct-q6_k.gguf` (24 GB)
3. A `llama-server` instance is **currently running** (port 11436, Qwen3.8-27B + MTP spec decode, ~23.5 GB VRAM in use) — must be stopped before sweeps.

**Gaps vs README deliverables:** no repo scaffolding, no benchmark harness, no fixed eval set, no results, no posts, most README-matrix models not downloaded, no power sampling.

## 3. Decision points (need a yes/no before Phase 1)

- **D1 — Matrix.** The README matrix predates what's on disk (no Qwen3.5/3.8). Options:
  - (a) Keep README matrix as-is, download ~215 GB of missing cells; on-disk models become bonus rows.
  - (b) Update matrix to current generation (e.g. replace Qwen3-30B-A3B Q4 with Qwen3.5-35B-A3B Q6 already on disk) — less download, more current numbers.
  - Suggested: (b) for the Qwen MoE/27B rows, (a) for the rest (Llama 3.1 8B, Qwen3-32B, Mistral Small 24B, Gemma 3 27B, Llama 3.3 70B, tiny tier) so the README story stays intact.
- **D2 — llama.cpp pin.** Use existing commit `5b59b83f4` for all runs (both backends), or update to latest release first? Suggested: update to latest once, rebuild both backends, record the new commit in the results metadata. gpt-oss MXFP4 support is already proven on this commit.
- **D3 — OS note in README.** Correct "Ubuntu 24.04" to "CachyOS (Arch-based) verified" (or "any recent Arch/Ubuntu").
- **D4 — HF token.** Gated repos (Llama 3.1/3.3, Gemma 3, Mistral) need `HF_TOKEN`. gpt-oss-120b is already downloaded, so a token may already exist — check `env` / `~/.cache/huggingface` in Phase 1.

## 4. Methodology

**Per cell (model × quant × backend):**
- `llama-bench` (from the matching build dir) for prompt-processing and generation throughput: fixed prompt tokens `-p {512, 2048}`, generation `-n 512`, `-r 3` repeats, `-ngl 99`, physical 16 threads (Vulkan) / `HSA` default (ROCm).
- TTFT + sustained tok/s: short `llama-server` session (OpenAI-compatible) with a Python client (stdlib only — avoid `llama-cpp-python` on py3.14): warmup request, then N=3 timed 512-token generations; record TTFT per request.
- **Power**: background sampler at 5 Hz → `rocm-smi --showpower --showtemp` to CSV; compute avg/W and tok/s/W. Optional: `amd-power-tune` (pacman) for SoC-level power; fallback is GPU-only power with a documented caveat (APU: CPU power is a significant share under load).
- **Memory**: `mem_info_vram_used` polled by the sampler (peak), plus llama-bench's own peak-VRAM report, plus system `free`.
- **Controlled conditions**: stop the running llama-server and any other GPU work; fix CPU governor; log GPU temp each run (throttling is a finding, not a bug).

**Fixed eval set (quality axis)** — `evals/`, version-controlled, ~50–100 items:
- EN: strict-JSON extraction, function-calling correctness, summarization, GSM8K-style math subset.
- FI: Finnish summarization, invoice-field extraction, grammar/translation, FI chat Q&A.
- Format: JSONL `{id, category, prompt, expected/check}`; scoring scripts in `evals/` (exact-match/JSON-parse for structured, LLM- or metric-based for open-ended — decide per category).
- Run against top 3–5 models from the perf sweep (not all 9–12 cells).

**Embeddings axis:** `bge-m3` vs `nomic-embed-text` on a small Finnish retrieval set (FI passages + query/answer pairs); score recall@k / MRR. Output decides the Finnish RAG stack.

**Reproducibility:** one command per sweep:
```
./scripts/run_sweep.sh --backend vulkan --matrix matrix.yaml --repeats 3
```
Driver stops GPU contention, launches power sampler, iterates cells, writes one row-set to `results/run-YYYYMMDD-HHMM/`, then `results/consolidated.csv` + `results/table.md` (rendered). Each sweep: git commit + tag `run-YYYYMMDD-N` including llama.cpp commit, driver version, env fingerprint.

## 5. Phased execution

| Phase | Work | Est. | Exit criteria |
|---|---|---|---|
| **0. Scaffold** (½ h) | `git init`; layout `scripts/ models/ results/ evals/ posts/ docs/`; `.gitignore` (no model blobs in git); `docs/env.md` with fingerprint; update README (D3, status) | 30 min | repo initialized, README corrected |
| **1. Toolchain + models** (mostly download) | Apply D2 (rebuild or pin both backends); verify both with `llama-simple` on a tiny model; `huggingface-cli` download of missing matrix cells (gated repos need D4) | 3–6 h wall (downloads dominate) | all matrix cells on disk; both backends load a model |
| **2. Harness v1 + smoke** (¼ d) | `scripts/run_sweep.sh`, power sampler, TTFT client, CSV/markdown renderers; smoke matrix = 1 tiny + 1 mid cell × both backends | 3–4 h | one clean end-to-end run producing valid CSV incl. tok/s, TTFT, peak VRAM, W |
| **3. Full Vulkan sweep** (evening) | all cells, backend A | 3–5 h | complete `results/run-*` for Vulkan |
| **4. Full ROCm sweep** (evening) | all cells, backend B; if gfx1151 instability → `HSA_OVERRIDE_GFX_VERSION=11.5.0` retry, then demote ROCm to "bonus axis" per README | 3–5 h | Vulkan-vs-ROCm comparison table |
| **5. Quality + embeddings** (1–2 d, parallel track) | build eval set (FI content is the slow part), scoring scripts, run top models; bge-m3 vs nomic-embed-text on FI retrieval set | 1–2 d | quality table + embedding verdict |
| **6. Publish** (½–1 d) | consolidated CSV + `table.md`; checkbox off in README; post drafts | 4–8 h | both post drafts committed |

Critical path: 0 → 1 → 2 → 3/4 (can run overnight) → 6. Phase 5 is independent and can proceed while sweeps run.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Sustained-load thermals on the APU (throttling mid-run) | log temps/power per run; treat sustained (not burst) tok/s as the metric; report throttled vs boosted where it matters |
| gpt-oss-120b (61 GB) + KV cache vs 96 GiB allocation at long ctx | cap ctx for that cell (e.g. 8K) unless 4K+ is the point; KV in q8_0 |
| ROCm gfx1151 instability on some ops (esp. MoE/MXFP4) | README already allows demoting ROCm to bonus axis; keep `build-probe` configs as reference |
| Gated HF downloads fail without token | check for existing token early (Phase 1, step 1); worst case drop one gated cell and note it |
| Py3.14 lacks `llama-cpp-python` wheels | harness uses CLI + server HTTP + stdlib only (already planned) |
| Current llama-server holding 23.5 GB VRAM / GPU at 77% | stop it (it serves `opencode` on 11436 — coordinate); driver auto-checks GPU is idle before each cell |
| Matrix drift (new Qwen releases keep landing) | freeze matrix in `matrix.yaml` at Phase 1; document model versions + file hashes in results metadata |

## 7. Deliverable mapping (README checkboxes)

- [ ] Benchmark script, one-command run → Phases 2–4 (`scripts/run_sweep.sh`)
- [ ] Vulkan vs ROCm comparison → Phase 4
- [ ] Results table CSV + markdown, repo others can run → Phases 2–4 (+ `docs/RUNNING.md` with OS notes for non-CachyOS users)
- [ ] Post: hardware + numbers → Phase 6
- [ ] Headline: "A 120B-class model on a 128 GB APU — what actually runs" → Phase 6 (gpt-oss-120b cell is the anchor; it's already on disk)
