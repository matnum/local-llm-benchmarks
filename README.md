# Local LLM Benchmark Suite

Tier 3 — micro project · **Priority 1 — build this first**

Sweeps models × quantizations × backends on the actual hardware and publishes the numbers everything else in this repo depends on.

## Hardware
AMD Ryzen AI Max+ 395 (Strix Halo): 16C/32T, Radeon 8060S iGPU (RDNA 3.5, 40 CUs), **128 GB unified LPDDR5X-8000 (~256 GB/s)**, 2 TB NVMe.

- Workhorse for LLMs: the **iGPU** (Vulkan or ROCm). The 50 TOPS XDNA 2 NPU is not a practical LLM path today — don't plan around it.
- Memory: unified — the iGPU can be allocated most of the 128 GB. Verified 2026-09-30: **96 GiB iGPU allocation** (`cat /sys/class/drm/card0/device/mem_info_vram_total` → 103,079,215,104 B), ~30 GiB left for the system. Verify per run.
- Ceiling rule of thumb (dense models): gen tok/s ≈ 256 GB/s ÷ model size in GB. MoE models only move *active* weights through the FFN path, so they beat this ceiling — attention and kernel overhead set the real floor.

## Software (verified on this box, 2026-09-30)
- OS: **Linux — CachyOS (Arch-based), kernel 7.2.8**. (This README originally said Ubuntu 24.04; the suite is OS-agnostic and both stacks below are verified on this OS — see `docs/env.md`.)
- llama.cpp: `~/.local/opt/llama.cpp` @ commit **`5b59b83f4`** — pinned for all runs (PLAN.md, decision D2).
- **Backend A (reference): Vulkan** — `~/.local/opt/llama.cpp/build-vulkan` (Mesa 26.2.3 RADV, `RADV STRIX_HALO`), `-ngl 99`, 16 physical threads.
- **Backend B: ROCm 7.2.4** — `~/.local/opt/llama.cpp/build` (HIP, ROCm-WMMA flash-attention, graphs on, VMM off). `rocminfo` sees **gfx1151 natively** — no `HSA_OVERRIDE_GFX_VERSION` needed; kept as a fallback if an op misbehaves.
- Ollama/LM Studio: for a later "deployment experience" comparison, not the benchmark path.
- **Benchmark driver** (`scripts/run_sweep.py`): stdlib + **PyYAML** only (`requirements.txt`). The `sampler.py`/`ttft_probe.py` helpers are stdlib-only. Run with a python that has PyYAML (system `/usr/bin/python3` has it): `python3 scripts/run_sweep.py --backend {vulkan,rocm}`.

## Benchmark matrix (GGUF — on-disk models only, per decision D1)
| Tier | Model | Quant | File size | Notes |
|---|---|---|---|---|
| Big | gpt-oss-120b (MoE) | native MXFP4 | ~59 GB | headline "120B on a 128 GB APU" cell |
| MoE mid | Qwen3.5-35B-A3B | Q6_K | ~27 GB | ~3B active |
| MoE mid | qwen3-coder-30b-a3b-instruct | Q6_K | ~23 GB | ~3B active |
| Mid | Qwen3.8-27B (UD) | Q5_K_XL | ~19 GB | dense; quant-pair row |
| Mid | Qwen3.8-27B (UD) | Q6_K | ~20.5 GB | dense; quant-pair row |

Machine-readable version: `matrix.yaml`. Models live in `~/.local/share/models/`. No downloads in this round — the original wider matrix is kept below as future cells (needs downloads + an HF token for the gated repos).

### Future cells (needs download + HF token — not in the current run)
| Tier | Model | Quant | Size (approx) |
|---|---|---|---|
| Small / routing | gpt-oss-20b (MoE) | native MXFP4 | ~13 GB |
| Small / routing | Qwen3-30B-A3B (MoE) | Q4_K_M | ~18 GB |
| Tiny | Qwen3-4B / Phi-4-mini 3.8B | Q8_0 | ~4–5 GB |
| Mid | Llama 3.1 8B | Q8_0 | ~8.5 GB |
| Mid | Qwen3-32B | Q5_K_M | ~20 GB |
| Mid | Mistral Small 3.2 24B | Q4_K_M | ~15 GB |
| Mid | Gemma 3 27B | Q4_K_M | ~17 GB |
| Big | Llama 3.3 70B | Q4_K_M | ~42 GB |

## Metrics per cell
- Generation tok/s (sustained, 512-token outputs), prompt-processing tok/s
- TTFT (time to first token), peak iGPU + system memory
- **Power draw (W) and tok/s per watt** — the APU differentiator for the content (desktop vs. this box at idle/load)
- Reproducibility: fixed prompts, fixed context lengths, N repeats, git-tagged script + results CSV, and the measured CPU power state (governor/EPP) stamped into `run.json`

## Fixed eval set (quality axis)
Small, version-controlled, ~50–100 items:
- EN: strict-JSON extraction, function-calling correctness, summarization, math subset (GSM8K-style)
- FI: Finnish summarization, Finnish extraction (invoice fields), FI grammar/translation, FI chat Q&A
- Embeddings: **bge-m3** (FI-supported) vs nomic-embed-text (EN-centric) on a Finnish retrieval set — this decides the Finnish RAG stack. **Deferred: needs model downloads** (no-download constraint / no HF token at the moment).

## Results — first full sweep (2026-10-02)

All 5 cells on both backends (Vulkan = `build-vulkan`, ROCm = `build`), `-ngl 99`,
llama-bench at the model-default context, server context 8192, 3 repeats + 3 streaming
512-token requests per cell. Every cell has a complete bench/server/TTFT/power record — no
gaps. Full per-cell metrics incl. power and temperature: [results/table.md](results/table.md).

| Cell | Quant | Backend | pp2048 t/s | tg512 bench t/s | server tg t/s | TTFT p50 ms | tok/s/W | VRAM GiB |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b-mxfp4 | MXFP4 (native) | vulkan | 780.42 | 55.12 | 53.18 | 143.9 | 0.62 | 59.09 |
| gpt-oss-120b-mxfp4 | MXFP4 (native) | rocm | 906.9 | 51.46 | 49.59 | 67.2 | 0.58 | 59.36 |
| qwen3-coder-30b-a3b-q6_k | Q6_K | vulkan | 1140.09 | 74.75 | 71.46 | 162.8 | 0.89 | 23.76 |
| qwen3-coder-30b-a3b-q6_k | Q6_K | rocm | 1288.43 | 63.2 | 58.59 | 22.5 | 0.68 | 24.12 |
| qwen3.5-35b-a3b-q6_k | Q6_K | vulkan | 1117.97 | 61.77 | 60.26 | 688.5 | 0.8 | 27.1 |
| qwen3.5-35b-a3b-q6_k | Q6_K | rocm | 1140.08 | 52.82 | 50.8 | 58.1 | 0.59 | 27.46 |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | vulkan | 328.65 | 10.6 | 10.52 | 1631.9 | 0.13 | 19.06 |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | rocm | 339.18 | 10.35 | 10.23 | 137.3 | 0.12 | 19.65 |
| qwen3.8-27b-q6_k | Q6_K | vulkan | 317.69 | 10.01 | 9.96 | 809.1 | 0.12 | 20.09 |
| qwen3.8-27b-q6_k | Q6_K | rocm | 336.27 | 9.84 | 9.73 | 150.0 | 0.11 | 20.59 |

**Vulkan takes generation, ROCm takes prompt processing — 5–0 each way.** Vulkan is faster on
`tg512` for every dense and MoE cell (ROCm trails by 1.7–15.5%), while ROCm is faster on
`pp2048` prefill on every cell (by 2.0–16.2%). ROCm also runs the package hotter (up to 88 C
vs 85 C). Before quoting a TTFT figure read `docs/RUNNING.md` §7: on Vulkan the first timed
request is much faster than the next two, so the mean alone is misleading — that is why this
table shows the median and `table.md` prints both.

## How to run
See **[docs/RUNNING.md](docs/RUNNING.md)** for the operator guide (stop the dev server,
commands, per-cell behavior, results layout, caveats). Short version:

```
/usr/bin/python3 scripts/run_sweep.py --backend {vulkan,rocm}          # full 5-cell sweep
/usr/bin/python3 scripts/run_sweep.py --backend rocm --cells qwen3.8-27b-q5_k_xl   # smoke
```

## Where things live
| Path | What |
|---|---|
| `scripts/run_sweep.py` | one-command sweep driver (stdlib + PyYAML) |
| `scripts/sampler.py`, `scripts/ttft_probe.py` | stdlib-only helpers (power/trace, TTFT probe) |
| `matrix.yaml` | frozen on-disk benchmark matrix (D1) |
| `docs/RUNNING.md` | operator guide — commands, results, caveats |
| `docs/env.md` | verified machine fingerprint + power/contention notes |
| `PLAN.md` | locked decisions D1–D4, methodology, phases, risks |
| `REVIEW.md` | independent review of the harness (B/I/M issues) |
| `results/consolidated.csv`, `results/table.md` | tracked rollups (raw `results/run-*/` is gitignored) |
| `results/superseded-pre-fix-20260930-2239/` | kept for reference, gitignored: measured server-side numbers predate the probe fix |

## Deliverables
- [x] Benchmark script, one-command run → `scripts/run_sweep.py` (built; clean run pending)
- [x] Vulkan vs ROCm backend comparison → both 5-cell sweeps done 2026-10-02 (runs `run-20261002-1734` / `run-20261002-1751`); Vulkan takes generation, ROCm takes prefill
- [x] Results table (CSV + rendered markdown) + repo others can run → `consolidate()` + `docs/RUNNING.md` (11 cells: 5 × both backends, plus a repeat rocm cell)
- [ ] Post: hardware + numbers (evergreen "own hardware" content asset)
- [ ] Headline post candidate: "A 120B-class model on a 128 GB APU — what actually runs"

## Status
In progress — plan in [PLAN.md](PLAN.md). Toolchain, builds, and all five matrix models are
provisioned (2026-09-30). The harness is **implemented, reviewed, and its blocking/important
issues fixed** on branch `fix/harness-b1-b3-i1-i4` (see [REVIEW.md](REVIEW.md)).

A **first clean end-to-end cell landed 2026-10-02** (`run-20261002-1719`, rocm,
`qwen3.8-27b-q5_k_xl` — see [results/table.md](results/table.md)); it exercised the
`llama-bench` table parser, the SSE probe and the power sampler against the pinned build.
Getting there exposed two further defects, both fixed: the probe's `gen_tps` divided by a
duration-minus-absolute-clock (reporting ~0.01 t/s for every model and silently zeroing
`tok_per_w`), and it did not set `ignore_eos`, so a model that ends its turn on a raw
completion prompt returned no text at all.

**The full sweep is done (2026-10-02): 5 cells × 2 backends, 11 valid cells, no integrity
gaps** — every cell has bench, server-side generation, TTFT, power and tok/s/W. Headline
result above. Next: the Finnish axis (Poro models — needs a D1 revision, it requires
downloads) and the open Vulkan TTFT-spread question.
