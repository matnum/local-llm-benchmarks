# Local LLM Benchmark Suite

Tier 3 — micro project · **Priority 1 — build this first**

Sweeps models × quantizations × backends on the actual hardware and publishes the numbers everything else in this repo depends on.

## Hardware
AMD Ryzen AI Max+ 395 (Strix Halo): 16C/32T, Radeon 8060S iGPU (RDNA 3.5, 40 CUs), **128 GB unified LPDDR5X-8000 (~256 GB/s)**, 2 TB NVMe.

- Workhorse for LLMs: the **iGPU** (Vulkan or ROCm). The 50 TOPS XDNA 2 NPU is not a practical LLM path today — don't plan around it.
- Memory: unified — the iGPU can be allocated most of the 128 GB. Target a **64–96 GB iGPU allocation** (BIOS/firmware "UMA framebuffer" if exposed, else kernel tuning); keep 32 GB+ for the system. Verify per run: `cat /sys/class/drm/card0/device/mem_info_vram_total`.
- Ceiling rule of thumb: gen tok/s ≈ 256 GB/s ÷ model size in GB (attention overhead aside).

## Software
- OS: **Linux (Ubuntu 24.04)** — practical path for llama.cpp/Ollama/ROCm.
- **Backend A (reference): llama.cpp Vulkan** (`-ngl 99`, 16 physical threads).
- **Backend B: ROCm (gfx1151)** — Strix Halo may need recent ROCm (6.3+) or `HSA_OVERRIDE_GFX_VERSION=11.5.0`; if unstable, Vulkan stays the reference and ROCm becomes a bonus axis, not a blocker.
- Ollama/LM Studio: for a later "deployment experience" comparison, not the benchmark path.

## Benchmark matrix (GGUF unless noted)
| Tier | Model | Quant | Size (approx) | Planning tok/s (VALIDATE — these are estimates) |
|---|---|---|---|---|
| Small / routing | gpt-oss-20b (MoE) | native MXFP4 | ~13 GB | 25–45 |
| Small / routing | Qwen3-30B-A3B (MoE) | Q4_K_M | ~18 GB | 20–35 |
| Tiny | Qwen3-4B / Phi-4-mini 3.8B | Q8_0 | ~4–5 GB | 40+ |
| Mid | Llama 3.1 8B | Q8_0 | ~8.5 GB | 20–30 |
| Mid | Qwen3-32B | Q5_K_M | ~20 GB | 7–10 |
| Mid | Mistral Small 3.2 24B | Q4_K_M | ~15 GB | 8–14 |
| Mid | Gemma 3 27B | Q4_K_M | ~17 GB | 8–12 |
| Big | Llama 3.3 70B | Q4_K_M | ~42 GB | 4–7 |
| Big | gpt-oss-120b (MoE) | native MXFP4 | ~61 GB | 4–8 |

Planning numbers assume ~256 GB/s unified bandwidth plus community reports for Strix Halo. The suite exists to replace them with measured values.

## Metrics per cell
- Generation tok/s (sustained, 512-token outputs), prompt-processing tok/s
- TTFT (time to first token), peak iGPU + system memory
- **Power draw (W) and tok/s per watt** — the APU differentiator for the content (desktop vs. this box at idle/load)
- Reproducibility: fixed prompts, fixed context lengths, N repeats, git-tagged script + results CSV

## Fixed eval set (quality axis)
Small, version-controlled, ~50–100 items:
- EN: strict-JSON extraction, function-calling correctness, summarization, math subset (GSM8K-style)
- FI: Finnish summarization, Finnish extraction (invoice fields), FI grammar/translation, FI chat Q&A
- Embeddings: **bge-m3** (FI-supported) vs nomic-embed-text (EN-centric) on a Finnish retrieval set — this decides the Finnish RAG stack

## Deliverables
- [ ] Benchmark script (llama.cpp CLI or Python bindings), one-command run
- [ ] Vulkan vs ROCm backend comparison
- [ ] Results table (CSV + rendered markdown) + repo others can run
- [ ] Post: hardware + numbers (evergreen "own hardware" content asset)
- [ ] Headline post candidate: "A 120B-class model on a 128 GB APU — what actually runs"

## Status
Not started.
