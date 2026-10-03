# Local LLM Benchmark Suite

A reproducible benchmark harness for LLM inference on a unified-memory AMD APU.
It sweeps models × quantizations × backends on the real hardware and publishes
the numbers — generation and prompt-processing throughput, time-to-first-token,
memory, power, and tokens-per-watt.

**Headline result (2026-10-02):** gpt-oss-120b (native MXFP4, ~59 GB) runs on a
128 GB unified-memory AMD Ryzen AI Max+ 395 at ~50 tok/s generation — a
120B-class model on a 128 GB APU. Across the whole matrix, **Vulkan wins
sustained generation and ROCm wins prompt processing, 5–0 each way.**

## Hardware under test

The reference machine — every number in this repo was measured on it:

| Component | Spec |
|---|---|
| CPU | AMD Ryzen AI Max+ 395 (Strix Halo), 16 cores / 32 threads |
| iGPU | Radeon 8060S (RDNA 3.5, 40 CUs) |
| Memory | 128 GB unified LPDDR5X-8000 (~256 GB/s) |
| Storage | 2 TB NVMe |
| OS | CachyOS (Arch-based), kernel 7.2.8 |

The iGPU is the LLM workhorse (Vulkan or ROCm). Unified memory lets the iGPU
take most of the 128 GB — 96 GiB iGPU allocation verified
(`mem_info_vram_total` = 103,079,215,104 B), leaving ~30 GiB for the system. The
50 TOPS XDNA 2 NPU is not a practical LLM path today and is not benchmarked.

A dense model's generation ceiling is roughly 256 GB/s ÷ model size in GiB. MoE
models only move *active* weights through the FFN path, so they beat that
ceiling — the 120B MoE cell is the headline demonstration.

## Software

| Piece | Version |
|---|---|
| llama.cpp | pinned at `5b59b83f4` for all runs |
| Backend A — Vulkan | Mesa 26.2.3 RADV (`STRIX_HALO`), `-ngl 99`, 16 threads |
| Backend B — ROCm | ROCm 7.2.4 (HIP, ROCm-WMMA flash-attention); `gfx1151` detected natively |
| Driver | `scripts/run_sweep.py` — stdlib + PyYAML only |

Full machine fingerprint: [docs/env.md](docs/env.md).

## Benchmark matrix

Five GGUF models already on disk (no downloads during the run):

| Tier | Model | Quant | File size | Notes |
|---|---|---|---|---|
| Big | gpt-oss-120b (MoE) | native MXFP4 | ~59 GB | headline "120B on a 128 GB APU" cell |
| MoE mid | Qwen3.5-35B-A3B | Q6_K | ~27 GB | ~3B active |
| MoE mid | qwen3-coder-30b-a3b-instruct | Q6_K | ~23 GB | ~3B active |
| Mid | Qwen3.8-27B (Unsloth dynamic) | Q5_K_XL | ~19 GB | dense; quant-pair row |
| Mid | Qwen3.8-27B (Unsloth dynamic) | Q6_K | ~20.5 GB | dense; quant-pair row |

Machine-readable matrix plus run-parameter defaults: [matrix.yaml](matrix.yaml).

## Metrics per cell

- Generation tok/s (sustained, 512-token outputs) and prompt-processing tok/s
- TTFT (time to first token), peak iGPU and system memory
- Power draw (W) and tok/s per watt — the APU differentiator
- Reproducibility: fixed prompts and context lengths, N repeats, a git-tagged
  script, and the measured CPU power state (governor/EPP) stamped into `run.json`

## Results — full sweep (2026-10-02)

5 cells × 2 backends, `-ngl 99`, 3 repeats + 3 streaming 512-token requests per
cell; llama-bench at the model-default context, server context 8192. Full
per-cell detail (power, temperature, per-request TTFT): [results/table.md](results/table.md).

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

**Vulkan takes generation, ROCm takes prompt processing — 5–0 each way.** Vulkan
is faster on `tg512` on all five cells (ROCm trails by 1.7–18.3%), while ROCm is
faster on `pp2048` prefill on all five (by 2.0–16.2%). ROCm also runs hotter (up
to 88 °C vs 85 °C).

> Before quoting a TTFT, read [docs/RUNNING.md](docs/RUNNING.md) §7: on Vulkan
> the first timed request is much faster than the next two, so the mean alone is
> misleading — this table shows the median, and `table.md` prints both.

## Running it

Prerequisites, commands, and per-cell behavior: [docs/RUNNING.md](docs/RUNNING.md).
Short version:

```bash
python3 scripts/run_sweep.py --backend vulkan   # full 5-cell sweep
python3 scripts/run_sweep.py --backend rocm
python3 scripts/run_sweep.py --consolidate      # rebuild rollups, no GPU
python3 -m unittest discover -s tests -v        # run the test suite
```

The only dependency is PyYAML (`pip install -r requirements.txt`); the helper
scripts are stdlib-only. The driver refuses to start unless the GPU is idle
(stop any running llama-server first).

## Reproducing on another machine

The harness is portable, but `matrix.yaml` pins absolute model paths
(`~/.local/share/models/...`) and `scripts/run_sweep.py` pins the llama.cpp build
directory (`~/.local/opt/llama.cpp`). To run elsewhere, edit the `file:` entries
in `matrix.yaml` and the `BUILDS` map in `scripts/run_sweep.py` to point at your
GGUF files and llama.cpp build, then follow [docs/RUNNING.md](docs/RUNNING.md).

## Caveats (read before trusting a number)

- **Two contexts.** The pinned `llama-bench` has no context flag, so bench runs at
  the model-default context (4096) while `llama-server` runs at 8192. Both are
  recorded separately in `run.json` — don't compare a bench number to a server
  number as if they were the same context.
- **Power is GPU-only**, not full SoC draw, so `tok/s per watt` understates the
  energy cost of prompt processing.
- **TTFT spread is wide on Vulkan** (see the note above).
- **Throttling is data, not a failure** — temperature is logged per run.

## Repository layout

| Path | What |
|---|---|
| `scripts/run_sweep.py` | one-command sweep driver (stdlib + PyYAML) |
| `scripts/sampler.py`, `scripts/ttft_probe.py` | stdlib-only helpers (power/trace, TTFT probe) |
| `matrix.yaml` | benchmark matrix + run-parameter defaults |
| `docs/RUNNING.md` | operator guide |
| `docs/env.md` | machine fingerprint |
| `results/consolidated.csv`, `results/table.md` | tracked rollups |
| `tests/` | stdlib-unittest suite (parser, consolidate, defaults wiring) |
| `PLAN.md`, `REVIEW.md` | methodology decisions + independent review |

## Roadmap

- **Quality eval set** (EN/FI): a small version-controlled set — strict-JSON
  extraction, function-calling, summarization, a GSM8K-style math subset, plus
  Finnish summarization/extraction/translation/chat — scored against the five
  local models.
- **Embeddings:** bge-m3 vs nomic-embed-text on a Finnish retrieval set (decides
  the Finnish RAG stack) — deferred until downloads are unblocked.
- **Wider matrix** — the cells below need downloads and an HF token for the gated
  repos.

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
