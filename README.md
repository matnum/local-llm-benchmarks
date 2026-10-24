# Local LLM Benchmark Suite

Reproducible llama.cpp benchmarks for large language models on a single
unified-memory AMD APU — a Ryzen AI Max+ 395 ("Strix Halo") with 128 GB of
LPDDR5X. Ten model/quantization cells are measured on two backends (Vulkan and
ROCm) for throughput, latency, memory, power and tokens-per-watt, plus a
Finnish-capability axis on FIN-bench-v2.

Every number here was measured on the reference machine described below. The
drivers, the tracked rollups and the tests that re-derive the published
percentages are all in this repository, so the results can be checked rather
than trusted.

[![tests](https://github.com/matnum/local-llm-benchmarks/actions/workflows/tests.yml/badge.svg)](https://github.com/matnum/local-llm-benchmarks/actions/workflows/tests.yml)
[![license: MIT + CC BY 4.0](https://img.shields.io/badge/license-MIT%20%2B%20CC%20BY%204.0-blue.svg)](#license)
![llama.cpp](https://img.shields.io/badge/llama.cpp-5b59b83f4-informational.svg)

## At a glance

- **10 cells × 2 backends.** Three MoE models (3–120B total, ~3B active for the
  mid-size ones) and seven dense models from 4B to 70B, all in GGUF.
- **A 120B-class model on a 128 GB APU.** gpt-oss-120b (native MXFP4, ~59 GiB)
  runs at **55.3 tok/s** generation (llama-bench) and 53.9 tok/s sustained
  streaming, at 0.63 tok/s per watt, with 59.7 GiB resident.
- **Vulkan leads generation, ROCm leads prompt processing.** **Vulkan leads
  generation on 10 of 10 cells — decisively on 6, by 2.1–18.0%; ROCm leads
  prefill on 9 of 10, by 6.2–56.8%.** The other five cells are inside the ±2%
  noise floor of three repeats and read as ties.
- **MoE models are 5–8× faster than dense models of similar footprint** at
  generation, and 5–8× more efficient per watt.
- **Dense models sit at 73–85% of the memory-bandwidth ceiling** (256 GB/s ÷
  model size in GiB), including the 39.6 GiB Poro 2 70B at 81% on Vulkan.
- **Finnish capability:** Poro 2 70B leads the FIN-bench-v2 `cf` axis at 0.629
  mean accuracy; at matched Q4_K_M, Poro 2 8B beats the English-centric
  Qwen3-8B by 10 points (0.530 vs 0.430).

## Contents

- [What is measured](#what-is-measured)
- [Hardware](#hardware)
- [Software](#software)
- [Benchmark matrix](#benchmark-matrix)
- [Results](#results)
- [Charts](#charts)
- [Finnish capability](#finnish-capability--fin-bench-v2)
- [Quick start](#quick-start)
- [Reproducibility](#reproducibility)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Roadmap](#roadmap)

## What is measured

Each cell is one model at one quantization, on each backend, with `-ngl 99`
(every layer on the iGPU) and 16 CPU threads:

- **Generation throughput** — `llama-bench` `tg512` (mean of 3 repeats) and a
  sustained 512-token stream from `llama-server`.
- **Prompt processing** — `llama-bench` `pp512` and `pp2048`.
- **Time to first token** — three streaming requests; mean and median are both
  recorded, and the median is the one to quote (see [Limitations](#limitations)).
- **Memory** — peak resident GPU memory (VRAM + GTT; the weights live in GTT on
  this box) and system availability, sampled at 5 Hz.
- **Power and efficiency** — GPU power during the bench and server phases, and
  tokens per watt.
- **Reproducibility metadata** — the pinned llama.cpp commit, the driver commit,
  CPU governor/EPP, both context sizes and the actual repeat counts are stamped
  into each run's `run.json`.

## Hardware

The reference machine — every number in this repository was measured on it:

| Component | Spec |
|---|---|
| CPU | AMD Ryzen AI Max+ 395 (Strix Halo), 16 cores / 32 threads |
| iGPU | Radeon 8060S (RDNA 3.5, 40 CUs), `gfx1151` |
| Memory | 128 GB unified LPDDR5X-8000 (~256 GB/s) |
| Storage | 2 TB NVMe |
| OS | CachyOS (Arch-based), kernel 7.2.10 at measurement time |

Unified memory lets the iGPU address the bulk of the 128 GB. The 2026-10-22
runs — the ones behind every table here — were measured with a **1 GiB VRAM
carve-out** (`mem_info_vram_total` = 1,073,741,824 B) and a **104 GiB GTT pool**
(`mem_info_gtt_total`), so model weights are resident in GTT and the rollup
publishes VRAM + GTT as peak memory. The 2026-10-02/03 runs predate that
configuration (96 GiB carve-out, `mem_info_vram_total` = 103,079,215,104 B,
~30 GiB left to the system) and are kept as their own rows in the rollup. The
50 TOPS XDNA 2 NPU is not a practical LLM path today and is not benchmarked.

Machine fingerprint — driver and library versions, power source, kernel —
is in [docs/env.md](docs/env.md).

## Software

| Piece | Version |
|---|---|
| llama.cpp | pinned at `5b59b83f4` for all runs |
| Backend A — Vulkan | Mesa 26.2.4 RADV (`STRIX_HALO`), `-ngl 99`, 16 threads |
| Backend B — ROCm | ROCm 7.2.4 (HIP, ROCm-WMMA flash-attention); `gfx1151` detected natively |
| Driver | `scripts/run_sweep.py` — Python 3, stdlib + PyYAML |

The harness measures the model server and the build directly; there is no
`llama-cpp-python` in the loop.

## Benchmark matrix

Ten cells, all GGUF, each run on both backends. Machine-readable form plus the
run-parameter defaults: [matrix.yaml](matrix.yaml).

| Cell | Model | Quant | Size | Arch | Notes |
|---|---|---|---|---|---|
| `qwen3-4b-q8_0` | Qwen3-4B | Q8_0 | 4.0 GiB | dense | smallest cell; prefill reference |
| `poro2-8b-q8_0` | Poro 2 8B | Q8_0 | 8.0 GiB | dense | Finnish-focused |
| `eurollm-9b-q8_0` | EuroLLM-9B-Instruct | Q8_0 | 9.1 GiB | dense | European-language generalist |
| `qwen3.8-27b-q5_k_xl` | Qwen3.8-27B (Unsloth dynamic) | Q5_K_XL | 19.4 GiB | dense | quant pair |
| `qwen3.8-27b-q6_k` | Qwen3.8-27B (Unsloth dynamic) | Q6_K | 20.5 GiB | dense | quant pair |
| `qwen3-32b-q5_k_m` | Qwen3-32B | Q5_K_M | 21.6 GiB | dense | largest dense Qwen |
| `qwen3-coder-30b-a3b-q6_k` | Qwen3-Coder-30B-A3B-Instruct | Q6_K | 23.4 GiB | MoE | ~3B active |
| `qwen3.5-35b-a3b-q6_k` | Qwen3.5-35B-A3B | Q6_K | 26.9 GiB | MoE | ~3B active |
| `poro2-70b-q4_k_m` | Poro 2 70B | Q4_K_M | 39.6 GiB | dense | Finnish-focused; largest dense cell |
| `gpt-oss-120b-mxfp4` | gpt-oss-120b | MXFP4 (native) | 59.0 GiB | MoE | headline "120B on 128 GB" cell |

## Results

Both backends, every cell, rendered from the tracked rollup
[results/consolidated.csv](results/consolidated.csv). Rows are ordered by model
size; `server tg` is the sustained 512-token stream from `llama-server`, and
`TTFT p50` is the median first-token latency of three streaming requests, each
one starting from an **erased KV cache** (the cold-request protocol). Columns come
from two passes on the same stack, each row carrying its own run id in the
rollup: `pp2048`, `tg512`, `tok/s/W` and `peak GPU` from the 2026-10-22 full
sweeps, `server tg` and `TTFT p50` from the 2026-10-24 cold re-measure
([why](docs/RUNNING.md#3c-the-cold-request-ttft-protocol)). Full per-cell detail —
power, temperature, per-request TTFT, benchmark context — is in
[results/table.md](results/table.md).

| Cell | Backend | pp2048 t/s | tg512 t/s | server tg t/s | TTFT p50 ms | tok/s/W | peak GPU GiB |
|---|---|---|---|---|---|---|---|
| `qwen3-4b-q8_0` | vulkan | 1857.9 | 48.89 | 48.09 | 55.3 | 0.57 | 5.77 |
| `qwen3-4b-q8_0` | rocm | 2329.5 | 46.83 | 46.23 | 43.5 | 0.54 | 5.53 |
| `poro2-8b-q8_0` | vulkan | 918.4 | 27.16 | 26.83 | 127.9 | 0.32 | 9.26 |
| `poro2-8b-q8_0` | rocm | 1239.1 | 26.59 | 26.48 | 64.4 | 0.31 | 8.89 |
| `eurollm-9b-q8_0` | vulkan | 791.0 | 23.57 | 23.25 | 140.9 | 0.27 | 10.67 |
| `eurollm-9b-q8_0` | rocm | 1060.1 | 22.91 | 22.90 | 85.3 | 0.27 | 10.30 |
| `qwen3.8-27b-q5_k_xl` | vulkan | 290.3 | 10.56 | 10.48 | 507.9 | 0.12 | 20.09 |
| `qwen3.8-27b-q5_k_xl` | rocm | 308.4 | 10.37 | 10.33 | 338.1 | 0.12 | 19.59 |
| `qwen3.8-27b-q6_k` | vulkan | 273.7 | 9.98 | 9.91 | 1211.4 | 0.12 | 21.13 |
| `qwen3.8-27b-q6_k` | rocm | 308.6 | 9.83 | 9.80 | 358.1 | 0.12 | 20.53 |
| `qwen3-32b-q5_k_m` | vulkan | 243.1 | 9.71 | 9.66 | 431.0 | 0.11 | 24.01 |
| `qwen3-32b-q5_k_m` | rocm | 292.9 | 9.54 | 9.51 | 225.0 | 0.11 | 23.68 |
| `qwen3-coder-30b-a3b-q6_k` | vulkan | 1110.0 | 74.86 | 73.55 | 179.6 | 0.90 | 24.39 |
| `qwen3-coder-30b-a3b-q6_k` | rocm | 1246.9 | 61.41 | 60.05 | 106.4 | 0.69 | 24.37 |
| `qwen3.5-35b-a3b-q6_k` | vulkan | 1062.4 | 62.30 | 61.12 | 186.9 | 0.80 | 27.63 |
| `qwen3.5-35b-a3b-q6_k` | rocm | 1054.7 | 51.32 | 50.53 | 129.2 | 0.59 | 27.40 |
| `poro2-70b-q4_k_m` | vulkan | 81.0 | 5.25 | 5.24 | 1179.4 | 0.06 | 42.52 |
| `poro2-70b-q4_k_m` | rocm | 127.1 | 5.18 | 5.18 | 545.4 | 0.06 | 42.15 |
| `gpt-oss-120b-mxfp4` | vulkan | 735.3 | 55.30 | 54.02 | 370.3 | 0.63 | 59.70 |
| `gpt-oss-120b-mxfp4` | rocm | 882.6 | 50.91 | 48.88 | 218.9 | 0.59 | 59.31 |

**Vulkan leads `tg512` on 10 of 10 cells — decisively on 6 (by 2.1–18.0%), with
4 ties — and ROCm leads `pp2048` on 9 of 10 (by 6.2–56.8%), with 1 tie.**
A cell counts as a tie when the gap is under 2%, the noise floor of three repeats
([Limitations](#limitations)); the direction is one-sided, no cell reverses. The
2026-10-03 round of five additional cells reproduces the split: Vulkan leads
`tg512` on all five (decisively on 3, by 2.1–4.2%) and ROCm leads `pp2048` on all
five (by 20.5–56.8%) — the direction held from a 4B cell to a 39.6 GiB 70B.

Three things worth knowing before quoting these numbers:

- **The gaps are small on dense models and large on MoE.** ROCm's generation
  shortfall is 1.3–4.2% on the seven dense cells, but 7.9% on gpt-oss-120b,
  17.6% on Qwen3.5-35B-A3B and 18.0% on Qwen3-Coder-30B-A3B. Prefill goes the
  other way: ROCm's advantage runs from 6.2% on Qwen3.8-27B (Q5_K_XL) to 56.8%
  on Poro 2 70B, with Qwen3.5-35B-A3B the one tie (0.7%).
- **MoE cells are 5–8× faster at generation and 5–8× more efficient per watt**
  than the dense 27–32B cells: 74.86 tok/s at 0.90 tok/s/W
  (Qwen3-Coder-30B-A3B) and 62.30 at 0.80 (Qwen3.5-35B-A3B), against 9.5–10.6
  tok/s at 0.11–0.12 for the dense cells. A few billion active parameters do
  most of the work.
- **Dense cells run at 73–85% of the bandwidth ceiling** (256 GB/s ÷ model size
  in GiB). The 39.6 GiB Poro 2 70B reaches 5.25 tok/s against a 6.46 tok/s
  ceiling — 81% on Vulkan, 80% on ROCm — and the 4B cell reaches 76% on Vulkan.
  The MoE cells clear the ceiling outright, which is the point of the
  architecture.

ROCm runs hotter throughout: 88–95 °C across the ten cells against 74–92 °C on
Vulkan, with the Qwen3-4B cell the hottest measured at 95 °C. Temperature is
logged per run.

> **The peak GPU column is residency, not the VRAM pool.** On this box the UMA
> carve-out is 1 GiB, so weights live in GTT while `mem_info_vram_used` sits near
> the idle floor: the rollup sums VRAM + GTT into `mem_peak_gib` and the table
> publishes that. Rows from runs that predate GTT sampling show the VRAM pool
> only, which is why the older tables' 59 GiB cell read as 0.87 GiB. See
> [docs/RUNNING.md](docs/RUNNING.md) §7.

> **What the TTFT column measures (read
> [docs/RUNNING.md](docs/RUNNING.md) §7 before quoting it).** Each timed request starts
> from an **erased KV cache**, so the number is full prompt prefill + first token for a
> fresh request — not first-token latency with the previous turn still cached. It is
> therefore *larger* than a warm-cache figure (20–493% above the pre-2026-10-24 warm
> column on Vulkan, 58–397% on ROCm) and it is the same quantity on both backends. It
> also removes the request-to-request spread that made the old column volatile: the
> mean/p50 ratio fell from up to 2.11 to at most 1.04, and the median per-request spread
> from 53 ms to 3 ms. The residual spread is model-specific — the 27B Q6_K cell still
> takes 1.2 s to first token on Vulkan (spread 51 ms), and the 70B 1.18 s — because a
> long prompt on a dense 20–40 GiB model is genuinely slow to prefill.

## Charts

All ten cells, both backends, ordered by model size (small → large). Each row is
one model with its Vulkan and ROCm results joined by a line, so the pair *and the
size of the gap* read directly. On the generation chart the dense-model
bandwidth ceiling is marked ✕ where it applies.

![Generation throughput — tg512 (tok/s)](charts/generation_tps.png)

![Prompt processing — pp2048 (tok/s)](charts/prefill_tps.png)

![Generation efficiency — tokens per watt](charts/tokens_per_watt.png)

The backend split as a single picture — bars are ROCm relative to Vulkan, so
positive is ROCm faster and negative Vulkan faster:

![Backend difference — ROCm vs Vulkan](charts/backend_delta.png)

The figures are generated from `results/consolidated.csv` by
`scripts/make_charts.py` (Vulkan = blue, ROCm = orange). Each PNG carries a
fingerprint of the rollup it was drawn from, and `tests/test_charts.py` fails
when a committed chart no longer matches the data — a stale figure cannot ship
next to fresh numbers.

## Finnish capability — FIN-bench-v2

Throughput says nothing about whether a model writes good Finnish. This is the
official [FIN-bench-v2](https://huggingface.co/datasets/TurkuNLP/FIN-bench)
suite, run on the same pinned llama.cpp build by `tools/score_ll` +
`scripts/finbench.py` (see
[docs/RUNNING.md](docs/RUNNING.md#9-finnish-capability-axis--fin-bench-v2-cf--p0)
for why it needs its own scorer).

Five models × the five task families that have a `cf` variant × prompt variant
`p0`, zero-shot — 7,296 items per model, all scored: no skips, no context
overflows, no unusable gold indices. The driver also supports the rest of the
upstream grid — two formulations (`cf`, `mcf`) × five prompt variants
(`p0`–`p4`), one pair per run (`--formulation` / `--variant`) — but the table
below is `cf`/`p0`, the only pair swept across all five models; `mcf`/`p0` has a
one-model probe, reported separately below as a negative result. Cells are
**acc / acc_norm**;
`acc_norm` divides the choice loglikelihood by its character length, exactly as
the harness does. Full detail: [results/finbench-table.md](results/finbench-table.md).

| Model | arc_c | belebele_fin | goldenswag | scandisent | sib200 | mean acc |
|---|---|---|---|---|---|---|
| `poro2-70b-q4_k_m` | 0.455 / 0.486 | 0.570 / 0.470 | 0.649 / 0.714 | 0.954 / 0.954 | 0.518 / 0.435 | **0.629** |
| `eurollm-9b-q8_0` | 0.421 / 0.445 | 0.537 / 0.471 | 0.521 / 0.587 | 0.944 / 0.944 | 0.596 / 0.567 | 0.604 |
| `poro2-8b-q8_0` | 0.396 / 0.420 | 0.469 / 0.417 | 0.587 / 0.638 | 0.942 / 0.942 | 0.356 / 0.263 | 0.550 |
| `poro2-8b-q4_k_m` | 0.389 / 0.420 | 0.452 / 0.407 | 0.579 / 0.630 | 0.922 / 0.922 | 0.306 / 0.217 | 0.530 |
| `qwen3-8b-q4_k_m` | 0.310 / 0.364 | 0.434 / 0.410 | 0.389 / 0.344 | 0.499 / 0.499 | 0.515 / 0.611 | 0.430 |

Three findings:

- **Finnish adaptation buys a lot at equal size.** Quant-matched at Q4_K_M,
  Poro 2 8B beats the English-centric Qwen3-8B by 10 points of mean accuracy
  (0.530 vs 0.430), winning four of five tasks: +7.9 points on arc_c, +19.0 on
  goldenswag and +42.3 on scandisent. The Q8_0 Poro 2 8B scores 0.550, so an
  unmatched Q8-vs-Q4 comparison overstates the gap by about 2 points.
- **Sentiment is a cliff, not a gradient.** Qwen3-8B scores 0.499 on a binary
  task — exactly chance — where every Finnish-capable model is at ~0.94. It
  cannot produce "positiivinen"/"negatiivinen" as continuations at all.
- **The Finnish-specific model loses topic classification.** sib200 is the one
  task where Poro 2 8B trails Qwen3-8B (0.356 Q8_0 / 0.306 Q4_K_M vs 0.515); the
  70B only ties it at 0.518, while the generalist EuroLLM-9B leads at 0.596.
  Finnish adaptation is not uniformly better: EuroLLM-9B beats the
  Finnish-focused Poro 2 8B on mean accuracy (0.604 vs 0.550 / 0.530), though
  the larger Poro 2 70B still leads overall at 0.629.

Caveats before quoting these: the Poro 2 8B vs Qwen3-8B comparison is
**quant-matched** (both Q4_K_M) — the twin pins the quant effect at ~2 points
(0.550 → 0.530), not the whole gap — while the EuroLLM-9B and Poro 2 8B Q8_0
cells still differ in quant from their neighbours; this is the `cf` formulation
at one prompt variant, so it is not comparable with published FIN-bench tables
(the paper notes formulation effects, singling out Poro 2 70B as strong in CF
specifically); `goldenswag` is scored on its `train` split because its test
labels are withheld upstream; and GPU-vs-CPU logprob numerics differ by up to
~0.12, harmless for a task score but able to flip a near-tie.

### Formulation sensitivity — one `mcf` / `p0` probe

The driver can also run the `mcf` formulation, which lists the options inside the
prompt. One full-split probe (`fi-mcf-p0-20261021`, Poro 2 8B Q4_K_M) shows it is
**not a like-for-like scale** and is reported as a negative result:

| Task | cf/p0 acc | mcf/p0 acc | Δ |
|---|---|---|---|
| arc_c | 0.389 | 0.466 | +0.077 |
| belebele_fin | 0.452 | 0.239 | −0.213 |
| goldenswag | 0.579 | 0.335 | −0.244 |
| scandisent | 0.922 | 0.811 | −0.111 |
| sib200 | 0.306 | 0.372 | +0.067 |
| **mean acc** | **0.530** | **0.445** | **−0.085** |

Why the mean is misleading: `belebele_fin` collapses to chance (0.239 against
0.25 for four options) because it answered `A` on 887 of the 900 items (the
other 13: 9 `B`, 4 `D`) — the same near-constant answer under both bare and
space-prefixed continuations in a 200-item scratch control (whose output is not
tracked here), so under `mcf` that task measures letter bias, not
comprehension. `goldenswag`'s `acc_norm` falls to 0.199 — below chance — because
upstream's `mcf` prompt asks for a letter while `doc_to_choice` remains the full
endings. `arc_c` and `sib200` improve (the options in context help), `scandisent`
degrades. So the `cf`/`p0` table above stays the published Finnish comparison;
the `mcf` numbers are kept in [results/finbench-table.md](results/finbench-table.md)
as a documented limitation rather than a capability score.

One measurement caveat that applies to every column: choices are scored as bare
continuations. Re-scoring 200 `scandisent` items with a space-prefixed choice
(the way `lm_eval` shifts trailing whitespace) changed 7 predictions
(0.940 → 0.965), so the boundary tokenisation is worth ~1–2 points on a binary
task and does not explain the `mcf` collapse.

## Quick start

The harness needs a Python 3 that can `import yaml` and, for the charts and the
full test suite, matplotlib. No GPU is needed to run the tests or rebuild the
rollups.

```bash
git clone https://github.com/matnum/local-llm-benchmarks.git
cd local-llm-benchmarks
python3 -m pip install -r requirements.txt

python3 -m unittest discover -s tests -t . -v    # full suite, no GPU
```

A sweep needs the models on disk, a pinned llama.cpp build and an idle GPU:

```bash
python3 scripts/run_sweep.py --backend vulkan    # 10-cell sweep
python3 scripts/run_sweep.py --backend rocm
python3 scripts/run_sweep.py --consolidate       # rebuild rollups, no GPU
python3 scripts/make_charts.py                   # redraw charts from the rollup
```

Prerequisites, per-cell behaviour, GPU-idle checks and the FIN-bench axis are in
[docs/RUNNING.md](docs/RUNNING.md). On the reference machine use
`/usr/bin/python3`: a `python3` from a venv without PyYAML is refused with an
install hint rather than silently running a different matrix.

## Reproducibility

What is pinned, and where:

- **llama.cpp** at `5b59b83f4` for every run. Both backends are builds of that
  same commit.
- **Run parameters** live in `matrix.yaml` (`defaults:`), not in code, and are
  wired into the driver's argparse defaults — a test fails if the two drift.
- **Each run** records the driver commit, llama.cpp commit, CPU governor/EPP,
  both context sizes and the real repeat counts in `run.json`.
- **Raw run directories** (`results/run-*/`) are gitignored: large and
  machine-specific. The rollups (`results/consolidated.csv`, `results/table.md`,
  `results/finbench-consolidated.csv`, `results/finbench-table.md`) are tracked.
  The suite cannot rebuild them from the run data (it does not have it on CI) —
  it diffs the README against the tracked CSV, so a value that drifts from the
  rollup fails. Rebuilding needs the run directories, i.e. the machine that
  measured them (`--consolidate`), and so does the drift guard that checks the
  tracked rollup still matches them (`--check`, exit 1 and the offending rows when
  it does not).
- **Published numbers are tested.** The percentages in this README are
  recomputed from the rollup by `tests/test_readme_ranges.py`, the results
  and FIN-bench tables are diffed against the rollups by
  `tests/test_readme_tables.py`, and `results/consolidated.csv` must carry one
  KV type and one TTFT protocol per cell across both backends
  (`tests/test_rollup_protocol_consistency.py`). The TTFT column's per-request data
  for **both** protocols is tracked in `results/ttft-per-request.csv` and
  re-derived from it (`tests/test_ttft_extract.py`), which also checks the
  per-request table in `docs/RUNNING.md` §3c against the extract. The bench, power,
  temperature and memory columns are covered too: `results/bench-logs/` carries the
  verbatim `llama-bench` output the three bench means come from and is re-parsed by
  the driver's own parser, and `results/sampler-support.csv` publishes the sample
  count and duration behind every window aggregate
  (`tests/test_bench_extract.py`). So the latency argument and the throughput
  argument are both checkable without the gitignored run directories. CI runs them
  on every pull request.

## Repository layout

| Path | What |
|---|---|
| `scripts/run_sweep.py` | the sweep driver (stdlib + PyYAML) |
| `scripts/download_models.sh` | one-time fetch of the GGUF files in `matrix.yaml` (nothing downloads during a run) |
| `scripts/sampler.py`, `scripts/ttft_probe.py` | stdlib helpers: power/VRAM trace, TTFT probe |
| `scripts/finbench.py` | FIN-bench-v2 driver (stdlib + PyYAML + Jinja2) |
| `scripts/fetch_finbench_upstream.py` | refreshes the vendored upstream prompt templates |
| `docs/finbench/` | upstream template snapshot (commit-pinned, sha256 per file) |
| `scripts/make_charts.py` | regenerates the charts from the rollup (needs matplotlib) |
| `scripts/make_social_preview.py` | regenerates the 1280×640 repository social card |
| `tools/score_ll.cpp` | loglikelihood scorer for the FIN-bench axis, linked against the pinned llama.cpp |
| `matrix.yaml` | the benchmark matrix and the run-parameter defaults |
| `docs/RUNNING.md` | operator guide: prerequisites, per-cell behaviour, caveats |
| `docs/env.md` | machine fingerprint at measurement time |
| `docs/reviews/` | independent reviews, with resolution notes |
| `results/` | tracked rollups and the TTFT per-request extract; raw run directories are gitignored |
| `charts/` | generated figures, fingerprinted against the rollup |
| `tests/` | unittest suite: parser/rollup, FIN-bench transforms, README tables and ranges, charts, scorer golden |
| `PLAN.md` | methodology decisions and the reasoning behind them |
| `CONTRIBUTING.md` | how to add a cell, and what a results PR must include |

## Limitations

- **One machine, one stack.** Every number comes from a single Ryzen AI Max+ 395
  board on CachyOS with Mesa 26.2.4 and ROCm 7.2.4. Another board, driver or
  kernel can move them, especially the Vulkan/ROCm split.
- **GPU power only.** Power is read from the amdgpu hwmon, not the whole SoC, so
  tokens-per-watt understates the energy cost of prompt processing.
- **Two contexts.** `llama-bench` runs at the model-default context (4096) and
  `llama-server` at 8192. Do not compare a bench number to a server number as if
  they were measured at the same context.
- **TTFT measures a cold request, and that is a different number from a warm one.**
  Every timed request starts from an erased KV cache (both backends), so the column is
  full prompt prefill + first token: 20–493% above the pre-2026-10-24 warm column on
  Vulkan and 58–397% on ROCm. That is why it is stable (mean/p50 ≤ 1.04, median spread
  3 ms) where the warm column was not (mean/p50 up to 2.11, spread up to 823 ms): the
  KV bookkeeping that used to land inside TTFT after a 512-token generation is now paid
  by an untimed erase. It does not make the cells equal — the 27B Q6_K cell is 1.21 s on
  Vulkan against 0.36 s on ROCm, both cold, because prefill speed differs — and the
  27B/70B cells stay above a second even cold. Any TTFT figure quoted from this repo
  must name the protocol; the rollup labels every streaming row
  `[<kv> KV, <warm|cold> TTFT]`.
- **Numbers are stack-stamped, and the tables were re-measured.** Every column in
  the table above comes from kernel 7.2.10 / Mesa 26.2.4 / ROCm 7.2.4: the
  throughput, efficiency and memory columns from the 2026-10-22 sweeps, the
  `server tg`/`TTFT` columns from the 2026-10-24 cold re-measure on the same stack.
  The earlier 2026-10-02/03 rows (kernel 7.2.8 / Mesa
  26.2.3) stay in the rollup under their own run ids; measured against them over
  all 20 cell/backend pairs, `pp2048` is 2.6–31.3% lower on the current stack and
  `tg512` stays inside −2.8…+0.9%. Never mix rows from two stacks in one
  comparison.
- **No error bars.** Three repeats catch a broken run; they do not support a 1%
  claim. The band is measured, not assumed. Throughput: a ten-cell warm re-probe moved
  `server tg` by −0.2…+1.4% (mean |Δ| 0.18%); the cold pass reproduced the same
  `server tg` within ±0.3% on Vulkan and ±0.4% mean on ROCm (one cell, gpt-oss on ROCm,
  at −2.7%). Latency: two Vulkan cold passes agree on `TTFT p50` within −1.2…+2.1%, and
  ROCm has a single cold pass. Differences under ~2% are within noise for this setup.
  Contention shows up in `server tg` first (a contended pass read the 4B cell 17% low
  while its TTFT p50 moved 0.5%) — the driver cannot tell you another process grabbed
  the GPU.
- **FIN-bench is one comparable formulation.** `cf` at prompt variant `p0`
  only. `mcf`/`p0` exists for a single cell as a documented negative result —
  it is not a like-for-like scale — and `p1`–`p4` are unmeasured; formulation
  is known to move scores.
- **Quantizations differ across the matrix.** The Finnish comparison is
  quant-matched; the throughput matrix is not — each cell uses the quant that
  fits the footprint being tested.

## License

Code and scripts are MIT-licensed — see [LICENSE](LICENSE). The published
results (`results/`, `charts/`) and this README are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): reuse them, including
commercially, with attribution to this repository.

## Citation

If you use these numbers or the harness, cite the repository. Machine-readable
metadata is in [CITATION.cff](CITATION.cff):

```bibtex
@software{nummenpaa_llm_benchmarks_2026,
  author  = {Nummenpää, Matias},
  title   = {Local LLM Benchmark Suite: llama.cpp inference benchmarks on an AMD Ryzen AI Max+ 395},
  year    = {2026},
  url     = {https://github.com/matnum/local-llm-benchmarks},
  license = {MIT}
}
```

## Acknowledgements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the runtime under test,
  pinned at `5b59b83f4`.
- [TurkuNLP](https://turkunlp.org/) for
  [FIN-bench-v2](https://huggingface.co/datasets/TurkuNLP/FIN-bench) and the
  dataset families behind it.
- The model authors and GGUF quantizers whose work is measured here: OpenAI
  (gpt-oss), Qwen, Poro (Silo AI), EuroLLM, and the maintainers of the
  [Unsloth](https://huggingface.co/unsloth) dynamic quantizations.

## Roadmap

- **Finnish capability, wider** — the driver now runs the full upstream grid
  (`--formulation mcf`, `--variant p1`–`p4`, one pair per run, prompts rendered
  from the vendored upstream templates). What remains is the GPU time to sweep
  it and publish the formulation / prompt-variant comparison.
- **An English/general quality axis** — strict-JSON extraction, function
  calling, and a GSM8K-style math subset over the same cells.
- **Embeddings** — bge-m3 vs nomic-embed-text on a Finnish retrieval set.
- **A warm-cache TTFT column is not published** — the cold-request protocol replaced
  it on 2026-10-24 (both backends, `docs/RUNNING.md` §3c). What a repeat would add is a
  second cold pass on ROCm: the latency noise band is measured on Vulkan only
  (−1.2…+2.1% across two passes), so the ROCm medians are single-pass.
- **Wider matrix.** One-time downloads, not yet swept: gpt-oss-20b (MXFP4),
  Qwen3-30B-A3B (Q4_K_M), Llama 3.1 8B (Q8_0), Mistral Small 3.2 24B (Q4_K_M),
  Gemma 3 27B (Q4_K_M), Llama 3.3 70B (Q4_K_M).

Methodology decisions, including the ones deliberately reversed, are recorded in
[PLAN.md](PLAN.md). Independent reviews of this repository live in
[docs/reviews/](docs/reviews/).
