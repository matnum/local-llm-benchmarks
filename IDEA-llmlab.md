# llmlab — Interactive CLI course for running LLMs locally

**Status:** M0 built 2026-10-04 at `~/projects/llmlab` (see the "M0 landed" note at the
foot of this file). This document stays as the idea record; the spec of record from M0
onward is `llmlab/docs/DESIGN.md`. Complements (does not replace) the benchmark suite in
this repo — it *consumes* this repo's measured data as teaching content.

## 1. Pitch

A terminal course that teaches everything about running LLMs locally — models, formats,
quantization, memory, context, sampling, backends, serving, benchmarking, power, and
model selection — by making the learner **run real models on their own machine and
reason about the numbers they just produced**, not by reading prose.

One-line differentiator: **every number the course shows was either measured on the
learner's machine by the lesson, or is a stamped reference measurement from a known
machine** (this repo's Strix Halo rig). The tool never presents an unmeasured figure
as fact.

## 2. Why this works from where we are

This repo already contains ~80% of the *content* a course about local LLMs needs, all
measured and reviewed:

| Asset in this repo | Lesson it powers |
|---|---|
| `results/consolidated.csv` — 10 cells × 2 backends: pp/tg tok/s, TTFT, VRAM, power, °C, tok/s/W | inference phases, MoE vs dense, memory, backends, efficiency |
| Headline cell: gpt-oss-120b MXFP4 @ ~50 t/s on 128 GB unified memory | "how big can local go" |
| Quant-pair rows (Qwen3.8-27B Q5_K_XL 10.6 t/s vs Q6_K 10.0 t/s) | quantization tradeoff with *measured* speed delta |
| Two-context caveat (bench 4096 vs server 8192) | context & memory — a real trap the learner steps into |
| Vulkan/ROCm 5–0 split (generation vs prefill) + 89 °C hot cell | backends; "throttling is data" |
| Vulkan TTFT warmup finding (`docs/RUNNING.md` §7) | serving & TTFT — a genuine open anomaly to explain |
| FIN-bench-v2 axis (5 models, 7,296 items/model; Qwen3-8B = chance on sentiment) | "size ≠ quality"; language-specific capability; capstone rubric |
| `scripts/sampler.py` (5 Hz hwmon power/temp/VRAM) | live instrumentation in lessons + `lab` mode |
| `scripts/ttft_probe.py` (streaming OpenAI client), `run_sweep.py` (`gpu_idle`, `wait_idle`, `parse_bench`, `killpg`) | serving lessons; safe experiment runner |
| `matrix.yaml` (model registry: id, tier, size, quant, path) | machine adaptivity: what can this learner's box actually run? |
| `docs/env.md`, `docs/RUNNING.md` | lesson prose; verified-environment discipline (D3) |
| 34-test stdlib-unittest suite, no-GPU-required | the quality bar the new tool copies |

## 3. Product shape

### 3.1 Commands

```
llmlab                 # resume course (same as: llmlab next)
llmlab next            # start/continue the next unfinished lesson
llmlab lesson <id>     # jump to a specific lesson (m00..m11)
llmlab lab             # instrumented REPL: type any command; while it runs,
                       # live VRAM / power / °C panel + output parsing
llmlab status          # progress, machine fingerprint, what is runnable here
llmlab selftest        # preflight: llama.cpp builds, on-disk models, GPU idle,
                       # hwmon access, bench-repo data reachable
```

### 3.2 The lesson loop (fixed mechanic)

1. **Brief** — ≤2 min of concept, one diagram, the claim to be tested.
2. **Predict** (where the claim is quantitative) — the learner states an expectation
   *before* running. Stored, then scored as % error after the run.
3. **Experiment** — the tool runs the real thing (pinned `llama-cli` / `llama-bench` /
   `llama-server`), or — on machines where the live run is too heavy — shows the
   matching measured cell from this repo with a clear `reference data` stamp.
   Long experiments show an ETA and are Ctrl-C safe (process-group kill).
4. **Observe** — guided reading: "find the `tg512` row; what does the 85% of ceiling
   mean?" No free scroll-past of the key row.
5. **Check** — 2–3 questions (predict-verify, multiple choice, or "which number in
   the output supports the claim?"). Immediate feedback; a wrong answer gets the
   explanation, not just the answer key.
6. **Takeaways** — 3 bullets max; progress + score persisted to `~/.llmlab/progress.json`.

### 3.3 Curriculum (12 modules; frozen at M0, stretch items explicit)

| # | Module | Teaches | Live experiment (default model: Qwen3-4B Q8_0) | Evidence from this repo |
|---|---|---|---|---|
| 00 | Your machine | what "local" means; fingerprint; unified memory; iGPU 96 GiB allocation | `selftest` live probe (hwmon `mem_info_vram_total`, CPU, RAM) | `docs/env.md` |
| 01 | First generation | tokens, streaming, wall time vs tok/s | one timed `llama-cli` run, 64 tokens | — |
| 02 | Two phases of inference | prefill vs decode; why pp is fast and tg is slow; bandwidth-bound decode | `llama-bench -p 2048 -n 512`; learner predicts tg from `256 GB/s ÷ file size` | Poro 2 8B: 27.1 t/s measured vs ~32 t/s ceiling (85%) |
| 03 | MoE vs dense | active params; why 120B MoE @ 50 t/s beats 70B dense @ 5 t/s | data-backed compare (optional live: 30B-A3B) | headline cells, 10-of-10 chart data |
| 04 | Memory & KV cache | weights + KV + overhead; `-ngl`; what fits in *your* RAM | predict KV delta for `-c 4096` vs `-c 8192`, then verify VRAM | VRAM column, all 10 cells; 59 GiB gpt-oss cell |
| 05 | Quantization | bits/weight; quality–speed–size tradeoff | same prompt, Q5_K_XL vs Q6_K of Qwen3.8-27B (speed) + side-by-side outputs (perceptual) | quant-pair rows; FIN-bench quant-matched pair (measured ~2-pt effect) |
| 06 | Context | context vs memory; the 4096-vs-8192 trap; long-context costs | same prompt at two ctx sizes, measure VRAM + pp tok/s | README "two contexts" caveat, made first-hand |
| 07 | Sampling | temp / top-p / top-k / seed; determinism | temp 0 twice → byte-identical; temp 1.2 twice → different (hash shown) | — |
| 08 | Backends | what a backend is; Vulkan vs ROCm; why it differs per machine | data-backed 5–0 split (optional live: one cell × both backends, ETA shown) | the unanimous split, 89 °C cell |
| 09 | Serving & TTFT | `llama-server`, OpenAI API, streaming, TTFT; warmup behavior | start server, 1 warmup + 3 timed requests; on Vulkan the *first* timed request is anomalously fast — learner finds it | `RUNNING.md` §7; `ttft_probe.py` logic |
| 10 | Power & thermals | tok/s/W, throttling as data, APU/unified-memory advantage | sampler live while 4B generates; compare learner's tok/s/W to the table | tok/s/W column (4B leads at 0.57) |
| 11 | Capstone: choose a model | integrate all axes: fit, speed, power, *quality*; language-specific capability | guided decision from a stated use case → rubric walk → defended pick | FIN-bench table (Poro 2 70B 0.629; Qwen3-8B chance-level sentiment) |

**Stretch (not in the 12):** M12 embeddings/RAG (bge-m3 vs nomic — already the
benchmark roadmap's deferred axis), M13 function-calling / English eval set
(the benchmark roadmap's open gap).

## 4. Architecture

```
llmlab/
  llmlab/
    __main__.py          # llmlab entry point
    cli.py               # argparse subcommands (stdlib-only at parse time)
    ui.py                # rich renderer; plain-ANSI fallback if rich is absent
    probe.py             # MachineProfile: CPU/RAM/hwmon/iGPU alloc, llama.cpp
                         # builds + commit, on-disk models, bench-repo reachability
    progress.py          # ~/.llmlab/progress.json (lessons done, scores, predictions)
    lessons/
      base.py            # Lesson protocol: brief, predict, run, observe, check, takeaways
      m00_machine.py … m11_capstone.py
    exp/
      runner.py          # subprocess experiments: env, timeout, ETA, process-group
                         # kill, GPU idle gate + settle wait (from run_sweep.py)
      power.py           # 5 Hz hwmon sampling (vendored from scripts/sampler.py)
      server.py          # llama-server lifecycle + streaming client (from
                         # scripts/ttft_probe.py::one_request)
    data/
      bench.py           # reads BENCH_REPO results (consolidated.csv, finbench csv,
                         # run-*/run.json); loud failure if schema drifts
      snapshots/         # bundled few-KB snapshot → works standalone, clearly stamped
  tests/                 # unittest, stdlib, no GPU required (copy of the bar here)
  docs/
```

**Dependency policy:** stdlib + PyYAML + `rich` (the single third-party runtime dep;
plain fallback keeps "works anywhere" honest). No GPU, no models, and no network are
required for `status` / `selftest` / any data-backed lesson.

**Machine adaptivity:** every lesson declares `requires` (min VRAM, model tier,
backend, hwmon). `probe.py`'s MachineProfile decides per lesson: **live mode** (run
it here) vs **reference mode** (show the stamped measurement and explain why this box
can't run it live — turned into a teaching moment: "here is exactly what it would
need"). `matrix.yaml` from this repo is the model registry.

**Reuse map (concrete):** `sampler.py` (99 LOC) → vendored as `exp/power.py`;
`ttft_probe.py` (116 LOC) → `exp/server.py`; `run_sweep.py::gpu_idle/wait_idle/
cpu_power_state/parse_bench/killpg` → `exp/runner.py`; `matrix.yaml` → registry;
`results/*` → data-backed lessons; `docs/` → prose source. Vendored files carry a
header pointing back to this repo + commit.

## 5. Milestones (each shippable)

| MS | Contents | Acceptance |
|---|---|---|
| M0 | skeleton: CLI, progress, selftest, probe, lesson framework; lessons 00–01 | `llmlab` walks lesson 01 end-to-end on this box (4B model), persists progress, renders on a plain terminal (no rich) |
| M1 | runner + lessons 02–04 (inference phases, MoE, memory) | lesson 02 predict-verify scores % error; lesson 04 KV prediction within 15% of measured VRAM delta |
| M2 | lessons 05–07 (quant, context, sampling) | determinism check byte-identical at temp 0; quant-pair perceptual diff shown side by side |
| M3 | lessons 08–10 (backends, serving, power) | serving lesson reproduces the Vulkan TTFT warmup anomaly on this box; power panel live during generation |
| M4 | lesson 11 capstone + `lab` REPL + test suite + docs | unit tests for framework/probe/progress/runner (no GPU); golden probe test on fixture; README |
| M5 | portability hardening (non-AMD, <32 GB boxes) + stretch modules M12/M13 as the benchmark roadmap unblocks | selftest green on a CPU-only fixture profile |

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| GPU contention (lesson races the sweep's idle gate) | reuse `gpu_idle` + `wait_idle` verbatim; hard fail, same as `run_sweep.py` |
| Heavy experiments (120B load takes minutes) | reference mode by default for heavy cells; live is opt-in with ETA; process-group kill on Ctrl-C |
| Nondeterministic outputs in sampling lessons | seeded / temp 0 for comparisons; compare hashes, not eyeballed text |
| Scope creep ("everything about LLMs") | 12 modules frozen at M0; stretch list explicit (RAG, function-calling) |
| Other machines (no amdgpu hwmon, no ROCm, 16 GB laptop) | probe gates hwmon/backend-dependent lessons; those lessons *degrade into teaching moments* ("why power data is unavailable here"), never silent skips |
| Coupling to this repo's layout | `data/bench.py` fails loudly on schema drift; `selftest` reports it; bundled snapshot keeps standalone operation |
| Unmeasured numbers sneaking into prose | rule inherited from D3: every number in a lesson is either produced by the lesson or carries a reference stamp with source + date |

## 7. Open questions — all four answered 2026-10-04 (M0)

Locked as `llmlab/docs/DESIGN.md` decisions L1–L4:

1. **Location:** own repo `~/projects/llmlab` (L1). Consumes this repo via `BENCH_REPO`
   (default `~/projects/local-llm-benchmarks`); a bundled snapshot is still to come.
2. **`rich`:** accepted as an *optional* extra with a plain-ANSI renderer as the default
   (L2). M0 went further than the seed proposed: there is no required third-party
   dependency at all, including PyYAML — `matrix.yaml` is read by a narrow in-repo
   subset reader that fails loudly on anything it does not understand (L5).
3. **Default lab model:** Qwen3-4B Q8_0 on `build-vulkan` (L3) — ~4.0 GiB on disk here,
   and the measured run is ~2 s wall for 64 tokens.
4. **Audience:** both, live-first on this APU (L4).

Two decisions the seed did not cover were also needed: the GPU idle gate is reused from
`run_sweep.py` for measurement lessons, while a lesson that only measures its own model
warns about contention instead of refusing (L7); and every displayed number is either
measured in-session or carries a reference stamp, with no third category (L6).

## 8. What this doc deliberately does *not* do

- No code, no commits, no changes to the benchmark harness.
- Does not touch this repo's own harness/review backlog. The `REVIEW.md` §6 items it
  pointed at were closed upstream on 2026-10-17; see §10.

## 9. M0 landed (2026-10-04)

Built at `~/projects/llmlab` (7 commits, `427840f` head at the time of writing): CLI,
machine probe, atomic progress store, safe experiment runner, the lesson framework, and
lessons **m00** and **m01**. 234 stdlib-unittest tests, green with no GPU, no model files
and no network (the rich renderer's tests run for real in the project venv, where rich is
installed; they skip under an interpreter without it).

Acceptance, run on this box against the pinned build (`5b59b83f4`):

- `llmlab` with no arguments walked m00 then m01 and persisted progress to
  `$LLMLAB_HOME/progress.json`; `llmlab status` then reported 2/2 with both lessons in
  `live` mode, and `llmlab lesson m01` refused to re-run a finished lesson.
- m01 ran the real thing: `Qwen3-4B-Q8_0` on `build-vulkan`, 64 tokens in **1.99 s** wall
  clock, **48.4 t/s** decode and **632.1 t/s** prefill as reported by the pinned binary.
  64/48.4 = 1.32 s of decoding, so 0.67 s of the run was load + prefill — the gap the
  lesson exists to explain. The learner's two predictions (5.00 s, 30.0 t/s) were scored
  151.2% and 38.0% off.
- `llmlab selftest` exits 0 here (one warning: a long-running `llama-server` holds
  ~23 GiB VRAM, so the box is not idle — the lesson says so before it measures) and
  exits 1 naming `llama.cpp builds` when `LLMLAB_LLAMA` points at nothing.
- The plain renderer was verified with rich absent (piped runs) and the rich renderer with
  it installed; both lay out identical content.

Not in M0, and next: lessons m02–m11, `exp/power.py`, `exp/server.py`,
`data/bench.py` (live reference cells read from this repo's `results/`), the bundled
snapshot, the `lab` REPL, and the capstone. M1 (m02–m04) is the natural next step.

## 10. Upstream changes since M0 (2026-10-17)

The benchmark repo moved after M0. Lessons that quote it must use these values — the
rule is that no stale figure is ever presented as current (L6).

- **FIN-bench is 5 models, not 4.** `poro2-8b-q4_k_m` was added to complete the
  quant-matched 8B comparison: 0.530 mean acc against Qwen3-8B Q4_K_M's 0.430 — a
  **10-point** gap, not the 12 that an unmatched Q8_0-vs-Q4_K_M pair suggested. The twin
  measures the quant effect at **~2 points** (Q8_0 0.550 → Q4_K_M 0.530), which is what
  lesson m05 should teach in place of the old "plausibly 1–2 points" caveat.
- **The suite is 34 tests, not 30.** Added `tests/test_score_ll.py` (the scorer's
  KV-rollback invariants plus a CPU golden) and `tests/test_readme_ranges.py` (pins the
  README's hand-typed backend deltas to the rollup).
- **The throughput matrix is 10 cells × 2 backends** — this doc previously said 11,
  which would have become "all 11 cells" in lesson m04.
- **`results/table.md`'s provenance stamp now records the last commit that changed the
  driver code**, not HEAD, so it no longer drifts on unrelated commits.
- **Still open, and therefore still honest teaching material:** the Vulkan TTFT warmup
  anomaly (`docs/RUNNING.md` §7), which lesson m09 turns into an exercise, and the
  deferred embeddings / English-eval axes.
