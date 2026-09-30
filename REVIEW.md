# Code & Plan Review — local-llm-benchmarks

**Date:** 2026-09-30 · **Reviewer:** independent (no shared context with implementer)
**Scope:** full repo at commit `e6c1e32` + untracked `scripts/` (3 harness files, ~460 LOC) + smoke-run artifacts.
**Verdict: NOT APPROVED as-is.** Scaffold/docs are solid; the harness has 3 blocking defects, none of which
were caught because no successful end-to-end run exists yet.

---

## 1. Overall assessment

The planning layer is excellent: decisions D1–D4 are explicit and mutually consistent, the matrix is frozen
and byte-verified against `~/.local/share/models/` (all 5 GGUF sizes in `matrix.yaml` match disk, ~152 GB
total, confirmed), `docs/env.md` is a real fingerprint, and the README corrections (CachyOS, on-disk matrix,
status) are accurate. `.gitignore` correctly keeps raw runs and models out of git.

The harness (`run_sweep.py`, `ttft_probe.py`, `sampler.py`) is the right shape — stdlib-only, idle-gated,
sampler-windowed, fail-loud on contamination via `--force` flagging. But the one smoke run that exists
(`results/run-20260930-1851/`, gitignored) **failed**: its `bench.log` ends in
`error: invalid parameter for argument: -c`, the run was `--force`d onto a busy GPU (87% busy, 22.1 GiB
VRAM at sampler start), and it produced 2 rows of resources.csv and no `cell.json`. So the harness has
never produced a valid result, and its three blocking bugs are all latent-but-provable.

Security posture is clean: no secrets, no `shell=True`, all `subprocess` calls use argv lists, server
binds 127.0.0.1, model paths come from a trusted local matrix (no traversal surface). No action needed.

## 2. Blocking issues

### B1 — `llama-bench -c` is not a valid flag in the pinned build → every bench fails
`run_bench()` (`scripts/run_sweep.py:71-72`) passes `-c <ctx>`. The pinned
`~/.local/opt/llama.cpp @ 5b59b83f4` `llama-bench` has **no context flag at all** (verified: `--help`
lists no `-c/--ctx`; only `-fitc/--fit-ctx` exists). The smoke run proves it: `bench.log:66`
`error: invalid parameter for argument: -c`.
**Impact:** `bench_cell()` raises on every cell; the pp/tg numbers — half the deliverable — are never
produced. Worse, the pinned binary cannot set context size for bench runs at all, so the plan's
"ctx 8192" methodology is unachievable with this build (bench runs at the model default, 4096) while
`run.json` metadata still records `ctx: 8192` — a data-integrity mismatch.
**Fix:** drop `-c` from `run_bench()` and either (a) accept default-ctx bench numbers and say so in
metadata/README (change plan wording), or (b) treat "ctx 8192 for bench" as a reason to rebuild
llama.cpp — which conflicts with D2 and needs an explicit decision. Either way, record the *effective*
ctx in `run.json`.

### B2 — `--rocm-override` is a no-op
`main()` builds `env = dict(os.environ)` and sets `HSA_OVERRIDE_GFX_VERSION=11.5.0`
(`run_sweep.py:165-167`), but that dict is **never passed** to `subprocess.run(...)` / `subprocess.Popen(...)`
(no `env=` keyword anywhere; subprocesses inherit the unmodified `os.environ`).
**Impact:** the Phase-4 mitigation for the #1 known risk (gfx1151 MoE/MXFP4 instability, PLAN §5) is
silently disabled. A ROCm sweep would "retry with override" and get identical failures.
**Fix:** thread `env` through `run_bench()` and `probe_cell()` and pass `env=env`.

### B3 — `consolidate()` appends to `consolidated.csv` instead of rebuilding
`consolidate()` reads **all** run dirs, then opens `results/consolidated.csv` in mode `"a"`
(`run_sweep.py:282-290`). Every call re-emits every run's rows → rows duplicate on each sweep.
**Reproduced:** in a scratch copy with one fixture cell, running `consolidate()` twice yielded
18 identical data rows (lines 2–10 duplicated at 11–19). Even with one run per sweep, the Vulkan pass
(Phase 3) followed by ROCm (Phase 4) would double every Vulkan row in the canonical results file.
**Fix:** open with `"w"` and rewrite the whole file (it is already rebuilt in memory from all run dirs —
that was clearly the intent).

## 3. Important issues

### I1 — Sampler temperature is 10× too large
`sampler.py:34`: `temp_c = rd(...) / 100.0`. amdgpu hwmon `temp1_input` is in **millidegrees**: raw
`/sys/class/hwmon/hwmon2/temp1_input` = 67000 right now → 67.0 °C, but the smoke `resources.csv`
records `680.0`. PLAN §3 makes temperature logging the throttling-detection mechanism ("throttling is a
finding"); at this scale the column is unusable (680 °C "throttling" on every run).
**Fix:** `/1000.0`.

### I2 — `table.md` probe-power column is always blank
`run_sweep.py:313` looks up `m.get("power_w_prob", "–")`; the metric is written as `power_w_probe`
(`wname[:5]` of `"probe_window"`). Reproduced in the rendered scratch table: probe W column = `–` even
with data present. **Fix:** `power_w_probe`.

### I3 — Idle check runs once, not per cell
PLAN §3/§5: "GPU must be idle **before each cell**". `run_sweep.py:190` checks once before the loop;
a 10-cell sweep spans hours (a 59 GB load alone takes minutes), so a background process starting
mid-sweep contaminates later cells with no detection. **Fix:** call `gpu_idle()` at the top of the
cell loop (respecting `--force`).

### I4 — Silent zero-fallbacks in the sampler
`sampler.py:17-22` `rd()` returns `0` on any read error, and the hwmon index is hardcoded
(`hwmon2`). hwmon numbering is not stable across reboots/driver updates. Failure mode: an
all-zeros `resources.csv` that looks like a plausible run — for a benchmark repo, silent zero power
is worse than a crash. **Fix:** resolve hwmon by `name == amdgpu` at start; fail loudly if the
expected files are missing; warn (not default) on per-sample read failures.

### I5 — The only on-disk result is a failed, contaminated run
`results/run-20260930-1851/` was `--force`d with the dev server running (busy 87%, VRAM 22.1 GiB —
the dev llama-server, per `docs/env.md`), bench failed (B1), 2 resource rows, no `cell.json`. It is
gitignored so it won't pollute git, but it *will* be picked up by `consolidate()`'s glob if it ever
gained a `cell.json`, and its `run.json` (`forced: true`) is the only evidence of a harness run.
**Fix:** delete the run dir or rename it `run-20260930-1851-SMOKE-FAILED/` so `consolidate()`'s
`run-*` glob can't treat it as data; re-run a real smoke after B1/B2 fixes.

## 4. Minor issues

- **M1 — `scripts/` is untracked** (`git status`: `?? scripts/`). The harness has never been committed;
  the "Scaffold" commit message oversells. Commit it (after fixes) so the Phase-2 work is reproducible.
- **M2 — Doc drift:** PLAN §3 documents `./scripts/run_sweep.sh --backend …`; implementation is
  `run_sweep.py` with no shell wrapper. Align one or the other.
- **M3 — Hardcoded metric names:** `consolidate()` only emits rows for `pp512/pp2048/tg512`
  (`run_sweep.py:258`); custom `--pp-tokens`/`--tg-tokens` (e.g. 1024) silently drop from the CSV
  while still present in `cell.json`. Same for the `table.md` column set. The note `"streaming, mean of
  N"` is a literal N, not the actual count (available in `ttft.json`).
- **M4 — Dependency undeclared:** `run_sweep.py` needs PyYAML (docstring admits it). System
  `/usr/bin/python3.14` has `python-yaml 6.0.3` (verified), but the `#!/usr/bin/env python3` shebang
  resolves to whatever `python3` the operator has on PATH — a venv python without yaml breaks at line 29
  with a bare ImportError. Add a one-line dependency note in the README/PLAN or a `requirements.txt`.
- **M5 — `bench_cell` opens `bench.log` without a context manager** (`run_sweep.py:106`): leaked fd /
  unflushed tail if `run_bench` raises (which it does today, per B1).
- **M6 — No model-file existence pre-check:** a typo'd/renamed GGUF surfaces only after minutes of
  nothing, deep in a 10-hour sweep. `os.path.isfile(cell["file"])` at loop start is a 1-liner.
- **M7 — `wait_health` timeout is 600 s** for server startup: plausible for 19–27 GB models, but the
  59 GB gpt-oss-120b cell may need longer; make it a function of `size_gib` or raise it.
- **M8 — Reproducibility step unimplemented:** PLAN §3 says every sweep is "git commit + tag
  `run-YYYYMMDD-N`"; the driver never does this (nor does it fix the CPU governor, PLAN §3
  "controlled conditions"). Acceptable for now if tracked, but it is a stated exit-criterion for the
  deliverable "repo others can run".

## 5. Missing evidence / tests

1. **No successful end-to-end cell run exists.** Phase-2 exit criterion ("one clean run producing a
   valid CSV: tok/s, TTFT, peak VRAM, W") is unmet. Every claim in the harness about what it measures
   is currently unexercised.
2. **The llama-bench table parser has never parsed real output** (the only run died on `-c`). Its
   regex-based column-walking is plausible but unverified against this build's md table format.
3. **`ttft_probe.py` SSE parsing is unverified** (server never started in the smoke run). Flag
   compatibility (`-fa/-ctk/-ctv/-np/--alias`) should be checked against this build once the server
   actually boots.
4. **No unit checks on sampler values**: the temp bug (I1) shows that "plausible-looking CSV" ≠
   correct units. Cross-check one sample against `rocm-smi --showpower` (the sampler's own docstring
   says to — do it in the smoke run).
5. **No test suite at all.** Minimum viable: a parser test against a captured golden `llama-bench`
   stdout, and a `consolidate()` test asserting no duplicate rows (would have caught B3).
6. **Vulkan backend never smoke-tested through the harness** (only the ROCm device init appears in
   `bench.log`). Plan Phase 1 required both backends to load + generate.

## 6. Recommended next step

In order, all in one sitting:
1. Fix B1 (drop `-c`, decide and record effective ctx), B2 (thread `env=`), B3 (`"w"` mode), I1 (`/1000`),
   I2 (`power_w_probe`), I3 (per-cell idle check), I4 (resolve hwmon by name, fail loud).
2. Delete/mark the failed smoke run dir (I5).
3. Re-run one 1-cell smoke (e.g. `qwen3.8-27b-q5_k_xl`, ROCm) with the GPU actually idle (dev server
   stopped — coordinate with user), and verify: valid `cell.json`, non-zero sane temp/power (cross-check
   `rocm-smi`), `table.md` probe-W column populated, `consolidated.csv` row count == 1 cell × metrics.
4. Commit `scripts/` + report (M1) and only then start Phase 3 (Vulkan sweep).

Nothing here blocks the *docs/plan* work — those are shippable. The harness must not run a full sweep
until B1–B3 are fixed: a full contaminated sweep costs ~5 h per backend and would poison the canonical
CSV.

---
*Method: full read of all 10 tracked files + 3 scripts; live verification of model files/sizes, llama.cpp
commit, build binaries, hwmon paths, and llama-bench flag surface; reproduction of B3 and I2 in a scratch
copy of the repo (fabricated fixture cell, real `consolidate()` code). No changes made to the reviewed
repo other than this file.*
