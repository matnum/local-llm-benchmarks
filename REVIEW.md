# Independent Code Review — local-llm-benchmarks

**Date:** 2026-10-03 · **Reviewer:** independent (no shared context with implementer)
**Scope:** full repo at HEAD `e90dfd1` (`main`), working tree including the uncommitted deletion of the
prior `REVIEW.md` and the untracked `SMOKE_COMMAND.txt`; all 3 harness scripts (~730 LOC) and the tracked
rollups (`results/consolidated.csv`, `results/table.md`).
**Verdict: APPROVED for the harness + results deliverable (no blocking issues).**
The prior review (commit `e6c1e32`, "NOT APPROVED") flagged 3 blocking (B1–B3) and 4 important (I1–I4)
issues. All of them are fixed and now verified against real run artifacts, not just code inspection. The
full sweep has landed and the rollups are internally consistent and idempotent.

> **Post-review resolution (2026-10-03):** the two Important issues below (I-1 dead `matrix.yaml`
> `defaults`, I-2 no tests) were subsequently fixed at the operator's request. I-1: `run_sweep.py` now
> reads `matrix.yaml` `defaults` via `build_parser()` and threads `threads/ngl/warmup/power-hz` through to
> the bench/server/sampler subprocesses. I-2: a stdlib-`unittest` suite (`tests/`, 9 tests) now covers the
> `llama-bench` parser, `consolidate()` rebuild-not-append, and the defaults wiring; `consolidate()` also
> got its file handles closed and the unused `meta` read removed. Verified: `python3 -m unittest discover
> -s tests` → 9 passed, and `--consolidate` still reproduces the tracked rollups byte-identically (121
> rows).
>
> Minor items also resolved (2026-10-03): M-1 (removed `SMOKE_COMMAND.txt`; this file restores the
> dangling `REVIEW.md` references), M-4 (renamed `run-20261002-1710/` → `superseded-aborted-…`), M-7
> (`table.md` now records the rollup's own harness commit), and the README was rewritten for public
> consumption (internal roadmap language removed; the `tg512` figure corrected to 1.7–18.3%).

---

## 1. Overall assessment

The repo has gone from scaffold to a completed benchmark deliverable since the prior review. The harness
(`scripts/run_sweep.py`, `scripts/sampler.py`, `scripts/ttft_probe.py`) is stdlib + PyYAML only, idle-gated
at start **and** per cell with a settle wait, fail-loud on missing sensors/models, and stamps CPU power
state + both contexts into `run.json`. Security posture is clean: no secrets (scanned; the only "token"
hits are `max_tokens`/HF-token *mentions*), no `shell=True`, all subprocess calls use argv lists, the
server binds `127.0.0.1`, and model paths come from a trusted local `matrix.yaml`.

The data is real and cross-checkable. I re-ran the driver's own `--consolidate` on a scratch copy of
`scripts/ + results/ + matrix.yaml` and it reproduced the tracked `consolidated.csv` and `table.md`
**byte-identically** — 121 data rows = 11 cells × 11 metrics (no duplicates), confirming B3 is fixed and
the tracked rollups are current. All 11 valid cells report `complete_decodes: true`, and every
`ttft.json` decode ran the full 512-token budget (finish_reason "length"). The README headline table
matches `consolidated.csv` value-for-value (spot-checked pp2048/tg512/TTFT-p50/tok-per-W/VRAM across both
backends), and the "Vulkan takes generation, ROCm takes prefill — 5–0 each way" claim is accurate against
the data.

Prior findings, verified fixed: B1 (`llama-bench -c` dropped, `ctx_bench`/`ctx_server` both recorded in
`run.json`), B2 (`--rocm-override` env now threaded into `run_bench`/`probe_cell`/server Popen), B3
(`consolidate()` opens `"w"`, rebuilds from all run dirs), I1 (`temp1_input` /1000.0 → sane 72–88 °C),
I2 (`power_w_probe` key now matches), I3 (per-cell idle check + 90 s settle), I4 (hwmon resolved by name,
loud failure on missing sensors, per-sample read failures skip + warn instead of silent zeros). M3/M5/M6/M7
from the prior review are also resolved (actual request counts in notes, `bench.log` closed in a
`finally`, model pre-check at loop start, server timeout scaled by model size).

---

## 2. Blocking issues

None.

---

## 3. Important issues

### I-1 — `matrix.yaml` `defaults` block is dead configuration (never read by the driver) — RESOLVED 2026-10-03
`matrix.yaml:5-14` declares `defaults: {backends, gpu_layers, threads_cpu, ctx_size, pp_tokens, tg_tokens,
repeats, warmup_requests, power_sample_hz}`. `run_sweep.py` loads the YAML but reads **only**
`matrix["cells"]` (`run_sweep.py:291-295`); a repo-wide grep for `defaults` returns zero hits in the driver.
Every value in that block is instead hardcoded elsewhere: threads `16` (`run_bench`, `run_sweep.py:171`),
server ctx `8192` (arg default `run_sweep.py:269`), pp `512,2048` (`:266`), tg `512` (`:267`), repeats `3`
(`:265`), warmup `1` (`probe_cell`, `:237`), sampler Hz `5` (`main`, `:369-370`).
**Impact:** the "frozen matrix" reproducibility surface is decorative — an operator who edits
`threads_cpu` or `ctx_size` in `matrix.yaml` expecting the sweep to honor it gets silent divergence (the
run still succeeds, but not under the declared conditions). This is a documented-knob-that-is-not-wired
gap, the kind that has bitten this exact repo before (the `-c`/ctx mismatch in B1).
**Fix:** either wire `matrix["defaults"]` into the argparse defaults (single source of truth), or delete
the `defaults` block and state in `matrix.yaml`/`RUNNING.md` that all run parameters are CLI flags.

### I-2 — No automated test suite — RESOLVED 2026-10-03
Still true from the prior review ("Missing evidence", item 5). The harness's correctness today rests on
one real sweep plus this review's re-run of `--consolidate`. The three worst historical defects — B3
(append-mode row duplication), I2 (`power_w_prob` vs `power_w_probe`), and the `gen_tps` absolute-clock
divisor bug — would each have been caught by a trivial golden-file test. The README's stated deliverable
is "repo others can run"; a repo whose parser and rollup have no tests is a regression magnet for anyone
who touches `run_sweep.py` without the GPU in front of them.
**Fix (minimum viable):** (a) a `tests/` fixture that feeds a captured `llama-bench` table through
`run_bench()`'s parser and asserts `pp512/pp2048/tg512` means+stddev; (b) a `consolidate()` test on a
two-run fixture asserting exact row count and no duplicates; (c) run them via `--consolidate`-style
stdlib-only invocation so no pytest dependency is introduced.

---

## 4. Minor issues

- **M-1 — Working-tree hygiene / dangling references — RESOLVED 2026-10-03.** `REVIEW.md` is deleted (uncommitted) but still
  referenced from `README.md:102,116` and `docs/RUNNING.md:8`. `SMOKE_COMMAND.txt` is untracked and stale
  (references "REVIEW.md step 3", duplicates `RUNNING.md` §3). This report restores `REVIEW.md`;
  `SMOKE_COMMAND.txt` should be deleted or folded into `RUNNING.md`.
- **M-2 — `consolidate()` reads `run.json` into `meta` and never uses it** (`run_sweep.py:425`). Dead
  read; worse, a malformed `run.json` would raise and abort the entire rollup. Drop the variable or use it
  (e.g. to stamp `forced`/`ctx` into the notes).
- **M-3 — `env_binpath(args)` is a misnomer** (`run_sweep.py:416-417`): it returns a binaries path, not an
  environment. Cosmetic; rename to `binpath(args)`.
- **M-4 — Leftover aborted run dir `results/run-20261002-1710/` — RESOLVED 2026-10-03.** Its `sweep.log:9-29` shows the pre-fix
  `gen_tps` probe abort (exit 1), leaving `bench.log` + a 949-row `resources.csv` + `server.log` with no
  `cell.json`. Gitignored and excluded from the rollup (no `cell.json`), so harmless — but it does not
  follow the repo's own `-SMOKE-FAILED` / `superseded-*` naming convention for known-bad runs.
- **M-5 — Power windows omit the server-load phase.** `window_stats` slices `resources.csv` into
  bench/probe/cell windows (`run_sweep.py:247-257`); the server weight-load between `bench_end` and
  `probe_start` falls in no power window (only `temp_c_max`/`vram_peak_gib` capture it via `cell_window`).
  Documented as "generation-phase efficiency" so not wrong, but the `W` columns are easy to misread as
  whole-cell draw. Consider adding a `load_window` or a note on the table.
- **M-6 — `llama-bench` parser is column-positional** (`run_sweep.py:177-192`): it assumes the value is the
  cell immediately after the `pp/tgN` test-name cell. Correct against the pinned build `5b59b83f4` (proven
  by 11 cells) but fragile if a future llama.cpp adds a column between the test name and t/s. A golden-file
  test (I-2) would make this breakage visible instead of a silent empty `tests` dict.
- **M-7 — Run metadata records the capture-time harness, not the consolidate-time harness — RESOLVED 2026-10-03.**
  `run-20261002-1751/rocm/run.json` stamps `harness_commit: 3e6b108`, but the rollup's `ttft_ms_p50` column
  was added by a later commit (`07f2162`) and the CSV was re-derived with the newer driver. Harmless
  because `cell.json` carries every raw value, but "repo others can run" would be tighter if the rollup
  also recorded the driver commit that produced it.

---

## 5. Missing evidence / tests

1. **No test suite** — see I-2. Nothing guards the parser, the window slicing, or `consolidate()`
   idempotency.
2. **The sampler's own power cross-check is unrecorded.** `sampler.py:16-18` mandates "cross-check one
   sample against `rocm-smi --showpower --showtemp` per cell"; I found no captured `rocm-smi` output or
   note in `results/`. The temperature units are proven sane (72–88 °C, I1 fixed), but the **power** units
   (µW→W) remain verified only by plausibility, not by the documented cross-check.
3. **Vulkan TTFT-spread root cause is still open** (`RUNNING.md` §7) — correctly documented and carried as
   `ttft_ms_p50` alongside the mean, so not a harness gap; just don't quote a Vulkan TTFT mean without the
   p50 caveat.

---

## 6. Recommended next step

In order, low-risk, no GPU needed:

1. Resolve `matrix.yaml` `defaults` (I-1): wire it into argparse defaults or delete it — pick one, don't
   leave a dead block that implies knobs that do nothing.
2. Add the two cheap tests from I-2 (parser golden-file + consolidate idempotency) as stdlib-only scripts
   so the "others can run" claim is actually protected.
3. Clean the working tree: remove `SMOKE_COMMAND.txt`, rename/delete `results/run-20261002-1710/`, and
   commit this `REVIEW.md` (which resolves the deleted-file + dangling-reference state in M-1).
4. Perform and record the one `rocm-smi` power cross-check (missing-evidence item 2) next time a cell runs.
5. No harness changes block the remaining deliverables (Finnish evals, embeddings, posts). The next sweep
   can proceed with the current driver.

Nothing in this review authorizes any financial, production, customer-facing, or security-sensitive action.
The reviewed artifacts are benchmark data and tooling inside this repository; external publication of the
headline numbers remains the operator's decision and should carry the Vulkan TTFT caveat attached.

---

*Method: full read of all 11 tracked files + 3 scripts; `git log/status/diff`; secrets + `shell=True` +
tracked-pyc scans; live re-run of the driver's own `--consolidate` on a scratch copy (byte-identical to the
tracked rollups, 121 rows, no duplicates); per-`ttft.json` `complete_decodes` check across all 11 valid
cells; value-for-value spot-check of the README headline table against `consolidated.csv`. No changes made
to the reviewed repo other than this file.*
