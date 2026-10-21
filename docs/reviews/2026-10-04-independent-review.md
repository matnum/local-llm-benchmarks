# Independent Code Review — local-llm-benchmarks

**Date:** 2026-10-04 · **Reviewer:** independent (no shared context with implementer)
**Scope:** full repo at HEAD `b334f30` (`main`), plus the uncommitted working-tree edit to
`scripts/finbench.py`. Focus is the FIN-bench-v2 Finnish-capability axis added since the prior
review (`scripts/finbench.py`, `tools/score_ll.cpp`, `results/finbench-*`, `tests/test_finbench.py`),
with the throughput harness (`run_sweep.py` / `sampler.py` / `ttft_probe.py`) re-verified in place.

**Verdict: APPROVED with two README factual errors and one incomplete uncommitted change to fix
(no blocking issues).** The FIN-bench axis is well-built and internally consistent: the tracked
rollups reproduce byte-identically from the per-model `summary.json` tree, and the Finnish table in
the README matches the data value-for-value. The harness is unchanged from the prior approval and
still green (28 tests).

> **Supersedes the prior REVIEW.md.** The prior review (dated 2026-10-03, "APPROVED for the harness +
> results deliverable") referenced commits (`e90dfd1`, `e6c1e32`, `3e6b108`, …) that no longer exist in
> the rewritten history, and it predates the entire FIN-bench axis. Its harness findings (B1–B3, I1–I4)
> are re-confirmed here as resolved. The new findings below concern the FIN-bench work and two
> README-analysis statements.

> **Resolution status (builder, 2026-10-21, verified at `baf4011`).** Every finding in
> §3–§5 is closed on `main`; re-checked against the code, the rollup and the suite (35
> tests, no skips on the reference machine): I-1 (the README now ranks Poro 2 70B first
> and credits EuroLLM-9B only against the 8B, 0.604 vs 0.550/0.530), I-2
> (`1.7–15.5%`, Vulkan denominator — the 18.3% divided by ROCm — now pinned by
> `tests/test_readme_ranges.py`), I-3 (the Q4_K_M twin is in `scripts/download_models.sh`,
> on disk, swept in `fi-20261003`, and the README caveat is quant-matched), M-1
> (per-task `skipped`, `scripts/finbench.py:328-337`), M-2 (basename only,
> `scripts/finbench.py:345`), M-3 (stamp `4853977`, and no driver change since), M-4
> (`goldenswag`'s `train` split documented in `docs/RUNNING.md` §9), and missing
> evidence 1 (`tests/test_score_ll.py`: golden loglikelihoods plus choice-order and
> KV-rollback invariants; it skips when the scorer binary or fixture GGUF is absent).
> Missing evidence 2 is partly closed: all six hand-typed ranges — first- and
> next-round blocks alike — are asserted against `consolidated.csv` by
> `tests/test_readme_ranges.py`, and `tests/test_charts.py` now fails when a committed
> chart carries a fingerprint from an older rollup. M-5 is unchanged by design
> (≤0.0001, cosmetic). The verdict and method above remain the reviewer's snapshot of
> 2026-10-04 at `b334f30`.

---

## 1. Overall assessment

The FIN-bench-v2 Finnish-capability axis is the substantial new work, and it is sound:

- `scripts/finbench.py` (455 LOC, stdlib + PyYAML) drives the official `TurkuNLP/finbenchv2-*` suites
  at the `cf` formulation / `p0` prompt variant, with each task's prompt and `process_docs` transform
  replicated from the upstream `LumiOpen/lm-evaluation-harness` fork and pinned in `tests/test_finbench.py`.
- `tools/score_ll.cpp` links the pinned llama.cpp's own `libllama` to compute summed choice
  loglikelihoods — the correct choice, documented, and the `acc`/`acc_norm` math in `finbench.evaluate()`
  is a faithful copy of `lm_eval/api/task.py` (`acc_norm` divides by character length, pinned by tests).
- The data is real and cross-checkable: I re-ran `finbench.consolidate()` on a scratch copy of the
  `summary.json` tree and it reproduced the tracked `finbench-consolidated.csv` and `finbench-table.md`
  **byte-identically**. The README's Finnish table matches the rollup (all 20 cells × acc/acc_norm
  spot-checked; the "12 points", "+8.6/+19.8/+44.3", "0.499 = chance" claims all reconcile).
- The upstream split choice that looked anomalous (goldenswag `train`, the only non-`test` task) is
  **correct**: I fetched the upstream `_goldenswag_fi_cf_yaml` and it sets `test_split: train`
  (goldenswag's test labels are withheld upstream).

Two factual errors in the README's own analysis prose and one incomplete uncommitted edit are the only
substantive defects. No security, provenance, or data-integrity problems were found.

---

## 2. Blocking issues

None.

---

## 3. Important issues

### I-1 — README claims "EuroLLM-9B also beats both Finnish-focused models on mean acc" — FALSE
`README.md:179-180` ("…and EuroLLM-9B also beats both Finnish-focused models on mean acc") is
contradicted by the repo's own table two paragraphs above it (`README.md:162-167`): Poro 2 70B has mean
acc **0.629**, EuroLLM-9B **0.604**, Poro 2 8B **0.550**. EuroLLM beats Poro 2 **8B** (0.604 vs 0.550)
but is **beaten by** Poro 2 **70B** (0.604 vs 0.629). The table ranks Poro 2 70B first (bolded).
**Impact:** a headline analysis claim that is wrong on its face and would ship to the public README; the
same table a reader is pointed to proves the opposite.
**Fix:** e.g. "EuroLLM-9B beats the Finnish-focused Poro 2 8B on mean acc (0.604 vs 0.550), though the
larger Poro 2 70B still leads overall at 0.629."

### I-2 — README's "ROCm trails by 1.7–18.3%" is wrong (and disagrees with the same README)
`README.md:87` says Vulkan wins `tg512` "on all five cells (ROCm trails by 1.7–18.3%)". Recomputing the
per-cell `tg512_bench_tps` delta from `results/consolidated.csv`, the actual range across all ten cells
is **1.70% to 15.45%** — which is exactly what `README.md:112` already states correctly ("1.7–15.5%").
The 18.3% is not produced by any metric for the referenced five cells (max server-tg trail is 18.01%,
and that is a different metric than the "tg512" bench the sentence names). The prior review's own
resolution note claimed this figure had been "corrected to 1.7–18.3%" — the correction was itself wrong.
**Impact:** an incorrect, hand-maintained percentage in the headline result section, and an internal
contradiction (18.3% at line 87 vs 15.5% at line 112 for the same quantity).
**Fix:** change `1.7–18.3%` to `1.7–15.5%` (or derive the range from `consolidated.csv`).

### I-3 — Uncommitted, incomplete wiring of a 5th FIN-bench model (`poro2-8b-q4_k_m`)
The working tree adds `"poro2-8b-q4_k_m"` to `MODELS` in `scripts/finbench.py:60` (uncommitted),
pointing at `~/.local/share/models/Llama-Poro-2-8B-Instruct.Q4_K_M.gguf`. That model is (a) **not on
disk** (only `…Q8_0.gguf` and the 70B `…Q4_K_M.gguf` are present), (b) **not in `download_models.sh`**
(which fetches only the Q8_0 for this model), and (c) **not in `matrix.yaml`**, and there are no
results for it. `--list` reports it `MISSING` and `--all` silently `SKIP`s it (`run_model`,
`finbench.py:322`) — a quiet gap, not a loud one.
**Impact:** if committed as-is, the repo advertises a model it cannot run and cannot even download. The
edit is clearly mid-flight (its comment explains it is meant to de-confound the Poro-2-8B-vs-Qwen3-8B
comparison by quant-matching, which is a good idea), but it is not ready.
**Fix:** complete it — add the Q4_K_M download to `download_scripts.sh`/`matrix.yaml`, download the
file, run the cell, and update the README caveat (which currently still says the quantisations are
"not matched") — or revert the hunk before committing.

---

## 4. Minor issues

- **M-1 — `skipped` is mislabeled per task in `summary.json`.** `scripts/finbench.py:328` accumulates
  `skipped` across *all* tasks, then `:337` writes that same total into *every* task's summary
  (`summaries[task]["skipped"] = skipped`). If two tasks each skip rows, every task reports the same
  (wrong) total. Latent today (all shipped results have `skipped: 0`). Fix: store `t_skipped` per task
  and keep a separate model-level total.
- **M-2 — `summary.json` embeds the absolute `/home/halo/…` model path.** `finbench.py:345` writes
  `"model": model_path` (the `os.path.expanduser`-expanded absolute path). This is protected only by
  `.gitignore` (`results/finbench-*/`); the throughput driver deliberately keeps `matrix.yaml` paths
  `~/-relative` with a test enforcing it (`tests/test_defaults.py:55`) precisely to avoid leaking the
  username, and the FIN-bench driver does not follow the same discipline. Fix: store the cell id (already
  present) and either drop `model` or store the `~/-relative` path.
- **M-3 — `results/table.md` provenance stamp is 3 commits stale.** Its header (`results/table.md:3`)
  records `bf93a0eddd19`, but HEAD is `b334f30` — the FIN-bench commits landed without a `--consolidate`.
  Informational only (the throughput data is unchanged), but a reader may infer an older harness. A fresh
  `python3 scripts/run_sweep.py --consolidate` refreshes it.
- **M-4 — goldenswag is scored on the `train` split and it is not documented.** `finbench.py:133` is the
  only task with `split: "train"`. This is **correct** (upstream `test_split: train`; test labels are
  withheld) and I verified it against upstream — but neither README nor `docs/RUNNING.md` says so, and a
  reader comparing against published FIN-bench tables could assume `test`. Add a one-line note.
- **M-5 — FIN-bench "mean acc" ranking is computed from 4-decimal-rounded values.** `finbench.py:366`
  rounds each task `acc` to 4 decimals into the CSV, and `:395` averages those rounded values for the
  ranking/mean column. The error is ≤0.0001 and invisible at 3-decimal display. Cosmetic.

---

## 5. Missing evidence / tests

1. **The C++ scorer has no test.** `tools/score_ll.cpp` performs the axis's core computation (context
   pass, per-choice token decode, log-softmax, KV rollback for choice reuse) and is validated only by a
   hand-recorded ~3e-4 CPU spot-check quoted in `docs/RUNNING.md:263-266`, not by an automated test. The
   Python side is well pinned (`tests/test_finbench.py`, 14 tests on metric + transforms), but the
   tokenization/logprob loop itself is untested. A golden test — feed a tiny fixture (known context +
   choices) through the binary and assert the summed loglikelihoods — would pin the highest-risk code.
2. **README percentage ranges are still hand-maintained.** Lines 87, 112-113 carry the generation/prefill
   delta ranges. Line 87 is already wrong (I-2). The prefill "2.0–16.2%" (line 88, first five cells) and
   "2.0–24.5%" (line 112, all ten) both reconcile with `consolidated.csv`, but they are typed by hand
   next to a machine-generated table and will drift again. Consider emitting them from the rollup (the
   chart script already derives its data that way).

---

## 6. Recommended next step

In order, low-risk, no GPU needed:

1. Fix the two README errors — `README.md:179-180` (EuroLLM "beats both") and `README.md:87`
   (`1.7–18.3%` → `1.7–15.5%`).
2. Resolve the uncommitted `poro2-8b-q4_k_m` change: complete it (download + matrix + run + README
   caveat update) or revert it. Do not commit the `MODELS` entry alone.
3. Fix M-1 (per-task `skipped`) and M-2 (`~/-relative` model path in `summary.json`).
4. Optionally re-run `--consolidate` to refresh the `table.md` provenance stamp (M-3).
5. Add a golden test for `tools/score_ll` next time the scorer is rebuilt (missing-evidence 1).
6. The remaining roadmap items (English eval set, embeddings, Vulkan TTFT root-cause) are unaffected by
   any of the above; no harness change blocks them.

Nothing in this review authorizes any financial, production, customer-facing, or security-sensitive
action. The reviewed artifacts are benchmark data and tooling inside this repository; external
publication of the headline numbers remains the operator's decision and should ship with the two README
corrections above and the existing Vulkan-TTFT caveat attached.

---

*Method: full read of `README.md`, `PLAN.md`, `docs/RUNNING.md`, `docs/env.md`, all five scripts, the
scorer (`tools/score_ll.cpp`), and all five test files. Ran the repo's own suite
(`/usr/bin/python3 -m unittest discover -s tests -v`) → 28 tests pass. Re-ran `finbench.consolidate()`
on a scratch copy of the `summary.json` tree → `finbench-consolidated.csv` and `finbench-table.md`
reproduced byte-identically. Re-ran `run_sweep.consolidate()` on a scratch mirror of `run.json` +
`cell.json` → `consolidated.csv` byte-identical, `table.md` identical except the (expected) harness-commit
stamp. Recomputed the per-cell generation/prefill deltas from `consolidated.csv`. Verified the
goldenswag `train` split against the upstream `_goldenswag_fi_cf_yaml`. Confirmed the uncommitted model's
GGUF is absent from disk and from `download_models.sh`. No changes made to the reviewed repo other than
this file.*
