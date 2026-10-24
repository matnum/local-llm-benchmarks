# Independent Code Review — local-llm-benchmarks

**Date:** 2026-10-24 (EEST) · **Reviewer:** independent (reviewer profile, no shared context with the implementer)
**Scope:** repo at `HEAD 4a1cadd` (`fix/finbench-negative-result-claims`) **plus the uncommitted
worktree** — the cold-TTFT-protocol change (`scripts/run_sweep.py`, `scripts/ttft_probe.py`,
`matrix.yaml`, `docs/RUNNING.md`, `tests/test_probe_only_and_tagging.py`, new
`tests/test_ttft_protocol.py`). Also re-verified the branch's headline claim: the FIN-bench
`mcf` negative result.

**Verdict: BLOCKED for publication.** The cold-TTFT protocol itself is implemented soundly and
its mechanism works against a live server (verified from the run artifacts). But at review time
the repository is internally inconsistent in three ways that a reader of this repo is invited to
check: the tracked rollup now carries **two different TTFT quantities in one column** (cold on
Vulkan, warm on ROCm), the repo's own README-table guard is **red (10 failures)**, and
`docs/RUNNING.md` §3c publishes numbers and properties that **the run it cites contradicts**.
None of that is a code bug; all of it is a publish-the-claim-before-the-data defect in a repo
whose stated value is "the results can be checked rather than trusted".

What the review did **not** find: any security, credential, network, or dependency-provenance
problem (the harness talks only to `127.0.0.1`; the FIN-bench driver reads public HF REST
endpoints with timeouts and no token), and no correctness fault in the probe's timing maths or
in the slot-erase mechanism.

---

## Evidence base (what was actually run)

| Check | Command | Result |
|---|---|---|
| Suite, worktree at review start | `/usr/bin/python3 -m unittest discover -s tests -t .` | **98 tests, OK** |
| Suite, after the sweep finished and auto-consolidated | same | **98 tests, FAILED (failures=10)** — all `tests.test_readme_tables…test_throughput_table_matches_rollup`, one per Vulkan cell |
| Rollup rebuilt from the on-disk run dirs (scratch copy, before the 2026-10-24 sweep finished) | `run_sweep.consolidate(repo=<copy>)` | 546 rows vs **531 tracked**; re-running the *documented* `--consolidate` was enough to break 5 README assertions |
| Protocol mix in the rollup produced by the sweep's own auto-consolidate | script over `results/consolidated.csv`, latest-wins per metric | `('vulkan','cold'): 10`, `('rocm','warm'): 10` for `ttft_ms_p50` |
| Cold per-request data vs the warm column | parse of `results/run-20261024-0843/vulkan/*/cell.json` | table below |
| Slot-erase API gating on the pinned build | `~/.local/opt/llama.cpp` @ `5b59b83f4`, `tools/server/server-context.cpp:4732,4773` | `POST /slots` needs only a non-empty `--slot-save-path`; `GET /slots` needs `--slots` (not used) → the design note in §3c is correct |
| FIN-bench `mcf` negative result (branch headline) | parse of `results/finbench-fi-mcf-p0-20261021/poro2-8b-q4_k_m/*.json` + `results/finbench-consolidated.csv` | **exactly right**: belebele preds `{A:887, B:9, D:4}`, acc 0.2389, goldenswag acc_norm 0.199, all deltas match the README table |

Timeline note: a Vulkan `--kv f16 --ttft-protocol cold --probe-only` sweep
(`results/run-20261024-0843`, started 08:43) was still running when this review began and
finished at 09:02:41; the driver's end-of-sweep `consolidate()` is what modified
`results/consolidated.csv` and `results/table.md` in the worktree during the review. No ROCm
cold pass exists on disk or in any doc.

Cold (Vulkan, f16, 3 × 512-token) vs the warm column it would replace:

| cell | warm TTFT p50 (HEAD) | cold TTFT p50 (now in rollup) | Δ | cold spread | mean/p50 |
|---|---|---|---|---|---|
| qwen3-4b-q8_0 | 27.6 | 55.0 | +99% | 106.4 | 1.63 |
| poro2-8b-q8_0 | 42.3 | 125.3 | +196% | 1.8 | 1.00 |
| eurollm-9b-q8_0 | 48.3 | 142.6 | +195% | 1.5 | 1.00 |
| qwen3.8-27b-q5_k_xl | 422.0 | 509.4 | +21% | 32.2 | 0.99 |
| qwen3.8-27b-q6_k | 818.5 | 1207.9 | +48% | 816.2 | 1.21 |
| qwen3-32b-q5_k_m | 185.7 | 431.2 | +132% | 26.8 | 0.98 |
| qwen3-coder-30b-a3b-q6_k | 44.0 | 180.9 | +311% | 3.2 | 1.00 |
| qwen3.5-35b-a3b-q6_k | 48.3 | 189.0 | +291% | 2.9 | 1.00 |
| poro2-70b-q4_k_m | 199.0 | 1174.9 | +490% | 41.0 | 0.99 |
| gpt-oss-120b-mxfp4 | 116.5 | 373.2 | +220% | 24.2 | 1.01 |

`tg_server_tps` also moved on two cells in the same ingest: 4B 48.12 → 39.85 (−17.2%),
27B Q6_K 9.92 → 8.91 (−10.2%); the other eight moved ≤0.8%.

---

## 2. Blocking issues

### B1 — The published TTFT column now mixes two protocols, and the repo's own guard is red

**Where:** `results/consolidated.csv`, `results/table.md` (both modified by the sweep's
auto-`consolidate()` at 09:02), `README.md:141-209`.
**What:** `ttft_ms_p50` carries **10 cold rows (Vulkan, `run-20261024-0843`) and 10 warm rows
(ROCm, `run-20261023-1159`)**. The README prints those 20 rows side by side per cell, so the
Vulkan-vs-ROCm pair for e.g. Qwen3-Coder reads 180.9 ms vs 21.4 ms — an apparent 8.4× Vulkan
latency penalty that is entirely the protocol difference. `docs/RUNNING.md:317-322` and
`README.md:201-209` both say TTFT must not be compared across cells or backends; the change
makes even the same-cell pairing cross-protocol.
**Impact:** Any commit/push of the current worktree publishes a column whose two halves are
different quantities, and **CI fails**: 10 of 98 tests are red right now
(`test_throughput_table_matches_rollup`, one subtest per Vulkan cell). The guard is working
exactly as designed — it is the change that is not finished.
**Fix:** do not publish until a ROCm cold pass exists
(`scripts/run_sweep.py --backend rocm --kv f16 --ttft-protocol cold --probe-only`), then
re-consolidate and update README table + prose + tests in one commit. If ROCm cold cannot run
soon, publish **two labelled columns** (warm, cold) or leave the column on warm and present cold
as a separate measurement — never one column with two protocols.

### B2 — §3c's evidence table and its conclusions are contradicted by the run they cite

**Where:** `docs/RUNNING.md:148-162` (§3c table + claims), `results/table.md:3` (new prose:
"cold TTFT … so mean and p50 agree"), `README.md:474-479`, `docs/RUNNING.md:331,335`,
`PLAN.md:17,20`.

1. The 4B row claims `54 / 56 / 55 ms (spread 1.9)`. The published cold run measures
   **55.0 / 54.2 / 160.6 ms (spread 106.4)**.
2. The 32B row claims `426 / 424 / 425 ms (spread 2.1)`. The published cold run measures
   **404.9 / 431.7 / 431.2 ms (spread 26.8)**.
   (The 27B row, `478 / 506 / 513` vs `482.8 / 509.4 / 515.0`, does match.)
3. "`gen_tps` was identical in both arms (… 4B 47.9)" is false for the run that exists: 4B
   `tg_server_tps` is **48.12 warm → 39.85 cold (−17.2%)**, and 27B Q6_K 9.92 → 8.91 (−10.2%).
4. The new rollup prose "cold TTFT … **so mean and p50 agree**" fails on 2 of the 10 published
   cold cells: 4B mean 89.9 vs p50 55.0 (1.63×) and 27B Q6_K mean 1461.3 vs p50 1207.9 (1.21×,
   with a 1996 ms third request). The protocol narrows the spread on 8 of 10 cells; it does not
   eliminate it, and the worst cell (`qwen3.8-27b-q6_k`) is still 1.2–2.0 s to first token with
   an erased cache — which §3c's "no KV bookkeeping in the number" does not explain.
5. Status text is stale in three places, in the same change that declares the protocol shipped:
   `docs/RUNNING.md:331` "the TTFT column needs no protocol change"; `:335` "A cold-request
   protocol is the remaining option and **has not been run**"; `README.md:474-479` "A cold-request
   protocol would isolate pure latency; **it has not been run**"; `PLAN.md:17` (D6) offers cold
   as a future option and `PLAN.md:20` (D9) says "a cold-request protocol **stays open work**".

**Impact:** This repo's contract is that every published figure is checkable against tracked
data. A doc table whose two of three rows cannot be reproduced from any artifact on disk, next to
three documents saying the opposite happened, is the exact defect class this repo has been
reviewed for before. It is also not verifiable by a reader at all: raw run dirs are gitignored.
**Fix:** rewrite §3c's table from `results/run-20261024-0843`'s actual per-request values (and
say whether the numbers are n=1); delete "mean and p50 agree" or narrow it to the cells where it
holds; delete the "`gen_tps` identical" claim or restate it with the two exceptions; reconcile
§7, README roadmap and PLAN D6/D9 in the same commit (§7's mechanism section should gain a line
on why the Q6_K cell still stalls cold).

---

## 3. Important issues

### I1 — The `--kv` default still does not match the published protocol (`run_sweep.py:362-365`)

`--ttft-protocol` now derives its default from `matrix.yaml`, and
`tests/test_ttft_protocol.py:158-164` asserts that it must ("the default must match what is
published, or the column silently drifts from the docs"). `--kv` is still a hardcoded literal
`"q8_0"` and `matrix.yaml` `defaults:` has no `kv` key — so a full sweep with **all defaults**
publishes a **q8_0-KV** TTFT column while every doc says the published protocol is f16, and
`run.json` will state that confidently. Same defect class, unfixed, one line away.
**Fix:** add `kv: f16` to `matrix.yaml` defaults and derive the parser default from it
(`defaults.get("kv", "q8_0")`), with the same kind of pinning test.

### I2 — A crashed full sweep is stamped `partial: false` and its cells enter the rollup (`run_sweep.py:453`)

`partial` is computed only from the **cell selection**
(`not tracks_rollup([c["id"] for c in cells], …)`), not from whether the sweep completed.
Evidence: `results/superseded-aborted-20261024-0835/vulkan/run.json` says `partial: false`
although the run died on its first cell (traceback in `sweep.log`); it is out of the rollup only
because a human renamed the directory to `superseded-*`. `tests/test_consolidate.py`'s
`RollupOwnershipTest` and `docs/RUNNING.md:125-126` both describe crash-safety that does not
exist in the code.
**Impact:** interrupt a 10-cell sweep at cell 7 and its 7 cells silently become the "latest"
values for those columns — the failure mode M-2 was written to prevent.
**Fix:** stamp `partial: true` when the run starts, clear it after the final `consolidate()`;
`consolidate()` already skips `partial` runs, so this closes the hole without new machinery.

### I3 — The protocol label is missing on exactly the rows that are not probe-only (`run_sweep.py:756-761`)

`kvnote` is built only `if probe_only or proto == "cold"`. A future **warm full sweep** (or any
pre-2026-10-24 full sweep) therefore emits `ttft_ms`/`ttft_ms_p50` rows with no KV type and no
protocol — the tracked CSV already carries 41 unlabelled `mean of 3` TTFT rows and their 41
unlabelled p50 counterparts. That makes
`results/table.md:3`'s "protocol stamped per row" false, and leaves the ambiguity the change was
introduced to remove.
**Fix:** always append the label (`[<kv> KV, <proto> TTFT]`), with `probe-only run` added when
applicable; update `tests/test_probe_only_and_tagging.py` accordingly.

### I4 — The warm→cold switch removes the repo's only measured noise band for TTFT

`README.md:419-423` and `PLAN.md:20` justify "differences under ~2% are noise" with the ten-cell
**warm** re-probe (`run-20261023-1140/-1159`, two runs). The cold column has **one pass, n=3
requests, no repeat**, on a protocol whose whole point is to change the quantity being measured —
and `qwen3.8-27b-q6_k` cold lands at 1996 ms on its third request. There is no cold noise floor.
**Fix:** run the cold pass twice (both backends) and publish the run-to-run band, or state
explicitly in §7 that the cold band is unmeasured and that its medians are single-pass.

### I5 — "The suite rebuilds the rollups from the per-run data to catch drift" is not true (`README.md:361-368`, `.github/workflows/tests.yml` comment)

No test reads `results/run-*`; raw run dirs are gitignored, so CI cannot. `tests/test_consolidate.py`
builds its own temp fixtures, and `tests/test_readme_tables.py` only diffs the README against the
**tracked** CSV. The workflow's comment ("reproduces consolidated.csv from the run data") inherits
the same overstatement.
**Impact:** exactly why B1 was reachable — the documented drift guard does not exist, and the
first `--consolidate` after unrelated probe runs silently invalidates the README.
**Fix:** correct both texts; optionally add a `--check` mode (or a test, skipped when
`results/run-*` is absent) that fails when a run dir newer than the rollup's newest `run_id`
exists locally.

---

## 4. Minor issues

- **M1 `run_sweep.py:256-269`** — `server_argv(..., slot_dir=None)` will happily render
  `--slot-save-path None` if a caller forgets the path in cold mode. Only `probe_cell` calls it
  today; make `slot_dir` required when `ttft_protocol == "cold"`, or assert it.
- **M2 `ttft_probe.py:139-145`** — the erase reply is read but never validated: a server that
  answers `{"id_slot":0,"n_erased":0}` (or omits the key) yields a warm-ish measurement that is
  still recorded as `"protocol": "cold"`. Fail the probe when the erase after the warmup drops
  nothing, and keep the count in the failure message.
- **M3 `ttft_probe.py:126`** — the probe's own default is `warm` while `matrix.yaml` and the CLI
  default to `cold`. A manual invocation (the usage line advertises `[PROTOCOL]` as optional)
  silently produces the pre-2026-10-24 quantity. Make the argument required.
- **M4 `docs/RUNNING.md:117`** — §3b still documents the note format as `[<kv> KV, probe-only
  run]`; it is now `[<kv> KV, <warm|cold> TTFT, probe-only run]`.
- **M5** — `README.md:370-391` (repository layout) omits `scripts/download_models.sh`, which
  `CONTRIBUTING.md:21` requires contributors to edit.
- **M6 `run_sweep.py:284`** — `stdout=open(f"{cdir}/server.log", "w")` is never closed; one
  leaked handle per cell (pre-existing, surfaced by the surrounding edits).
- **M7 `docs/RUNNING.md:143-146`** — "The server must be started with `--slot-save-path` for the
  slots API to exist at all" is right for the `POST` action but slightly overbroad: `GET /slots`
  is gated on `--slots` (`server-context.cpp:4732`). Worth one clause so the next person does not
  add an unnecessary flag.

---

## 5. Missing evidence or tests

1. **No test exercises the cold path against a real server.** `tests/test_ttft_protocol.py` is
   genuinely good (mocked SSE, erase-count assertions, defaults wiring, rollup notes), but the
   failure mode that matters operationally — server started *without* `--slot-save-path` → HTTP
   501 from `POST /slots` → unhandled `HTTPError` kills the cell — is untested and undescribed.
2. **No test guards the actual regression that occurred:** a rollup whose `ttft_ms*` rows mix
   protocols must either be refused or be required to say so in a place the README renders.
   Today nothing fails when the column becomes mixed; only the README-vs-CSV value diff fires,
   and only after the fact.
3. **The §3c measurements are not reproducible by a reader.** Raw run dirs are gitignored, so
   neither the 3-cell comparison table nor the cold column can be re-derived from the repo. Either
   publish a small tracked per-request extract for the protocol comparison (the way `docs/finbench/`
   vendors the upstream snapshot) or point at a run directory that ships.
4. **ROCm cold has never been measured** — so every cross-backend statement about the new
   protocol is unverified, including the one the README table now implies.
5. **No repeat of the cold pass**, hence no noise band (I4).
6. **No test pins the status text** (`"has not been run"`, `"stays open work"`) to the code/docs
   state, which is why three documents drifted in the same change.
7. Not verified by this review: chart freshness beyond the suite's own fingerprint test (it
   passed — the charts plot `tg512`/`pp2048`/`tok_per_w`, none of which the cold pass touches),
   and the FIN-bench driver's network path (read-only, timeouts present, no token).

---

## 6. Recommended next step

In this order, in one PR:

1. **Do not commit or push the current `results/` diff, and do not create the tag the driver
   printed** (`run-20261024-1`) — `scripts/run_sweep.py:713-724` (`results_match_head()`) is the
   mechanical guard for that, and the README is inconsistent with the rollup right now.
2. Run the missing backend: `scripts/run_sweep.py --backend rocm --kv f16 --ttft-protocol cold
   --probe-only`, then `--consolidate`. (Both backends, both protocols if you keep warm as a
   column.)
3. Fix B2: rewrite §3c from `run-20261024-0843`'s real per-request values, drop or narrow
   "mean and p50 agree", drop "`gen_tps` identical", add the Q6_K cold-stall note, and reconcile
   §7:331/335, README:474-479, PLAN D6/D9.
4. Fix I1/I2/I3 in code (`kv` default from matrix, `partial` stamped at start, label every
   streaming row), each with the pinning test the existing suite already models.
5. Regenerate README table + prose + tests together; charts are expected unchanged — confirm
   with `scripts/make_charts.py` and the suite.
6. Re-run `/usr/bin/python3 -m unittest discover -s tests -t .` and require **98/98 green** (98
   today; more once the new pins land) before tagging.

Scope note: this review authorises nothing. It does not approve publishing or pushing the current
numbers, creating the run tag, or any change to a published figure that is not backed by a
re-measurement; the reference machine's state (a sweep finished at 09:02, worktree dirty) is the
operator's to resolve.
