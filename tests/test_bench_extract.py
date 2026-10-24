"""The bench, power, temperature and memory columns must be checkable, not trusted.

The TTFT column got a tracked per-request extract (review 5.3). The same review
found the other published columns had no equivalent: `results/consolidated.csv`
asserted `pp2048`/`tg512` means and window aggregates that a reader could only
believe.

What is derivable here is bounded by what the harness stores. `llama-bench -r 3`
prints one row of `mean ± sd` and does not keep the three repeats — not in the
log, not in `cell.json` — so a per-repeat extract cannot be built from existing
data. The derivable artifact is the log itself: `results/bench-logs/` carries the
verbatim `llama-bench` output for the two sweeps that own the published bench
columns, and this test re-runs the driver's own `parse_bench()` over it, so a
retyped or mis-ingested figure fails.

Power, temperature and memory are aggregates of a 5 Hz trace (12k rows per sweep);
shipping the traces would outweigh this repository. `results/sampler-support.csv`
instead publishes, per window, the sample count, duration and the aggregate the
rollup copies — so the support behind a published figure is visible, and a
window backed by a handful of samples fails here.
"""
import csv
import glob
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import run_sweep  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLUP = os.path.join(REPO, "results", "consolidated.csv")
BENCH_LOGS = os.path.join(REPO, "results", "bench-logs")
SUPPORT = os.path.join(REPO, "results", "sampler-support.csv")

# The sweeps that own the published bench / power / memory columns (REVIEW: the
# 2026-10-22 full sweeps). Streaming columns come from the 2026-10-24 probes.
PUBLISHED_RUNS = {("run-20261022-2042", "vulkan"), ("run-20261022-2124", "rocm")}

BENCH_METRICS = ("pp512_bench_tps", "pp2048_bench_tps", "tg512_bench_tps")
SD = re.compile(r"sd ±([\d.]+)")
NOTE_REPEATS = re.compile(r"llama-bench mean of (\d+)")

# metric -> (supported window, field in the support file)
SUPPORTED = {
    "power_w_probe": ("probe_window", "power_w_mean"),
    "power_w_bench": ("bench_window", "power_w_mean"),
    "temp_c_max": ("cell_window", "temp_c_max"),
    "mem_peak_gib": ("cell_window", "mem_gib_max"),
    "vram_peak_gib": ("cell_window", "vram_gib_max"),
}

# 5 Hz sampler: a published window must rest on seconds of data, not two samples.
MIN_SAMPLES = 100
MIN_DURATION_S = 20.0


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@unittest.skipUnless(os.path.isfile(ROLLUP), "no tracked rollup")
class BenchLogTest(unittest.TestCase):

    def setUp(self):
        self.rows = {}
        for r in _read(ROLLUP):
            self.rows[(r["run_id"], r["backend"], r["cell"], r["metric"])] = r
        self.logs = sorted(glob.glob(os.path.join(BENCH_LOGS, "*", "*", "*.log")))
        self.assertTrue(self.logs, "results/bench-logs/ is tracked — it may not be empty")

    def latest(self, backend, cell, metric):
        """The row consolidate() publishes: the highest run id for that metric."""
        found = [k for k in self.rows if k[1] == backend and k[2] == cell and k[3] == metric]
        return self.rows[max(found, key=lambda k: k[0])]

    def test_every_vendored_log_reproduces_its_rollup_row(self):
        for path in self.logs:
            run_id, backend = path.split(os.sep)[-3:-1]
            cell = os.path.basename(path)[:-len(".log")]
            with open(path) as f:
                parsed = run_sweep.parse_bench(f.read())
            for test, stats in parsed.items():
                metric = f"{test}_bench_tps"
                row = self.rows.get((run_id, backend, cell, metric))
                if row is None:
                    self.fail(f"the log measures {metric}, which the rollup lacks")
                with self.subTest(run=run_id, backend=backend, cell=cell, metric=metric):
                    self.assertAlmostEqual(stats["mean"], float(row["value"]), delta=0.01,
                                           msg="rollup mean != vendored llama-bench mean")
                    sd = SD.search(row["note"] or "")
                    if stats["stddev"] is None:
                        self.assertIsNone(sd, "rollup claims an sd the log does not print")
                    else:
                        self.assertIsNotNone(sd, "rollup dropped the log's ±sd")
                        self.assertAlmostEqual(stats["stddev"], float(sd.group(1)),
                                               delta=0.01, msg="rollup sd != log sd")
                    reps = NOTE_REPEATS.search(row["note"] or "")
                    self.assertIsNotNone(reps)
                    self.assertEqual(int(reps.group(1)), 3)

    def test_the_vendored_set_is_the_published_set(self):
        vendored = {(p.split(os.sep)[-3], p.split(os.sep)[-2]) for p in self.logs}
        self.assertEqual(vendored, PUBLISHED_RUNS,
                         "bench-logs must cover exactly the sweeps that publish bench columns")
        cells = {os.path.basename(p)[:-4] for p in self.logs}
        self.assertEqual(len(cells), 10, "one log per matrix cell, per sweep")

    def test_published_bench_rows_come_from_vendored_logs(self):
        for backend in ("vulkan", "rocm"):
            for cell in sorted({k[2] for k in self.rows}):
                for metric in BENCH_METRICS:
                    row = self.latest(backend, cell, metric)
                    with self.subTest(backend=backend, cell=cell, metric=metric):
                        self.assertIn((row["run_id"], backend), PUBLISHED_RUNS,
                                      f"published {metric} comes from {row['run_id']}, "
                                      f"which has no vendored log")


@unittest.skipUnless(os.path.isfile(ROLLUP), "no tracked rollup")
class SamplerSupportTest(unittest.TestCase):

    def setUp(self):
        self.rows = {}
        for r in _read(ROLLUP):
            self.rows[(r["run_id"], r["backend"], r["cell"], r["metric"])] = r
        self.support = {(r["run_id"], r["backend"], r["cell"], r["window"]): r
                        for r in _read(SUPPORT)}

    def latest(self, backend, cell, metric):
        found = [k for k in self.rows if k[1] == backend and k[2] == cell and k[3] == metric]
        return self.rows[max(found, key=lambda k: k[0])]

    def test_every_window_has_measured_support(self):
        self.assertTrue(self.support, "results/sampler-support.csv is tracked")
        for key, r in self.support.items():
            with self.subTest(run=key[0], backend=key[1], cell=key[2], window=key[3]):
                self.assertGreaterEqual(int(r["n_samples"]), MIN_SAMPLES,
                                        f"window backed by only {r['n_samples']} samples")
                self.assertGreaterEqual(float(r["duration_s"]), MIN_DURATION_S)

    def test_support_covers_three_windows_per_cell(self):
        cells = {(k[0], k[1], k[2]) for k in self.support}
        for key in cells:
            with self.subTest(run=key[0], backend=key[1], cell=key[2]):
                self.assertEqual({k[3] for k in self.support if k[:3] == key},
                                 {"bench_window", "probe_window", "cell_window"})

    def test_published_power_temp_mem_match_the_support_rows(self):
        for backend in ("vulkan", "rocm"):
            for cell in sorted({k[2] for k in self.rows}):
                for metric, (window, field) in SUPPORTED.items():
                    row = self.latest(backend, cell, metric)
                    key = (row["run_id"], backend, cell, window)
                    with self.subTest(backend=backend, cell=cell, metric=metric):
                        self.assertIn(key, self.support,
                                      f"{metric} is published from {row['run_id']} but that "
                                      f"window has no support row")
                        self.assertAlmostEqual(float(row["value"]),
                                               float(self.support[key][field]), delta=0.01,
                                               msg=f"{metric} != support {field}")


if __name__ == "__main__":
    unittest.main()
