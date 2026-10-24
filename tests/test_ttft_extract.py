"""The tracked per-request extract must agree with the rollup it explains.

`docs/RUNNING.md` §3c/§7 publish per-request TTFT values (the whole argument for
the cold protocol is that the *spread* collapsed), but raw run directories are
gitignored — without an extract those numbers are unverifiable by a reader, which
is the defect review 2026-10-24 raised. `results/ttft-per-request-20261024.csv` is
that extract for the published cold pass.

This test re-derives the rollup's `ttft_ms`/`ttft_ms_p50` from the extract, so the
extract and the published column cannot drift apart: edit one and the other fails.
"""
import csv
import os
import statistics
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT = os.path.join(REPO, "results", "ttft-per-request-20261024.csv")
ROLLUP = os.path.join(REPO, "results", "consolidated.csv")

RUNS = {("run-20261024-0910", "vulkan"), ("run-20261024-0930", "rocm")}


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@unittest.skipUnless(os.path.isfile(EXTRACT) and os.path.isfile(ROLLUP),
                     "extract or rollup missing")
class PerRequestExtractTest(unittest.TestCase):

    def setUp(self):
        self.rows = _read(EXTRACT)
        self.rollup = {}
        for r in _read(ROLLUP):
            self.rollup[(r["run_id"], r["backend"], r["cell"], r["metric"])] = float(r["value"])

    def test_extract_covers_the_published_cold_runs(self):
        pairs = {(r["run_id"], r["backend"]) for r in self.rows}
        self.assertEqual(pairs, RUNS, "the extract must describe exactly the published runs")
        self.assertEqual({r["protocol"] for r in self.rows}, {"cold"})
        self.assertEqual(len(self.rows), 60, "10 cells x 2 backends x 3 requests")

    def test_erased_tokens_show_the_cache_was_cleared(self):
        """Every timed request after the warmup must have dropped a populated cache."""
        for r in self.rows:
            with self.subTest(run=r["run_id"], cell=r["cell"], backend=r["backend"],
                              request=r["request"]):
                self.assertGreater(int(r["erased_tokens"]), 0,
                                   "a cold run whose erase dropped nothing is a warm run")

    def test_extract_medians_reproduce_the_published_column(self):
        groups = {}
        for r in self.rows:
            key = (r["run_id"], r["backend"], r["cell"])
            groups.setdefault(key, []).append(float(r["ttft_ms"]))
        for key, vals in groups.items():
            run_id, backend, cell = key
            with self.subTest(run=run_id, backend=backend, cell=cell):
                self.assertEqual(len(vals), 3)
                self.assertAlmostEqual(statistics.median(vals),
                                       self.rollup[key + ("ttft_ms_p50",)], delta=0.15,
                                       msg="extract median != published p50")
                self.assertAlmostEqual(statistics.fmean(vals),
                                       self.rollup[key + ("ttft_ms",)], delta=0.15,
                                       msg="extract mean != published mean")

    def test_published_rows_cite_these_runs(self):
        """The README's cold column must come from the runs the extract documents."""
        for (run_id, backend, cell, metric) in self.rollup:
            if metric != "ttft_ms_p50" or (run_id, backend) not in RUNS:
                continue
            with self.subTest(cell=cell, backend=backend):
                self.assertIn(cell, {r["cell"] for r in self.rows})


if __name__ == "__main__":
    unittest.main()
