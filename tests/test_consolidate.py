"""Tests for consolidate(): rebuild-not-append, no duplicates, metric names.

Regression guards for the two historical bugs this repo already shipped once:
  B3 — consolidate() appended to consolidated.csv (rows doubled every call);
  I2 — table.md looked up "power_w_prob" instead of "power_w_probe" (blank column).
"""

import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import run_sweep  # noqa: E402


def _cell(cell_id, backend, run_id):
    """A minimal but schema-complete cell.json (11 metrics per cell when rolled up)."""
    return {
        "run": run_id,
        "backend": backend,
        "cell": cell_id,
        "model": "m",
        "quant": "q",
        "size_gib": 1.0,
        "bench": {
            "pp512": {"mean": 100.0, "stddev": 1.0, "repeats": 3},
            "pp2048": {"mean": 90.0, "stddev": 2.0, "repeats": 3},
            "tg512": {"mean": 10.0, "stddev": 0.1, "repeats": 3},
        },
        "ttft": {
            "requests": [{"ttft_ms": 50.0, "tokens": 512} for _ in range(3)],
            "gen_tps_mean": 10.0,
            "ttft_ms_mean": 55.0,
            "ttft_ms_p50": 52.0,
        },
        "resources": {
            "probe_window": {"power_w_mean": 80.0, "power_w_max": 90.0,
                             "temp_c_max": 70.0, "vram_gib_max": 10.0,
                             "sys_avail_gib_min": 20.0},
            "bench_window": {"power_w_mean": 60.0, "power_w_max": 100.0,
                             "temp_c_max": 70.0, "vram_gib_max": 10.0,
                             "sys_avail_gib_min": 20.0},
            "cell_window": {"power_w_mean": 65.0, "power_w_max": 100.0,
                            "temp_c_max": 75.0, "vram_gib_max": 12.0,
                            "sys_avail_gib_min": 18.0},
        },
        "tok_per_w": 0.5,
    }


def _write_run(repo, run_id, backend, cell_id):
    bdir = os.path.join(repo, "results", run_id, backend)
    os.makedirs(bdir, exist_ok=True)
    with open(os.path.join(bdir, "run.json"), "w") as f:
        json.dump({"run_id": run_id, "backend": backend}, f)
    cdir = os.path.join(bdir, cell_id)
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, "cell.json"), "w") as f:
        json.dump(_cell(cell_id, backend, run_id), f)


class ConsolidateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        os.makedirs(os.path.join(self.repo, "results"), exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _rows(self):
        with open(os.path.join(self.repo, "results", "consolidated.csv"),
                  newline="") as f:
            return list(csv.DictReader(f))

    def test_rebuild_not_append(self):
        _write_run(self.repo, "run-A", "vulkan", "cellA")
        _write_run(self.repo, "run-B", "rocm", "cellB")
        run_sweep.consolidate(repo=self.repo)
        first = self._rows()
        run_sweep.consolidate(repo=self.repo)  # B3: must rebuild, not append
        second = self._rows()
        self.assertEqual(len(first), 2 * 11)   # 2 cells x 11 metrics
        self.assertEqual(first, second)        # identical, no doubling
        keys = [(r["run_id"], r["backend"], r["cell"], r["metric"]) for r in second]
        self.assertEqual(len(keys), len(set(keys)))  # no duplicate rows

    def test_metric_names(self):
        _write_run(self.repo, "run-A", "vulkan", "cellA")
        run_sweep.consolidate(repo=self.repo)
        metrics = {r["metric"] for r in self._rows()}
        self.assertIn("power_w_probe", metrics)   # I2: correct key
        self.assertNotIn("power_w_prob", metrics)
        self.assertIn("tg_server_tps", metrics)
        self.assertIn("ttft_ms_p50", metrics)

    def test_table_md_renders_every_cell(self):
        _write_run(self.repo, "run-A", "vulkan", "cellA")
        _write_run(self.repo, "run-B", "rocm", "cellB")
        run_sweep.consolidate(repo=self.repo)
        with open(os.path.join(self.repo, "results", "table.md")) as f:
            md = f.read()
        self.assertIn("cellA", md)
        self.assertIn("cellB", md)


class RollupOwnershipTest(unittest.TestCase):
    """Only a full sweep may rewrite the published rollup.

    consolidate() scans every run dir, so a one-cell probe (--cells X) would make
    that run the "latest" value for X in consolidated.csv, table.md, the charts and
    every README number pinned to them. That happened once during the TTFT
    investigation; the driver now refuses to consolidate a partial sweep.
    """

    def test_full_matrix_tracks_the_rollup(self):
        self.assertTrue(run_sweep.tracks_rollup(["a", "b", "c"], ["a", "b", "c"]))
        self.assertTrue(run_sweep.tracks_rollup(["b", "a"], ["a", "b"]))

    def test_subset_and_superset_do_not(self):
        self.assertFalse(run_sweep.tracks_rollup(["a"], ["a", "b", "c"]))
        self.assertFalse(run_sweep.tracks_rollup(["a", "b", "c", "d"], ["a", "b", "c"]))


if __name__ == "__main__":
    unittest.main()
