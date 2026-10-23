"""Guards for the resident-GPU-memory metric (M-1).

The 2026-10-22 re-measure would have published a 59 GiB model as "0.87 GiB VRAM
peak". After the UMA carve-out was shrunk the weights live in GTT, and the
sampler was reading `mem_info_vram_used` only — a counter that then sits at its
idle floor no matter how big the model is. Both ends of the fix are pinned here:

  * sampler: read_mem reads both pools and the trace carries them in a fixed
    column order;
  * run_sweep.window_stats: mem_gib_max is the peak of the per-sample SUM, and is
    absent (not zero, not copied from VRAM) when the trace has no GTT column;
  * run_sweep.consolidate: mem_peak_gib appears in the rollup only when it was
    measured, and vram_peak_gib says what it actually is.
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import run_sweep  # noqa: E402
import sampler  # noqa: E402

GIB = 1024 ** 3


def _resource_rows(with_gtt, samples):
    """Build resources.csv dicts. `samples` is [(vram_gib, gtt_gib), ...]."""
    rows = []
    for i, (vram, gtt) in enumerate(samples):
        row = {"ts": f"{1000.5 + i}", "power_w": "80.00", "temp_c": "70.0",
               "gpu_busy_pct": "99", "vram_used_gib": f"{vram:.2f}",
               "sys_avail_gib": "60.00"}
        if with_gtt:
            row["gtt_used_gib"] = f"{gtt:.2f}"
        rows.append(row)
    return rows


class SamplerPoolsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.drm = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name, value):
        with open(os.path.join(self.drm, name), "w") as f:
            f.write(f"{value}\n")

    def test_reads_both_pools_in_gib(self):
        self._write("mem_info_vram_used", 2 * GIB)
        self._write("mem_info_gtt_used", 59 * GIB)
        mem = sampler.read_mem(self.drm)
        self.assertAlmostEqual(mem["vram"], 2.0, places=6)
        self.assertAlmostEqual(mem["gtt"], 59.0, places=6)

    def test_missing_pool_is_none_not_zero(self):
        self._write("mem_info_vram_used", 2 * GIB)
        mem = sampler.read_mem(self.drm)
        self.assertAlmostEqual(mem["vram"], 2.0, places=6)
        self.assertIsNone(mem["gtt"])

    def test_trace_header_carries_both_pools(self):
        fields = list(sampler.FIELDS)
        self.assertIn("gtt_used_gib", fields)
        self.assertLess(fields.index("vram_used_gib"), fields.index("gtt_used_gib"))
        self.assertLess(fields.index("gtt_used_gib"), fields.index("sys_avail_gib"))


class WindowMemStatsTest(unittest.TestCase):
    def test_peak_is_the_sum_of_the_pools_in_one_sample(self):
        # The VRAM peak and the GTT peak deliberately land on different samples:
        # summing the two maxima would report 45 GiB for a 39 GiB model.
        rows = _resource_rows(True, [(0.5, 20.0), (0.9, 40.5), (3.0, 1.0)])
        stats = run_sweep.window_stats(rows, 0, 1e9)
        self.assertAlmostEqual(stats["vram_gib_max"], 3.0)
        self.assertAlmostEqual(stats["gtt_gib_max"], 40.5)
        self.assertAlmostEqual(stats["mem_gib_max"], 41.4)

    def test_pre_gtt_trace_gets_no_mem_peak(self):
        rows = _resource_rows(False, [(5.0, 0.0), (59.1, 0.0)])
        stats = run_sweep.window_stats(rows, 0, 1e9)
        self.assertAlmostEqual(stats["vram_gib_max"], 59.1)
        self.assertNotIn("mem_gib_max", stats)
        self.assertNotIn("gtt_gib_max", stats)

    def test_window_is_bounded(self):
        rows = _resource_rows(True, [(0.5, 1.0), (0.5, 50.0)])
        stats = run_sweep.window_stats(rows, 0, 1001.0)
        self.assertAlmostEqual(stats["mem_gib_max"], 1.5)


class IdleGateTest(unittest.TestCase):
    """M-1: the contamination gate must see GTT, not only the VRAM pool."""

    def _idle(self, busy, vram_gib, gtt_gib):
        values = {"gpu_busy_percent": busy,
                  "mem_info_vram_used": int(vram_gib * GIB),
                  "mem_info_gtt_used": int(gtt_gib * GIB)}

        def fake_rd(path):
            return values[os.path.basename(path)]

        with mock.patch.object(run_sweep, "rd", side_effect=fake_rd):
            return run_sweep.gpu_idle()

    def test_foreign_server_holding_gtt_is_not_idle(self):
        # 23 GiB llama-server in GTT, VRAM pool at its 0.14 GiB floor.
        ok, busy, mem = self._idle(2, 0.14, 23.0)
        self.assertFalse(ok)
        self.assertAlmostEqual(mem, 23.14, places=2)

    def test_empty_gpu_is_idle(self):
        ok, busy, mem = self._idle(1, 0.14, 0.02)
        self.assertTrue(ok)
        self.assertAlmostEqual(mem, 0.16, places=2)

    def test_missing_gtt_counter_degrades_to_vram(self):
        values = {"gpu_busy_percent": 1, "mem_info_vram_used": int(0.14 * GIB)}

        with mock.patch.object(run_sweep, "rd",
                               side_effect=lambda p: values.get(os.path.basename(p))):
            ok, _busy, mem = run_sweep.gpu_idle()
        self.assertTrue(ok)
        self.assertAlmostEqual(mem, 0.14, places=2)


def _cell_json(cell_id, backend, run_id, mem_gib_max=None):
    cell_window = {"power_w_mean": 65.0, "power_w_max": 100.0, "temp_c_max": 75.0,
                   "vram_gib_max": 0.87, "sys_avail_gib_min": 60.03}
    if mem_gib_max is not None:
        cell_window["mem_gib_max"] = mem_gib_max
        cell_window["gtt_gib_max"] = mem_gib_max - 0.87
    return {
        "run": run_id, "backend": backend, "cell": cell_id,
        "model": "m", "quant": "q", "size_gib": 59.0,
        "bench": {"pp512": {"mean": 100.0, "stddev": 1.0, "repeats": 3},
                  "pp2048": {"mean": 90.0, "stddev": 2.0, "repeats": 3},
                  "tg512": {"mean": 10.0, "stddev": 0.1, "repeats": 3}},
        "ttft": {"requests": [{"ttft_ms": 50.0, "tokens": 512} for _ in range(3)],
                 "gen_tps_mean": 10.0, "ttft_ms_mean": 55.0, "ttft_ms_p50": 52.0},
        "resources": {
            "probe_window": {"power_w_mean": 80.0, "power_w_max": 90.0,
                             "temp_c_max": 70.0, "vram_gib_max": 0.87,
                             "sys_avail_gib_min": 60.03},
            "bench_window": {"power_w_mean": 60.0, "power_w_max": 100.0,
                             "temp_c_max": 70.0, "vram_gib_max": 0.87,
                             "sys_avail_gib_min": 60.03},
            "cell_window": cell_window,
        },
        "tok_per_w": 0.5,
    }


class RollupMemMetricTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        os.makedirs(os.path.join(self.repo, "results"), exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_run(self, run_id, backend, cell_id, mem_gib_max):
        bdir = os.path.join(self.repo, "results", run_id, backend)
        cdir = os.path.join(bdir, cell_id)
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(bdir, "run.json"), "w") as f:
            json.dump({"run_id": run_id, "backend": backend}, f)
        with open(os.path.join(cdir, "cell.json"), "w") as f:
            json.dump(_cell_json(cell_id, backend, run_id, mem_gib_max), f)

    def _by_metric(self):
        with open(os.path.join(self.repo, "results", "consolidated.csv"), newline="") as f:
            return {r["metric"]: r for r in csv.DictReader(f)}

    def test_measured_trace_rolls_up_mem_peak(self):
        self._write_run("run-A", "vulkan", "cellA", mem_gib_max=60.5)
        run_sweep.consolidate(repo=self.repo)
        m = self._by_metric()
        self.assertAlmostEqual(float(m["mem_peak_gib"]["value"]), 60.5)
        self.assertIn("VRAM + GTT", m["mem_peak_gib"]["note"])
        self.assertIn("residency = mem_peak_gib", m["vram_peak_gib"]["note"])

    def test_pre_gtt_run_publishes_no_mem_peak(self):
        self._write_run("run-A", "vulkan", "cellA", mem_gib_max=None)
        run_sweep.consolidate(repo=self.repo)
        m = self._by_metric()
        self.assertNotIn("mem_peak_gib", m)
        self.assertAlmostEqual(float(m["vram_peak_gib"]["value"]), 0.87)
        self.assertEqual(m["vram_peak_gib"]["note"], "")


class PartialRunExclusionTest(unittest.TestCase):
    """M-2: a --cells probe must never become a published row.

    The driver already refuses to consolidate a partial sweep, but consolidate()
    scans every run dir, so the next full sweep used to fold the probe's rows back
    in as the "latest" value for the cells it touched (run-20261022-0917).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        os.makedirs(os.path.join(self.repo, "results"), exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_run(self, run_id, partial):
        bdir = os.path.join(self.repo, "results", run_id, "vulkan")
        cdir = os.path.join(bdir, "cellA")
        os.makedirs(cdir, exist_ok=True)
        meta = {"run_id": run_id, "backend": "vulkan"}
        if partial is not None:
            meta["partial"] = partial
        with open(os.path.join(bdir, "run.json"), "w") as f:
            json.dump(meta, f)
        with open(os.path.join(cdir, "cell.json"), "w") as f:
            json.dump(_cell_json("cellA", "vulkan", run_id, mem_gib_max=60.5), f)

    def _run_ids(self):
        with open(os.path.join(self.repo, "results", "consolidated.csv"), newline="") as f:
            return {r["run_id"] for r in csv.DictReader(f)}

    def test_partial_run_is_skipped(self):
        self._write_run("run-full", partial=False)
        self._write_run("run-probe", partial=True)
        run_sweep.consolidate(repo=self.repo)
        self.assertEqual(self._run_ids(), {"run-full"})

    def test_run_without_the_flag_still_rolls_up(self):
        # Runs recorded before the flag existed carry no key and must keep working.
        self._write_run("run-old", partial=None)
        run_sweep.consolidate(repo=self.repo)
        self.assertEqual(self._run_ids(), {"run-old"})


if __name__ == "__main__":
    unittest.main()
