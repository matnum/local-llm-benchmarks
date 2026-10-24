"""Guards for `--probe-only` and the run tag (M8).

Two things are pinned here:

  * A **probe-only** run re-measures one phase. It must never look like a full
    sweep: `consolidate()` publishes its streaming columns only, because its cell
    window covers the probe alone — no bench heat soak — so temperature,
    residency and tokens-per-watt from it would understate the cell those numbers
    are published from.
  * The run **tag** is derived, not hand-typed. The 2026-10-22 sweeps went
    untagged because the name/message were typed by a human at the end of a long
    run; `next_run_tag()` and `tag_message()` now produce both from the run.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import run_sweep  # noqa: E402

STREAMING = {"tg_server_tps", "ttft_ms", "ttft_ms_p50"}
WHOLE_CELL = {"temp_c_max", "vram_peak_gib", "mem_peak_gib", "tok_per_w",
              "power_w_probe", "power_w_bench"}


def _ttft():
    return {"gen_tps_mean": 55.3, "ttft_ms_mean": 108.1, "ttft_ms_p50": 116.6,
            "requests": [{}, {}, {}]}


def _windows():
    win = {"power_w_mean": 80.0, "power_w_max": 90.0, "temp_c_max": 70.0,
           "vram_gib_max": 0.87, "gtt_gib_max": 58.8, "mem_gib_max": 59.7,
           "sys_avail_gib_min": 60.0}
    return {"probe_window": dict(win), "bench_window": dict(win), "cell_window": dict(win)}


def _cell_json(run, backend, cell, probe_only, kv="f16"):
    return {
        "cell": cell, "backend": backend, "run": run, "kv": kv,
        "probe_only": probe_only, "model": "gpt-oss-120b", "quant": "MXFP4",
        "size_gib": 59.0,
        "bench": {} if probe_only else {
            "pp2048": {"mean": 735.3, "stddev": 27.3, "repeats": 3},
            "tg512": {"mean": 55.3, "stddev": 0.2, "repeats": 3},
        },
        "ttft": _ttft(),
        "resources": ({"probe_window": _windows()["probe_window"],
                       "bench_window": None,
                       "cell_window": _windows()["probe_window"]} if probe_only
                      else _windows()),
        "tok_per_w": 0.63,
    }


class RollupProbeOnlyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        os.makedirs(os.path.join(self.repo, "results"), exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_run(self, run_id, probe_only, kv="f16"):
        bdir = os.path.join(self.repo, "results", run_id, "vulkan")
        cdir = os.path.join(bdir, "gpt-oss-120b-mxfp4")
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(bdir, "run.json"), "w") as f:
            json.dump({"run_id": run_id, "backend": "vulkan", "partial": False,
                       "kv_cache": kv, "probe_only": probe_only}, f)
        with open(os.path.join(cdir, "cell.json"), "w") as f:
            json.dump(_cell_json(run_id, "vulkan", "gpt-oss-120b-mxfp4", probe_only, kv), f)

    def _rows(self):
        with open(os.path.join(self.repo, "results", "consolidated.csv"), newline="") as f:
            return list(csv.DictReader(f))

    def test_probe_only_run_publishes_streaming_columns_only(self):
        self._write_run("run-20261023-1000", probe_only=True, kv="f16")
        run_sweep.consolidate(repo=self.repo)
        metrics = {r["metric"] for r in self._rows()}
        self.assertEqual(metrics, STREAMING,
                         f"a probe-only run must not publish whole-cell metrics: "
                         f"{sorted(metrics - STREAMING)}")

    def test_probe_only_notes_name_the_kv_type_and_the_run_type(self):
        self._write_run("run-20261023-1000", probe_only=True, kv="f16")
        run_sweep.consolidate(repo=self.repo)
        for r in self._rows():
            with self.subTest(metric=r["metric"]):
                # Legacy cell.json: no ttft_protocol stamp, so the note must say
                # `warm` — those runs reused the prompt cache.
                self.assertIn("f16 KV, warm TTFT, probe-only run", r["note"])

    def test_probe_only_run_does_not_override_the_full_sweep_metrics(self):
        """Latest-wins is per metric: the TTFT column moves, heat/memory must not."""
        self._write_run("run-20261022-2042", probe_only=False, kv="q8_0")
        self._write_run("run-20261023-1000", probe_only=True, kv="f16")
        run_sweep.consolidate(repo=self.repo)
        latest = {}
        for r in self._rows():
            latest[r["metric"]] = r                     # file order: newest run last
        self.assertEqual(latest["ttft_ms_p50"]["run_id"], "run-20261023-1000")
        self.assertEqual(latest["tg_server_tps"]["run_id"], "run-20261023-1000")
        with self.subTest("whole-cell metrics stay on the full sweep"):
            for metric in WHOLE_CELL:
                self.assertEqual(latest[metric]["run_id"], "run-20261022-2042",
                                 f"{metric} must stay attributed to the full sweep")
        with self.subTest("bench rows unchanged"):
            self.assertEqual(latest["pp2048_bench_tps"]["run_id"], "run-20261022-2042")
            self.assertAlmostEqual(float(latest["pp2048_bench_tps"]["value"]), 735.3)


class ParserProbeOnlyTest(unittest.TestCase):
    def test_flag_defaults_off_and_is_selectable(self):
        p = run_sweep.build_parser({})
        self.assertFalse(p.parse_args([]).probe_only)
        self.assertTrue(p.parse_args(["--probe-only"]).probe_only)
        self.assertFalse(p.parse_args([]).tag)
        self.assertTrue(p.parse_args(["--tag"]).tag)


class RunTagTest(unittest.TestCase):
    def test_first_tag_of_a_day_is_one(self):
        self.assertEqual(run_sweep.next_run_tag([], "20261023"), "run-20261023-1")

    def test_sequence_continues_from_the_highest_number(self):
        existing = ["run-20261022-1", "run-20261022-2", "run-20261022-10"]
        self.assertEqual(run_sweep.next_run_tag(existing, "20261022"), "run-20261022-11")

    def test_other_dates_and_junk_are_ignored(self):
        existing = ["run-20261003-4", "finbench-20261003", "run-20261022-1",
                    "run-20261023-x", "run-20261023"]
        self.assertEqual(run_sweep.next_run_tag(existing, "20261023"), "run-20261023-1")

    def test_message_carries_the_provenance(self):
        msg = run_sweep.tag_message("rocm", 10, "b2cb81c9f00", "5b59b83f4e2",
                                    "7.2.10-1-cachyos", "Mesa 26.2.4-arch3.1",
                                    "f16", probe_only=True)
        self.assertIn("ROCm probe-only sweep", msg)
        self.assertIn("10 cells", msg)
        self.assertIn("b2cb81c", msg)
        self.assertIn("5b59b83f4", msg)
        self.assertIn("7.2.10-1-cachyos / Mesa 26.2.4-arch3.1", msg)
        self.assertIn("KV f16", msg)

    def test_full_sweep_message_says_full(self):
        msg = run_sweep.tag_message("vulkan", 10, "abc1234", "def5678",
                                    "k", "m", "q8_0", probe_only=False)
        self.assertIn("Vulkan full sweep", msg)
        self.assertIn("KV q8_0", msg)

    def test_pending_tags_are_counted_so_back_to_back_sweeps_differ(self):
        """A sweep stamps its tag but the human commits later: the next sweep must not reuse it."""
        with tempfile.TemporaryDirectory() as repo:
            for run, backend, tag in (("run-A", "vulkan", "run-20261023-1"),
                                      ("run-B", "rocm", "run-20261023-2"),
                                      ("run-C", "vulkan", None)):
                d = os.path.join(repo, "results", run, backend)
                os.makedirs(d)
                with open(os.path.join(d, "run.json"), "w") as f:
                    json.dump({"run_id": run, "backend": backend, "tag": tag}, f)
            broken = os.path.join(repo, "results", "run-D", "vulkan")
            os.makedirs(broken)
            with open(os.path.join(broken, "run.json"), "w") as f:
                f.write("not json")
            names = run_sweep.planned_tags(repo)
            self.assertEqual(sorted(names), ["run-20261023-1", "run-20261023-2"])
            self.assertEqual(run_sweep.next_run_tag(names, "20261023"), "run-20261023-3")


if __name__ == "__main__":
    unittest.main()
