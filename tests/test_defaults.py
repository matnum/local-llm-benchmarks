"""Tests that matrix.yaml `defaults` is actually wired into the driver (I-1).

Before the fix these knobs existed in matrix.yaml but were hardcoded as literals
in run_sweep.py, so editing them silently did nothing. build_parser() now derives
argparse defaults from the matrix; these tests lock that in.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import run_sweep  # noqa: E402


def _backend_choices(parser):
    return [a.choices for a in parser._actions if a.dest == "backend"][0]


class DefaultsWiredTest(unittest.TestCase):
    def test_matrix_defaults_drive_argparse(self):
        defaults = run_sweep.load_matrix()["defaults"]
        p = run_sweep.build_parser(defaults)
        self.assertEqual(p.get_default("repeats"), defaults["repeats"])
        self.assertEqual(p.get_default("tg_tokens"), defaults["tg_tokens"])
        self.assertEqual(p.get_default("ctx"), defaults["ctx_size"])
        self.assertEqual(p.get_default("threads"), defaults["threads_cpu"])
        self.assertEqual(p.get_default("warmup"), defaults["warmup_requests"])
        self.assertEqual(p.get_default("power_hz"), defaults["power_sample_hz"])
        self.assertEqual(p.get_default("pp_tokens"),
                         ",".join(str(x) for x in defaults["pp_tokens"]))
        self.assertEqual(p.get_default("ngl"), 99)  # gpu_layers "all" -> 99
        self.assertEqual(_backend_choices(p), defaults["backends"])
        self.assertIsNone(p.get_default("backend"))  # required, not defaulted

    def test_modified_defaults_change_parser(self):
        # Proof of wiring: a different defaults dict must change the parser defaults.
        p = run_sweep.build_parser({
            "repeats": 7, "threads_cpu": 4, "ctx_size": 4096,
            "pp_tokens": [256], "tg_tokens": 128,
            "warmup_requests": 2, "power_sample_hz": 10,
            "gpu_layers": 42, "backends": ["rocm"],
        })
        self.assertEqual(p.get_default("repeats"), 7)
        self.assertEqual(p.get_default("threads"), 4)
        self.assertEqual(p.get_default("ctx"), 4096)
        self.assertEqual(p.get_default("pp_tokens"), "256")
        self.assertEqual(p.get_default("tg_tokens"), 128)
        self.assertEqual(p.get_default("warmup"), 2)
        self.assertEqual(p.get_default("power_hz"), 10)
        self.assertEqual(p.get_default("ngl"), 42)  # numeric gpu_layers passthrough
        self.assertEqual(_backend_choices(p), ["rocm"])

    def test_matrix_model_paths_are_home_relative(self):
        # matrix.yaml is committed and public-facing: it must not leak the local
        # machine's username via an absolute /home/<user>/... model path. The driver
        # expands the ~/ prefix at run time (see run_sweep.main).
        matrix = run_sweep.load_matrix()
        for cell in matrix["cells"]:
            self.assertTrue(cell["file"].startswith("~/"), cell["file"])
            self.assertNotIn("/home/", cell["file"])

    def test_kv_cache_type_is_selectable_and_defaults_to_the_old_setting(self):
        """--kv exists, defaults to q8_0 (the historical protocol) and rejects junk.

        TTFT on some Vulkan cells is dominated by quantised-KV bookkeeping, so the
        probe has to be able to run with f16 KV and say which it used (RUNNING §7).
        """
        p = run_sweep.build_parser({})
        self.assertEqual(p.parse_args([]).kv, "q8_0")
        self.assertEqual(p.parse_args(["--kv", "f16"]).kv, "f16")
        with self.assertRaises(SystemExit):
            p.parse_args(["--kv", "q4_0"])

    def test_stack_versions_are_reportable(self):
        """D7: numbers are stack-stamped, so the driver must report the Mesa string."""
        self.assertIn("Mesa", run_sweep.mesa_version())


if __name__ == "__main__":
    unittest.main()
