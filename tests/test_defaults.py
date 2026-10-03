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


if __name__ == "__main__":
    unittest.main()
