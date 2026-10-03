"""Tests for the llama-bench table parser (extracted from run_bench as parse_bench)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import run_sweep  # noqa: E402


# Captured verbatim from a real llama-bench -r 3 run (build 5b59b83f4) — see
# results/run-20261002-1751/rocm/qwen3.8-27b-q6_k/bench.log. The t/s cell is
# "mean ± stddev" because -r > 1.
GOLDEN_TABLE = """\
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  99 |           pp512 |        345.31 ± 4.63 |

build: 5b59b83f4 (1)

--- stderr ---
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 98304 MiB)
"""


class ParseBenchTest(unittest.TestCase):
    def test_parses_mean_and_stddev(self):
        got = run_sweep.parse_bench(GOLDEN_TABLE)
        self.assertEqual(set(got), {"pp512"})
        self.assertEqual(got["pp512"]["mean"], 345.31)
        self.assertEqual(got["pp512"]["stddev"], 4.63)

    def test_parses_multiple_rows(self):
        table = (
            GOLDEN_TABLE
            + "| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  99 |          pp2048 |        336.27 ± 1.24 |\n"
            + "| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  99 |           tg512 |          9.84 ± 0.00 |\n"
        )
        got = run_sweep.parse_bench(table)
        self.assertEqual(set(got), {"pp512", "pp2048", "tg512"})
        self.assertEqual(got["pp2048"]["mean"], 336.27)
        self.assertEqual(got["pp2048"]["stddev"], 1.24)
        self.assertEqual(got["tg512"]["mean"], 9.84)
        self.assertEqual(got["tg512"]["stddev"], 0.0)

    def test_single_run_no_stddev(self):
        # -r 1 prints a bare number (no "±"), so stddev must be None, not a crash.
        table = (
            "| model | size | params | backend | ngl | test | t/s |\n"
            "| --- | ---: | ---: | --- | --: | ---: | ---: |\n"
            "| m | 1.00 GiB | 1.00 B | ROCm | 99 | tg512 | 12.34 |\n"
        )
        got = run_sweep.parse_bench(table)
        self.assertEqual(got["tg512"]["mean"], 12.34)
        self.assertIsNone(got["tg512"]["stddev"])

    def test_empty_raises(self):
        with self.assertRaises(RuntimeError):
            run_sweep.parse_bench("no table here\njust log lines\n")


if __name__ == "__main__":
    unittest.main()
