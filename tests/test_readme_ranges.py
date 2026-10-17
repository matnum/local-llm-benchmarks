"""Pin the hand-typed backend-delta percentages in README.md to the rollup.

REVIEW.md I-2 was a wrong headline percentage (18.3%, divided by ROCm instead of
Vulkan) sitting a few lines above a machine-generated table, and it survived a
previous "correction" because nothing checked it. The numbers are still written by
hand, so this recomputes them from results/consolidated.csv using the README's own
convention — ROCm's shortfall as a percentage of Vulkan — and fails when the prose
and the data disagree.
"""
import csv
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLUP = os.path.join(REPO, "results", "consolidated.csv")
README = os.path.join(REPO, "README.md")

# The five cells of the original sweep — what the README's "five cells" refers to.
FIRST_FIVE = {
    "gpt-oss-120b-mxfp4", "qwen3.5-35b-a3b-q6_k", "qwen3-coder-30b-a3b-q6_k",
    "qwen3.8-27b-q5_k_xl", "qwen3.8-27b-q6_k",
}


def _both_backends(metric):
    cells = {}
    with open(ROLLUP, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["metric"] == metric:
                cells.setdefault(row["cell"], {})[row["backend"]] = float(row["value"])
    return {c: v for c, v in cells.items() if {"vulkan", "rocm"} <= set(v)}


def _range(metric, cells=None):
    """ROCm's trail behind Vulkan, formatted the way the README writes it."""
    pct = sorted(abs((v["vulkan"] - v["rocm"]) / v["vulkan"] * 100)
                 for c, v in _both_backends(metric).items()
                 if cells is None or c in cells)
    return f"{pct[0]:.1f}\u2013{pct[-1]:.1f}"


@unittest.skipUnless(os.path.isfile(ROLLUP) and os.path.isfile(README),
                     "rollup or README missing")
class ReadmeRangesTest(unittest.TestCase):

    def setUp(self):
        # Collapse whitespace so a prose reflow cannot break the anchors.
        self.text = " ".join(open(README, encoding="utf-8").read().split())

    def test_prose_ranges_match_rollup(self):
        cases = [
            ("Vulkan is faster on `tg512` on all five cells (ROCm trails by",
             FIRST_FIVE, "tg512_bench_tps"),
            ("ROCm is faster on `pp2048` prefill on all five (by",
             FIRST_FIVE, "pp2048_bench_tps"),
            ("Vulkan wins generation in 10 of 10 (ROCm trails by",
             None, "tg512_bench_tps"),
            ("ROCm wins prefill in 10 of 10 (by",
             None, "pp2048_bench_tps"),
        ]
        for anchor, cells, metric in cases:
            with self.subTest(anchor=anchor):
                self.assertIn(anchor, self.text,
                              "README anchor moved — update this test")
                start = self.text.index(anchor)
                window = self.text[start:start + 140]
                want = _range(metric, cells)
                self.assertRegex(window, re.escape(want) + "%",
                                 f"README region {window[:110]!r} "
                                 f"but the rollup says {want}%")


if __name__ == "__main__":
    unittest.main()
