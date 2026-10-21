"""Pin the hand-typed claims in README.md to the tracked rollups.

The independent review's finding I-2 was a wrong headline percentage (18.3%,
divided by ROCm instead of Vulkan) sitting a few lines above a machine-generated
table, and it survived a previous "correction" because nothing checked it. The
numbers are still written by hand, so this recomputes them from
results/consolidated.csv using the README's own convention — the backend gap as
a percentage of Vulkan, the baseline arm — and fails when prose and data
disagree.

The throughput and FIN-bench tables are diffed against their rollups by
tests/test_readme_tables.py; this file covers the prose.
"""
import csv
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLUP = os.path.join(REPO, "results", "consolidated.csv")
README = os.path.join(REPO, "README.md")

# The 2026-10-03 next-round five, called out pairwise in the README.
SECOND_FIVE = {
    "qwen3-4b-q8_0", "poro2-8b-q8_0", "eurollm-9b-q8_0",
    "qwen3-32b-q5_k_m", "poro2-70b-q4_k_m",
}
# Dense 27-32B cells, the baseline for the MoE-advantage claim.
DENSE_27_32 = {"qwen3.8-27b-q5_k_xl", "qwen3.8-27b-q6_k", "qwen3-32b-q5_k_m"}
MOE_CELLS = {"qwen3-coder-30b-a3b-q6_k", "qwen3.5-35b-a3b-q6_k", "gpt-oss-120b-mxfp4"}


def _both_backends(metric):
    cells = {}
    with open(ROLLUP, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["metric"] == metric:
                cells.setdefault(row["cell"], {})[row["backend"]] = float(row["value"])
    return {c: v for c, v in cells.items() if {"vulkan", "rocm"} <= set(v)}


def _range(metric, cells=None):
    """The backend gap as a percentage of Vulkan, formatted as the README writes it."""
    pct = sorted(abs((v["vulkan"] - v["rocm"]) / v["vulkan"] * 100)
                 for c, v in _both_backends(metric).items()
                 if cells is None or c in cells)
    return f"{pct[0]:.1f}\u2013{pct[-1]:.1f}"


def _ratio_range(metric, numerators, denominators):
    """Whole-number ratio range, e.g. '5-8' for MoE versus dense cells."""
    vals = _both_backends(metric)
    ratios = [vals[n][b] / vals[d][b]
              for n in sorted(numerators) for d in sorted(denominators)
              for b in ("vulkan", "rocm")]
    return f"{round(min(ratios))}\u2013{round(max(ratios))}"


@unittest.skipUnless(os.path.isfile(ROLLUP) and os.path.isfile(README),
                     "rollup or README missing")
class ReadmeClaimsTest(unittest.TestCase):

    def setUp(self):
        # Collapse whitespace so a prose reflow cannot break the anchors.
        with open(README, encoding="utf-8") as f:
            self.text = " ".join(f.read().split())

    def _assert_window(self, anchor, want):
        self.assertIn(anchor, self.text, "README anchor moved — update this test")
        start = self.text.index(anchor)
        window = self.text[start:start + 140]
        self.assertRegex(window, re.escape(want),
                         f"README region {window[:110]!r} but the rollup says {want!r}")

    def test_prose_ranges_match_rollup(self):
        cases = [
            # All ten cells (the "At a glance" block and again under Results).
            ("Vulkan wins generation in 10 of 10 (ROCm trails by",
             _range("tg512_bench_tps") + "%"),
            ("ROCm wins prefill in 10 of 10 (by",
             _range("pp2048_bench_tps") + "%"),
            # The 2026-10-03 round of five, called out pairwise.
            ("Vulkan is faster on `tg512` in 5 of 5 (ROCm trails by",
             _range("tg512_bench_tps", SECOND_FIVE) + "%"),
            ("ROCm is faster on `pp2048` in 5 of 5 (by",
             _range("pp2048_bench_tps", SECOND_FIVE) + "%"),
        ]
        for anchor, want in cases:
            with self.subTest(anchor=anchor):
                self._assert_window(anchor, want)

    def test_moe_advantage_matches_rollup(self):
        """The '5-8x faster / 5-8x more efficient' claim, recomputed."""
        gen = _ratio_range("tg512_bench_tps", MOE_CELLS, DENSE_27_32)
        eff = _ratio_range("tok_per_w", MOE_CELLS, DENSE_27_32)
        self._assert_window("MoE cells are",
                            f"{gen}\u00d7 faster at generation and {eff}\u00d7 more efficient")


if __name__ == "__main__":
    unittest.main()
