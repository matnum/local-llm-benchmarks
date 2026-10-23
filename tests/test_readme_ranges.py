"""Pin the hand-typed claims in README.md to the tracked rollups.

The independent review's finding I-2 was a wrong headline percentage (18.3%,
divided by ROCm instead of Vulkan) sitting a few lines above a machine-generated
table, and it survived a previous "correction" because nothing checked it. The
numbers are still written by hand, so this recomputes them from
results/consolidated.csv using the README's own convention — the backend gap as
a percentage of Vulkan, the baseline arm — and fails when prose and data
disagree.

Two conventions the re-measure forced into the open, both pinned here:

  * A backend's lead is counted only over the cells it actually leads; a cell
    whose gap is inside the noise floor is a tie, not a member of either set.
  * Ranges are computed over the decisive cells of the leading backend. Mixing a
    tie into a "by X-Y%" range produced a headline range whose floor was a cell
    that had not been won at all.

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

# The README's own noise floor for three repeats: "Differences under ~2% are
# within noise for this setup" (Limitations). A gap below this is a tie.
TIE_PCT = 2.0

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


def _signed(metric):
    """Backend gap per cell as a percentage of Vulkan, signed (Vulkan faster > 0)."""
    return {c: (v["vulkan"] - v["rocm"]) / v["vulkan"] * 100
            for c, v in _both_backends(metric).items()}


def _lead(metric, backend, cells=None):
    """(all cells this backend leads, those whose lead clears the noise floor)."""
    sign = 1 if backend == "vulkan" else -1
    gaps = _signed(metric)
    lead = [c for c, g in gaps.items()
            if sign * g > 0 and (cells is None or c in cells)]
    return lead, [c for c in lead if abs(gaps[c]) >= TIE_PCT]


def _range(metric, backend, cells=None):
    """Gap range over the decisive wins, as the README writes it: '2.1-18.0'."""
    gaps = _signed(metric)
    _, decisive = _lead(metric, backend, cells)
    vals = sorted(abs(gaps[c]) for c in decisive)
    return f"{vals[0]:.1f}\u2013{vals[-1]:.1f}"


def _ties(n):
    return "tie" if n == 1 else "ties"


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
        window = self.text[start:start + 160]
        self.assertRegex(window, re.escape(want),
                         f"README region {window[:110]!r} but the rollup says {want!r}")

    def test_prose_counts_ranges_and_ties_match_rollup(self):
        total = len(_both_backends("tg512_bench_tps"))
        _, gen_dec = _lead("tg512_bench_tps", "vulkan")
        pp_lead, pp_dec = _lead("pp2048_bench_tps", "rocm")
        # A tie is a cell whose gap is inside the noise floor, whichever way it
        # leans — not "cells nobody leads": the 35B prefill cell leans Vulkan by
        # 0.7%, below the floor, so it counts as a tie rather than a ROCm win.
        gen_ties = total - len(gen_dec)
        pp_ties = total - len(pp_dec)
        _, gen_dec5 = _lead("tg512_bench_tps", "vulkan", SECOND_FIVE)
        _, pp_dec5 = _lead("pp2048_bench_tps", "rocm", SECOND_FIVE)

        cases = [
            # At a glance.
            ("Vulkan leads generation on ",
             f"{total} of {total} cells — decisively on {len(gen_dec)}, "
             f"by {_range('tg512_bench_tps', 'vulkan')}%"),
            ("ROCm leads prefill on ",
             f"{len(pp_lead)} of {total}, by {_range('pp2048_bench_tps', 'rocm')}%"),
            # Results: all ten cells.
            ("Vulkan leads `tg512` on ",
             f"{total} of {total} cells — decisively on {len(gen_dec)} "
             f"(by {_range('tg512_bench_tps', 'vulkan')}%), "
             f"with {gen_ties} {_ties(gen_ties)}"),
            ("and ROCm leads `pp2048` on ",
             f"{len(pp_lead)} of {total} "
             f"(by {_range('pp2048_bench_tps', 'rocm')}%), "
             f"with {pp_ties} {_ties(pp_ties)}"),
            # The 2026-10-03 round of five, called out pairwise.
            ("reproduces the split: Vulkan leads `tg512` on",
             f"all five (decisively on {len(gen_dec5)}, "
             f"by {_range('tg512_bench_tps', 'vulkan', SECOND_FIVE)}%)"),
            ("`pp2048` on all five (by",
             f"{_range('pp2048_bench_tps', 'rocm', SECOND_FIVE)}%)"),
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
