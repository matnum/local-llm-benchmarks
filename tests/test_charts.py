"""Tests for make_charts: every measured cell must reach the figures.

Regression guard for the staleness class this repo already hit once — MODEL_ORDER
was a hardcoded five-cell list, so the 2026-10-03 next-round cells were silently
absent from every chart while the README claimed to show the results. Order and
labels are now derived from the rollup + matrix; these tests pin that.
"""

import collections
import csv
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

try:
    import make_charts
except ImportError as _exc:  # matplotlib / PyYAML are optional extras
    make_charts = None
    _IMPORT_ERROR = _exc

BACKENDS = ("vulkan", "rocm")
EXPECTED_CHARTS = ["backend_delta.png", "generation_tps.png",
                   "prefill_tps.png", "tokens_per_watt.png"]


@unittest.skipIf(make_charts is None, "matplotlib/PyYAML not installed")
class ChartDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(make_charts.CSV, newline="") as f:
            cls.rollup = list(csv.DictReader(f))
        cls.vals = make_charts.load_latest()
        cls.meta = make_charts.load_matrix()
        cls.cells = make_charts.comparable_cells(cls.vals)

    def _cells_with_both_backends(self):
        by_cell = collections.defaultdict(set)
        for r in self.rollup:
            by_cell[r["cell"]].add(r["backend"])
        return {c for c, bs in by_cell.items() if set(BACKENDS) <= bs}

    def test_every_two_backend_cell_is_plotted(self):
        """The staleness bug: a cell in the rollup that no chart knows about."""
        self.assertEqual(set(self.cells), self._cells_with_both_backends())
        self.assertGreaterEqual(len(self.cells), 9)

    def test_cells_are_ordered_by_model_size(self):
        sizes = [self.meta[c]["size_gib"] for c in self.cells]
        self.assertEqual(sizes, sorted(sizes))

    def test_labels_unique_and_informative(self):
        labels = [make_charts.row_label(c, self.meta) for c in self.cells]
        self.assertEqual(len(labels), len(set(labels)), "duplicate axis labels")
        for label in labels:
            self.assertIn("GiB", label)

    def test_every_measured_cell_has_matrix_metadata(self):
        """Labels and the ceiling marker are built from matrix.yaml."""
        measured = {r["cell"] for r in self.rollup}
        self.assertFalse(measured - set(self.meta),
                         f"measured cells missing from matrix.yaml: {measured - set(self.meta)}")

    def test_each_cell_has_the_plotted_metrics(self):
        for cell in self.cells:
            for backend in BACKENDS:
                for metric in ("tg512_bench_tps", "pp2048_bench_tps", "tok_per_w"):
                    self.assertIn(metric, self.vals[(cell, backend)],
                                  f"{cell}/{backend} lacks {metric}")

    def test_main_writes_all_charts_as_real_pngs(self):
        """End-to-end, into a temp dir so the tracked PNGs are never touched."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(make_charts, "OUT", tmp):
                make_charts.main()
            self.assertEqual(sorted(os.listdir(tmp)), EXPECTED_CHARTS)
            for name in EXPECTED_CHARTS:
                with open(os.path.join(tmp, name), "rb") as f:
                    head = f.read(24)
                self.assertEqual(head[:8], b"\x89PNG\r\n\x1a\n", f"{name} is not a PNG")
                width, height = struct.unpack(">II", head[16:24])
                self.assertGreater(width, 600, f"{name} suspiciously narrow")
                self.assertGreater(height, 400, f"{name} suspiciously short")


if __name__ == "__main__":
    unittest.main()
