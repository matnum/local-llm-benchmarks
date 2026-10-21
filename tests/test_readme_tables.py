"""Diff the README's result tables against the tracked rollups.

The README is the public face of the numbers; the rollups are the data. A table
cell that drifted (a stale value after a re-consolidation, a typo in a
transcription, a row pasted from the wrong backend) is invisible on review and
obvious to a reader who checks. Parse the markdown tables and compare every
column, with a tolerance that reflects display rounding.

Covered:
  * the 20-row throughput table  -> results/consolidated.csv
  * the FIN-bench table           -> results/finbench-consolidated.csv
"""
import csv
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO, "README.md")
ROLLUP = os.path.join(REPO, "results", "consolidated.csv")
FIN_ROLLUP = os.path.join(REPO, "results", "finbench-consolidated.csv")

THROUGHPUT_HEADER = "| Cell | Backend | pp2048 t/s | tg512 t/s | server tg t/s | TTFT p50 ms | tok/s/W | VRAM GiB |"
FIN_HEADER = "| Model | arc_c | belebele_fin | goldenswag | scandisent | sib200 | mean acc |"
TASKS = ("arc_c", "belebele_fin", "goldenswag", "scandisent", "sib200")

# Column index -> (rollup metric, tolerance). Tolerances are display rounding,
# not equality: the README prints 1-2 decimals.
THROUGHPUT_COLUMNS = (
    (2, "pp2048_bench_tps", 0.06),
    (3, "tg512_bench_tps", 0.006),
    (4, "tg_server_tps", 0.006),
    (5, "ttft_ms_p50", 0.06),
    (6, "tok_per_w", 0.006),
    (7, "vram_peak_gib", 0.006),
)


def _rows(text, header):
    """Markdown table rows following ``header`` -> list of stripped cell lists."""
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == header)
    except StopIteration:  # header reflowed or removed
        raise AssertionError(f"README table header not found: {header!r}")
    rows = []
    for line in lines[start + 1:]:
        if not line.startswith("|"):
            break
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):  # separator
            continue
        rows.append(cells)
    return rows


def _clean(cell):
    return cell.replace("`", "").strip()


@unittest.skipUnless(os.path.isfile(README) and os.path.isfile(ROLLUP),
                     "README or rollup missing")
class ReadmeTablesTest(unittest.TestCase):

    def setUp(self):
        with open(README, encoding="utf-8") as f:
            self.text = f.read()
        self.vals = {}
        with open(ROLLUP, newline="") as f:
            for r in csv.DictReader(f):
                self.vals.setdefault((r["cell"], r["backend"]), {})[r["metric"]] = float(r["value"])

    def test_throughput_table_matches_rollup(self):
        rows = _rows(self.text, THROUGHPUT_HEADER)
        self.assertEqual(len(rows), 20, "expected ten cells x two backends")
        for row in rows:
            cell, backend = _clean(row[0]), row[1]
            key = (cell, backend)
            with self.subTest(cell=cell, backend=backend):
                self.assertIn(key, self.vals, f"{cell}/{backend} has no rollup rows")
                for idx, metric, tol in THROUGHPUT_COLUMNS:
                    shown = float(row[idx])
                    self.assertAlmostEqual(
                        shown, self.vals[key][metric], delta=tol,
                        msg=f"README {cell}/{backend} {metric} = {shown}, "
                            f"rollup differs")

    def test_finbench_table_matches_rollup(self):
        fin = {}
        with open(FIN_ROLLUP, newline="") as f:
            for r in csv.DictReader(f):
                fin[(r["cell"], r["task"])] = (float(r["acc"]), float(r["acc_norm"]))

        rows = _rows(self.text, FIN_HEADER)
        self.assertEqual(len(rows), 5, "expected five FIN-bench models")
        for row in rows:
            cell = _clean(row[0])
            accs = []
            for task, cell_text in zip(TASKS, row[1:6]):
                acc_s, _, norm_s = cell_text.partition("/")
                acc, norm = float(acc_s), float(norm_s)
                with self.subTest(cell=cell, task=task):
                    self.assertIn((cell, task), fin, f"{cell}/{task} not in the rollup")
                    want_acc, want_norm = fin[(cell, task)]
                    self.assertAlmostEqual(acc, want_acc, delta=0.0006,
                                           msg=f"README acc {cell}/{task}")
                    self.assertAlmostEqual(norm, want_norm, delta=0.0006,
                                           msg=f"README acc_norm {cell}/{task}")
                accs.append(acc)
            with self.subTest(cell=cell, column="mean acc"):
                self.assertAlmostEqual(float(row[6]), sum(accs) / len(accs), delta=0.0006,
                                       msg=f"README mean acc for {cell}")


if __name__ == "__main__":
    unittest.main()
