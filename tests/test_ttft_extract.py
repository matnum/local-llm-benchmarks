"""The tracked per-request extract must agree with the rollup and with the docs.

`docs/RUNNING.md` §3c/§7 publish per-request TTFT values (the whole argument for
the cold protocol is that the *spread* collapsed), but raw run directories are
gitignored — without an extract those numbers are unverifiable by a reader, which
is the defect review 2026-10-24 raised. `results/ttft-per-request.csv` is that
extract: the published cold pass (`run-20261024-0910` Vulkan, `-0930` ROCm) *and*
the warm column it replaced (`run-20261023-1140`/`-1159`), so both sides of the
protocol comparison are checkable.

Three things are pinned here:
  * the extract describes exactly the published runs, with integrity markers
    (cold rows must show a non-empty erase; warm rows must not claim one);
  * the rollup's `ttft_ms`/`ttft_ms_p50` are re-derived from the extract, so the
    extract and the published column cannot drift apart;
  * §3c's pilot table is re-read from the markdown and checked against the extract,
    so a number cannot be retyped from memory (the B2 failure mode).
"""
import csv
import os
import re
import statistics
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT = os.path.join(REPO, "results", "ttft-per-request.csv")
ROLLUP = os.path.join(REPO, "results", "consolidated.csv")
RUNNING = os.path.join(REPO, "docs", "RUNNING.md")

COLD_RUNS = {("run-20261024-0910", "vulkan"), ("run-20261024-0930", "rocm")}
WARM_RUNS = {("run-20261023-1140", "vulkan"), ("run-20261023-1159", "rocm")}
RUNS = COLD_RUNS | WARM_RUNS

# "| `qwen3-4b-q8_0` | vulkan | 22.5 / 80.0 / 27.6 | 57.5 | 55.3 / 53.4 / 55.6 | 2.2 |"
SECTION_3C_ROW = re.compile(
    r"^\| `(?P<cell>[^`]+)` \| (?P<backend>vulkan|rocm) \| "
    r"(?P<warm>[-\d. /]+) \| (?P<warm_spread>[\d.]+) \| "
    r"(?P<cold>[-\d. /]+) \| (?P<cold_spread>[\d.]+) \|$", re.MULTILINE)


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@unittest.skipUnless(os.path.isfile(ROLLUP), "no tracked rollup")
class PerRequestExtractTest(unittest.TestCase):

    def setUp(self):
        # The extract is tracked, so a checkout always has it. Do not *skip* when it
        # is missing: deleting it would otherwise make this whole guard disappear
        # silently (and take the §3c markdown check with it).
        self.assertTrue(os.path.isfile(EXTRACT),
                        f"{EXTRACT} is tracked — restore it rather than dropping the "
                        f"checkable per-request data")
        self.assertTrue(os.path.isfile(RUNNING), "docs/RUNNING.md is missing")
        self.rows = _read(EXTRACT)
        self.rollup = {}
        for r in _read(ROLLUP):
            self.rollup[(r["run_id"], r["backend"], r["cell"], r["metric"])] = float(r["value"])

    def groups(self, protocol):
        """(run_id, backend, cell) -> per-request ttft in measurement order."""
        out = {}
        for r in self.rows:
            if r["protocol"] != protocol:
                continue
            out.setdefault((r["run_id"], r["backend"], r["cell"]), []).append(
                float(r["ttft_ms"]))
        return out

    def test_extract_covers_both_published_columns(self):
        pairs = {(r["run_id"], r["backend"]) for r in self.rows}
        self.assertEqual(pairs, RUNS, "the extract must describe exactly the published runs")
        self.assertEqual({r["protocol"] for r in self.rows}, {"cold", "warm"})
        self.assertEqual(len(self.rows), 120,
                         "4 runs x 10 cells x 3 requests (2 cold, 2 warm)")

    def test_erased_tokens_mark_the_protocol(self):
        """Cold rows must show the erase dropped a populated cache; warm rows cannot."""
        for r in self.rows:
            with self.subTest(run=r["run_id"], cell=r["cell"], backend=r["backend"],
                              request=r["request"]):
                if r["protocol"] == "cold":
                    self.assertGreater(int(r["erased_tokens"]), 0,
                                       "a cold run whose erase dropped nothing is a warm run")
                else:
                    self.assertEqual(r["erased_tokens"], "",
                                     "a warm request never touches the slots API")

    def test_extract_medians_reproduce_the_published_column(self):
        for protocol in ("cold", "warm"):
            for (run_id, backend, cell), vals in self.groups(protocol).items():
                with self.subTest(protocol=protocol, run=run_id, backend=backend, cell=cell):
                    self.assertEqual(len(vals), 3)
                    self.assertAlmostEqual(statistics.median(vals),
                                           self.rollup[(run_id, backend, cell, "ttft_ms_p50")],
                                           delta=0.15, msg="extract median != published p50")
                    self.assertAlmostEqual(statistics.fmean(vals),
                                           self.rollup[(run_id, backend, cell, "ttft_ms")],
                                           delta=0.15, msg="extract mean != published mean")

    def test_published_rows_cite_these_runs(self):
        """Every published streaming row must come from a run the extract documents."""
        for (run_id, backend, cell, metric) in self.rollup:
            if metric != "ttft_ms_p50" or (run_id, backend) not in RUNS:
                continue
            with self.subTest(cell=cell, backend=backend):
                self.assertIn(cell, {r["cell"] for r in self.rows})

    def test_published_warm_to_cold_delta_range_reproduces(self):
        """README/§3c quote +20..+493% (Vulkan) and +58..+397% (ROCm) on p50."""
        for backend, lo, hi in (("vulkan", 20, 493), ("rocm", 58, 397)):
            with self.subTest(backend=backend):
                cold = {k[2]: v for k, v in self.groups("cold").items() if k[1] == backend}
                warm = {k[2]: v for k, v in self.groups("warm").items() if k[1] == backend}
                self.assertEqual(set(cold), set(warm))
                deltas = [round((statistics.median(cold[c]) / statistics.median(warm[c])
                                 - 1) * 100) for c in cold]
                self.assertEqual((min(deltas), max(deltas)), (lo, hi),
                                 f"{backend} warm->cold p50 deltas are {sorted(deltas)}, "
                                 f"but the docs publish {lo}..{hi}%")

    def test_cold_mean_and_p50_agree_within_the_stated_ratio(self):
        """results/table.md prose: cold mean/p50 within a few percent, worst 1.04x."""
        ratios = {}
        for protocol in ("cold", "warm"):
            for key, vals in self.groups(protocol).items():
                ratios[(protocol, key)] = statistics.fmean(vals) / statistics.median(vals)
        cold_worst = max(r for (p, _), r in ratios.items() if p == "cold")
        warm_worst = max(r for (p, _), r in ratios.items() if p == "warm")
        self.assertLessEqual(round(cold_worst, 2), 1.04,
                             f"cold mean/p50 is {cold_worst:.3f}x, but table.md says 1.04x")
        self.assertGreater(warm_worst, 1.5,
                          "the warm column is supposed to be the spread-out one; if it is "
                          "no longer, the prose comparing the protocols needs revisiting")

    def test_section_3c_table_matches_the_extract(self):
        """B2: §3c published warm values that existed in no artifact on disk."""
        with open(RUNNING) as f:
            table = SECTION_3C_ROW.findall(f.read())
        self.assertTrue(table, "§3c's per-request table is gone from docs/RUNNING.md")
        sources = {"cold": self.groups("cold"), "warm": self.groups("warm")}
        for cell, backend, warm_txt, warm_spread, cold_txt, cold_spread in table:
            cited = {"warm": (warm_txt, warm_spread), "cold": (cold_txt, cold_spread)}
            for protocol in ("warm", "cold"):
                txt, spread_txt = cited[protocol]
                key = next((k for k in sources[protocol]
                            if k[1] == backend and k[2] == cell), None)
                self.assertIsNotNone(
                    key, f"§3c cites {cell}/{backend} {protocol} but the extract has no "
                         f"such run — publish the run it came from")
                vals = sources[protocol][key]
                values = [float(x) for x in txt.split(" / ")]
                with self.subTest(cell=cell, backend=backend, protocol=protocol):
                    self.assertEqual(len(values), len(vals))
                    for got, want in zip(values, vals):
                        self.assertAlmostEqual(got, want, delta=0.05,
                                               msg=f"§3c says {txt}, extract says {vals}")
                    self.assertAlmostEqual(float(spread_txt), max(vals) - min(vals),
                                           delta=0.15, msg="cited spread != extract spread")


if __name__ == "__main__":
    unittest.main()
