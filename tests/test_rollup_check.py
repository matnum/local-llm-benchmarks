"""The local drift guard: `--check` must catch what a late --consolidate would change.

Review I5 (2026-10-24): the README and the workflow comment claimed "the suite
rebuilds the rollups from the per-run data to catch drift". It cannot — raw run
dirs are gitignored, so CI never has them — and the only real guard was the
README-vs-CSV value diff, which fires *after* `--consolidate` has already
invalidated the README (that is how B1 was reachable). `run_sweep.check_rollup()`
is the operator-side replacement: local-only, read-only, and it names the rows.

These tests build a throwaway repo with synthetic run dirs, so they exercise the
guard itself rather than this machine's results.
"""
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import run_sweep  # noqa: E402
from tests.test_consolidate import _write_run  # noqa: E402


class CheckRollupTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        os.makedirs(os.path.join(self.repo, "results"), exist_ok=True)
        _write_run(self.repo, "run-A", "vulkan", "cellA")
        _write_run(self.repo, "run-A", "rocm", "cellA")
        run_sweep.consolidate(repo=self.repo)

    def tearDown(self):
        self._tmp.cleanup()

    def _check(self):
        return run_sweep.check_rollup(repo=self.repo)

    def _csv_path(self):
        return os.path.join(self.repo, "results", "consolidated.csv")

    def _md_path(self):
        return os.path.join(self.repo, "results", "table.md")

    def test_fresh_rollup_is_clean(self):
        problems, n_runs = self._check()
        self.assertEqual(problems, [])
        self.assertEqual(n_runs, 1, "one run dir holds both backends")

    def test_a_new_run_dir_is_reported(self):
        """The regression B1 needed: a run landed but the rollup was not rebuilt."""
        _write_run(self.repo, "run-B", "vulkan", "cellA")
        problems, _ = self._check()
        self.assertTrue(problems, "a run dir the rollup omits must fail the check")
        self.assertTrue(any("run-B" in p and "new row" in p for p in problems),
                        f"the message must name the run: {problems}")

    def test_an_edited_value_is_reported(self):
        with open(self._csv_path(), newline="") as f:
            rows = list(csv.DictReader(f))
            fields = list(rows[0].keys())
        rows[0]["value"] = "999.0"
        with open(self._csv_path(), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        problems, _ = self._check()
        self.assertTrue(any("changed" in p and "999.0" in p for p in problems),
                        f"a hand-edited figure must fail the check: {problems}")

    def test_an_edited_prose_line_is_reported(self):
        """table.md's first block line carries both the stamp and the prose."""
        with open(self._md_path()) as f:
            text = f.read()
        with open(self._md_path(), "w") as f:
            f.write(text.replace("cold TTFT erases", "cold TTFT wraps"))
        problems, _ = self._check()
        self.assertTrue(any("table.md line 3" in p for p in problems),
                        f"the prose must be compared, not skipped: {problems}")

    def test_the_harness_stamp_alone_is_not_drift(self):
        """Every unrelated commit moves the stamp; that must not red the guard."""
        with open(self._md_path()) as f:
            text = f.read()
        before = text.split("`")[1]
        with open(self._md_path(), "w") as f:
            f.write(text.replace(before, "0" * 12, 1))
        problems, _ = self._check()
        self.assertEqual(problems, [], "only the stamp changed, so nothing drifted")

    def test_no_run_dirs_means_nothing_to_check(self):
        """A CI checkout has no run dirs; the guard must say so, not fail."""
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        os.makedirs(os.path.join(empty.name, "results"), exist_ok=True)
        problems, n_runs = run_sweep.check_rollup(repo=empty.name)
        self.assertEqual((problems, n_runs), ([], 0))


if __name__ == "__main__":
    unittest.main()
