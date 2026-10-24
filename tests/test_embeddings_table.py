"""The published embeddings table and the README's copy of it must not drift.

Same rule as the other axes (tests/test_readme_tables.py, test_rollup_check.py):
the generated artifact is the source of truth, the README reproduces it, and a
freshness check runs only where the gitignored raw runs exist — on CI there is
nothing to compare, so it says so and passes instead of pretending.
"""

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import embed_eval  # noqa: E402

TABLE = os.path.join(REPO, "results", "embeddings-table.md")
README = os.path.join(REPO, "README.md")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class PublishedTableTest(unittest.TestCase):
    def test_table_is_tracked_and_has_the_expected_rows(self):
        table = _read(TABLE)
        rows = [line for line in table.splitlines() if line.startswith("| ") and "Model" not in line]
        self.assertEqual(len(rows), 4, "two models x two backends")
        for row in rows:
            self.assertEqual(row.count("|"), 10, row)  # 9 columns

    def test_readme_reproduces_the_table_verbatim(self):
        table = _read(TABLE).strip()
        self.assertIn(table, _read(README), "README's table drifted from results/embeddings-table.md")

    def test_readme_states_the_headline_counts_the_table_shows(self):
        table = _read(TABLE)
        readme = _read(README)
        # hit@1 0.867 is 26 of 30 questions; 0.600 is 18.
        self.assertIn("26 of 30", readme)
        self.assertIn("18", readme)
        for model in ("bge-m3", "nomic-embed-text-v1.5"):
            self.assertIn(f"| {model} |", table)

    def test_freshness_against_local_runs(self):
        """With raw run dirs present, the tracked table must be their render."""
        runs = embed_eval.load_summaries()
        if not runs:
            self.skipTest(
                "no results/embed-*/summary.json on this machine (gitignored) — "
                "freshness cannot be checked on a fresh clone or CI"
            )
        self.assertEqual(_read(TABLE).strip(), embed_eval.render_table(runs).strip())

    def test_paired_comparison_in_the_readme_matches_the_runs(self):
        runs = embed_eval.load_summaries()
        if not runs:
            self.skipTest("no raw runs on this machine")
        latest = {(s["model"], s["backend"]): s for s in runs}
        if ("bge-m3", "vulkan") not in latest or ("nomic-embed-text-v1.5", "vulkan") not in latest:
            self.skipTest("the two compared runs are not both present")
        wins = embed_eval.paired_counts(latest[("bge-m3", "vulkan")],
                                        latest[("nomic-embed-text-v1.5", "vulkan")])
        self.assertIn(
            f"{wins['left']} queries better, {wins['right']} worse, {wins['tie']} equal",
            _read(README),
        )


if __name__ == "__main__":
    unittest.main()
