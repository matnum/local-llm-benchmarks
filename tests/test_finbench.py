"""Tests for the FIN-bench-v2 driver: metric definitions, data plumbing, rollups.

The metric math and the two process_docs replicas are the parts that can be silently
wrong while still producing plausible-looking numbers, so they are pinned here against
the upstream definitions (lm_eval/api/task.py, <family>/*/*_utils.py). The prompt grid
itself is pinned in tests/test_finbench_upstream.py.
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import finbench  # noqa: E402


class MetricTest(unittest.TestCase):
    def test_acc_and_acc_norm_can_disagree(self):
        """acc uses the raw sum; acc_norm divides by the choice's CHARACTER length."""
        short, long_ = "aa", "bbbbbbbbbb"          # 2 vs 10 characters
        # The short choice wins on the raw sum but loses per character.
        lls = {"t::0": [-1.0, -2.0]}
        items = [("t::0", "p", [short, long_], 1)]
        res = finbench.evaluate("t", items, lls)
        self.assertEqual(res["items"][0]["pred"], 0)        # raw: -1.0 > -2.0
        self.assertEqual(res["items"][0]["pred_norm"], 1)   # per char: -0.5 < -0.2
        self.assertEqual(res["acc"], 0.0)
        self.assertEqual(res["acc_norm"], 1.0)

    def test_acc_norm_divisor_is_characters_not_tokens(self):
        """len() of the choice string, not its token count (a token count would tie here)."""
        choices = ["aa", "bbbb"]
        lls = {"t::0": [-2.0, -3.0]}
        res = finbench.evaluate("t", [("t::0", "p", choices, 1)], lls)
        # -2/2 = -1.0 vs -3/4 = -0.75 -> the 4-char choice wins after normalisation
        self.assertEqual(res["items"][0]["pred_norm"], 1)

    def test_ties_go_to_the_first_choice(self):
        res = finbench.evaluate("t", [("t::0", "p", ["a", "b"], 1)], {"t::0": [-1.0, -1.0]})
        self.assertEqual(res["items"][0]["pred"], 0)

    def test_context_overflow_items_are_excluded_and_counted(self):
        items = [("t::0", "p", ["a", "b"], 0), ("t::1", "p", ["a", "b"], 0)]
        lls = {"t::0": [-1.0, -2.0], "t::1": [-1e9, -1e9]}
        ntoks = {"t::0": [1, 1], "t::1": [-1, -1]}
        res = finbench.evaluate("t", items, lls, ntoks)
        self.assertEqual(res["n"], 1)
        self.assertEqual(res["context_overflow"], 1)
        self.assertEqual(res["acc"], 1.0)

    def test_unusable_gold_is_counted_not_crashed(self):
        res = finbench.evaluate("t", [("t::0", "p", ["a", "b"], -100)], {"t::0": [-1.0, -2.0]})
        self.assertEqual(res["gold_index_errors"], 1)
        self.assertEqual(res["n"], 1)


class TransformTest(unittest.TestCase):
    """cf/p0 prompts render to the upstream text (templates vendored, see
    tests/test_finbench_upstream.py for the whole 5x2x5 grid)."""

    def test_arc_c_prompt_matches_upstream_template(self):
        doc = {"question": "K?", "choices": {"text": ["a", "b"], "label": ["A", "B"]}, "answerKey": "B"}
        prompt, choices, gold = finbench.item_for("arc_c", doc)
        self.assertEqual(prompt, "Vastaus kysymykseen K?, on:")
        self.assertEqual(choices, ["a", "b"])
        self.assertEqual(gold, 1)

    def test_scandisent_maps_labels_to_finnish_words(self):
        prompt, choices, gold = finbench.item_for("scandisent", {"text": "T", "label": "positive"})
        self.assertEqual(choices, ["positiivinen", "negatiivinen"])
        self.assertEqual(gold, 0)                      # string gold -> index in choices
        self.assertEqual(prompt, "Arvostelijan teksti: T\nTunne:")

    def test_goldenswag_process_docs_replica(self):
        doc = {"activity_label": "Ruoka", "ctx_a": "Alku [title] jatkoa", "ctx_b": "loppu",
               "endings": ["eka  jatko", "toka"], "label": "1"}
        prompt, choices, gold = finbench.item_for("goldenswag", doc)
        self.assertTrue(prompt.startswith("Kerro loogisin jatko"))
        self.assertNotIn("[title]", prompt)            # replaced with ". "
        self.assertIn(". jatkoa Loppu", prompt)        # [title] -> ". ", ctx_b capitalised
        self.assertEqual(choices, ["eka jatko", "toka"])   # double space collapsed
        self.assertEqual(gold, 1)

    def test_sib200_uses_dataset_choices_and_index(self):
        doc = {"text": "U", "choices": ["x", "y", "z"], "answer_idx": 2}
        prompt, choices, gold = finbench.item_for("sib200", doc)
        self.assertEqual((choices, gold), (["x", "y", "z"], 2))
        self.assertTrue(prompt.endswith("Aihe:"))

    def test_belebele_prompt_and_gold_index(self):
        doc = {"flores_passage": "P", "question": "Q", "mc_answer1": "a", "mc_answer2": "b",
               "mc_answer3": "c", "mc_answer4": "d", "correct_answer_num": "3"}
        prompt, choices, gold = finbench.item_for("belebele_fin", doc)
        self.assertEqual((choices, gold), (["a", "b", "c", "d"], 2))
        self.assertIn("Tässä on teksti: P", prompt)

    def test_mcf_changes_the_choice_set_for_belebele_only(self):
        """mcf is not just a prompt swap: belebele scores the letters, arc_c the text."""
        bele = {"flores_passage": "P", "question": "Q", "mc_answer1": "a", "mc_answer2": "b",
                "mc_answer3": "c", "mc_answer4": "d", "correct_answer_num": "3"}
        self.assertEqual(finbench.item_for("belebele_fin", bele, "mcf", "p0")[1],
                         ["A", "B", "C", "D"])
        arc = {"question": "K?", "choices": {"text": ["a", "b"], "label": ["A", "B"]},
               "answerKey": "B"}
        self.assertEqual(finbench.item_for("arc_c", arc, "mcf", "p0")[1], ["a", "b"])

    def test_every_grid_cell_resolves_to_an_upstream_task(self):
        for family in finbench.FAMILIES:
            for formulation in finbench.FORMULATIONS:
                for variant in finbench.VARIANTS:
                    rel, task, _ = finbench._template(family, formulation, variant)
                    self.assertTrue(rel.startswith(f"{family}/{formulation}/"), rel)
                    self.assertTrue(rel.endswith(f"_{variant}.yaml"), rel)
                    self.assertTrue(task, f"no task name for {rel}")


class RunModelTest(unittest.TestCase):
    """run_model() wiring: per-task skip counts and de-identified model provenance."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.results = os.path.join(self._tmp.name, "results")
        self.gguf = os.path.join(self._tmp.name, "models",
                                 "Llama-Poro-2-8B-Instruct.Q8_0.gguf")
        os.makedirs(os.path.dirname(self.gguf), exist_ok=True)
        with open(self.gguf, "w"):
            pass

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, skips):
        with mock.patch.object(finbench, "RESULTS", self.results), \
             mock.patch.object(finbench, "fetch_rows",
                               side_effect=lambda task, data_dir, limit=None, quiet=False: []), \
             mock.patch.object(finbench, "build_items",
                               side_effect=lambda task, rows, limit=None, **kw: ([], skips[task])), \
             mock.patch.object(finbench, "score", return_value=({}, {})):
            return finbench.run_model("poro2-8b-q8_0", self.gguf, "t",
                                      ["arc_c", "sib200"], 0, "d", 99, 4096, 16, False)

    def test_skipped_is_reported_per_task(self):
        """Each task's summary carries ITS own skips, not the cumulative total."""
        summaries = self._run({"arc_c": 3, "sib200": 0})
        self.assertEqual(summaries["arc_c"]["skipped"], 3)
        self.assertEqual(summaries["sib200"]["skipped"], 0)

    def test_summary_model_is_basename_not_absolute_path(self):
        """summary.json must not leak the machine's /home/<user> path."""
        self._run({"arc_c": 0, "sib200": 0})
        with open(os.path.join(self.results, "finbench-t", "poro2-8b-q8_0",
                               "summary.json")) as f:
            s = json.load(f)
        self.assertEqual(s["model"], "Llama-Poro-2-8B-Instruct.Q8_0.gguf")
        self.assertNotIn("/home/", s["model"])
        self.assertNotIn("/halo", s["model"])

    def test_summary_stamps_formulation_variant_and_upstream_commit(self):
        with mock.patch.object(finbench, "RESULTS", self.results), \
             mock.patch.object(finbench, "fetch_rows",
                               side_effect=lambda task, data_dir, limit=None, quiet=False: []), \
             mock.patch.object(finbench, "build_items",
                               side_effect=lambda task, rows, limit=None, **kw: ([], 0)), \
             mock.patch.object(finbench, "score", return_value=({}, {})):
            finbench.run_model("poro2-8b-q8_0", self.gguf, "t", ["arc_c"], 0, "d",
                               99, 4096, 16, False, formulation="mcf", variant="p3")
        s = json.load(open(os.path.join(self.results, "finbench-t", "poro2-8b-q8_0",
                                        "summary.json")))
        self.assertEqual((s["formulation"], s["prompt_variant"]), ("mcf", "p3"))
        self.assertEqual(s["upstream_commit"], finbench.UPSTREAM_COMMIT)


class ConsolidateTest(unittest.TestCase):
    """Rollups: pilots never publish, and a second pair adds rows, not replaces."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.results = os.path.join(self._tmp.name, "results")
        self.run_dir = os.path.join(self.results, "finbench-fi-test", "cell")
        os.makedirs(self.run_dir, exist_ok=True)
        self.tracked_csv = os.path.join(self.results, "finbench-consolidated.csv")
        self.tracked_md = os.path.join(self.results, "finbench-table.md")
        for path in (self.tracked_csv, self.tracked_md):
            with open(path, "w") as f:
                f.write("sentinel\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _write_summary(self, run_id, formulation, variant, pilot, cell="cell"):
        run_dir = os.path.join(self.results, f"finbench-{run_id}", cell)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump({"cell": cell, "model": "m.gguf", "run_id": run_id,
                       "formulation": formulation, "prompt_variant": variant,
                       "pilot_limit": pilot,
                       "tasks": {"arc_c": {"n": 100 if not pilot else pilot,
                                           "acc": 0.4, "acc_norm": 0.42, "skipped": 0}}}, f)

    def test_pilot_never_reaches_the_tracked_rollup(self):
        self._write_summary("fi-test", "mcf", "p2", pilot=150)
        with mock.patch.object(finbench, "RESULTS", self.results):
            finbench.consolidate("fi-test")
        self.assertEqual(open(self.tracked_csv).read(), "sentinel\n")
        self.assertEqual(open(self.tracked_md).read(), "sentinel\n")
        self.assertEqual(sorted(os.listdir(self.run_dir)), ["summary.json"],
                         "nothing should be written for a pilot-only tree")

    def test_full_runs_accumulate_and_combo_columns_are_written(self):
        self._write_summary("fi-mcf", "mcf", "p2", pilot=0, cell="cell_a")
        self._write_summary("fi-cf", "cf", "p0", pilot=0, cell="cell_b")
        with mock.patch.object(finbench, "RESULTS", self.results):
            finbench.consolidate()
        rows = list(csv.DictReader(open(self.tracked_csv)))
        combos = {(r["formulation"], r["variant"]) for r in rows}
        self.assertEqual(combos, {("mcf", "p2"), ("cf", "p0")})
        md = open(self.tracked_md).read()
        self.assertIn("## `cf` / `p0`", md)
        self.assertIn("## `mcf` / `p2`", md)
        # the pilot cell never appears
        self.assertNotIn("| cell |", md)


if __name__ == "__main__":
    unittest.main()
