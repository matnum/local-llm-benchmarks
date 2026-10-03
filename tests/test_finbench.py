"""Tests for the FIN-bench-v2 driver: metric definitions, transforms, exclusions.

The metric math and the two process_docs replicas are the parts that can be silently
wrong while still producing plausible-looking numbers, so they are pinned here against
the upstream definitions (lm_eval/api/task.py, <family>/cf/*_utils.py).
"""

import os
import sys
import unittest

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
    def test_arc_c_prompt_matches_upstream_template(self):
        doc = {"question": "K?", "choices": {"text": ["a", "b"], "label": ["A", "B"]}, "answerKey": "B"}
        prompt, choices, gold = finbench.TASKS["arc_c"]["transform"](doc)
        self.assertEqual(prompt, "Vastaus kysymykseen K?, on:")
        self.assertEqual(choices, ["a", "b"])
        self.assertEqual(gold, 1)

    def test_scandisent_maps_labels_to_finnish_words(self):
        prompt, choices, gold = finbench.TASKS["scandisent"]["transform"]({"text": "T", "label": "positive"})
        self.assertEqual(choices, ["positiivinen", "negatiivinen"])
        self.assertEqual(gold, 0)                      # string gold -> index in choices
        self.assertEqual(prompt, "Arvostelijan teksti: T\nTunne:")

    def test_goldenswag_process_docs_replica(self):
        doc = {"activity_label": "Ruoka", "ctx_a": "Alku [title] jatkoa", "ctx_b": "loppu",
               "endings": ["eka  jatko", "toka"], "label": "1"}
        prompt, choices, gold = finbench.TASKS["goldenswag"]["transform"](doc)
        self.assertTrue(prompt.startswith("Kerro loogisin jatko"))
        self.assertNotIn("[title]", prompt)            # replaced with ". "
        self.assertIn(". jatkoa Loppu", prompt)        # [title] -> ". ", ctx_b capitalised
        self.assertEqual(choices, ["eka jatko", "toka"])   # double space collapsed
        self.assertEqual(gold, 1)

    def test_sib200_uses_dataset_choices_and_index(self):
        doc = {"text": "U", "choices": ["x", "y", "z"], "answer_idx": 2}
        prompt, choices, gold = finbench.TASKS["sib200"]["transform"](doc)
        self.assertEqual((choices, gold), (["x", "y", "z"], 2))
        self.assertTrue(prompt.endswith("Aihe:"))

    def test_belebele_prompt_and_gold_index(self):
        doc = {"flores_passage": "P", "question": "Q", "mc_answer1": "a", "mc_answer2": "b",
               "mc_answer3": "c", "mc_answer4": "d", "correct_answer_num": "3"}
        prompt, choices, gold = finbench.TASKS["belebele_fin"]["transform"](doc)
        self.assertEqual((choices, gold), (["a", "b", "c", "d"], 2))
        self.assertIn("Tässä on teksti: P", prompt)

    def test_every_task_declares_its_upstream_source(self):
        for name, spec in finbench.TASKS.items():
            self.assertIn("upstream", spec, name)
            self.assertIn("cf/", spec["upstream"], name)
            self.assertTrue(spec["doc_to_text"], name)


if __name__ == "__main__":
    unittest.main()
