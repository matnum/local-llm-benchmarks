"""Pin the FIN-bench-v2 prompt grid to the vendored upstream templates.

The driver renders the upstream `doc_to_text` strings rather than paraphrasing
them (docs/finbench/upstream-<sha>.json, fetched by scripts/fetch_finbench_upstream.py
from LumiOpen/lm-evaluation-harness @ finbench_v2). These tests keep three things
honest:

  1. the snapshot is what it claims to be (per-file sha256);
  2. every one of the 5 families x 2 formulations x 5 variants renders from a real
     upstream template, with no unresolved Jinja left in the prompt;
  3. a handful of prompts are compared to the text as written upstream — including
     two upstream quirks that a "clean-up" would silently change: scandisent/mcf/p3
     spells "positivinen", and belebele/mcf prompts number the options 1-4 while
     the scored choices are the letters A-D.
"""
import hashlib
import json
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import finbench  # noqa: E402

SNAPSHOT = sorted(
    os.path.join(REPO, "docs", "finbench", f)
    for f in os.listdir(os.path.join(REPO, "docs", "finbench"))
    if f.startswith("upstream-") and f.endswith(".json"))

DOCS = {
    "arc_c": {"question": "MIKÄ-KYSYMYS", "choices": {"text": ["VASTAUS-A", "VASTAUS-B"],
                                                      "label": ["A", "B"]},
              "answerKey": "B"},
    "belebele_fin": {"flores_passage": "KAPPALE", "question": "KYSYMYS",
                     "mc_answer1": "VAST1", "mc_answer2": "VAST2",
                     "mc_answer3": "VAST3", "mc_answer4": "VAST4",
                     "correct_answer_num": "2"},
    "goldenswag": {"activity_label": "Aihe", "ctx_a": "Alku", "ctx_b": "loppu",
                   "endings": ["JATKO-A", "JATKO-B", "JATKO-C", "JATKO-D"],
                   "label": "0"},
    "scandisent": {"text": "ARVOSTELU", "label": "positive"},
    "sib200": {"text": "UUTINEN", "choices": ["politiikka", "urheilu"], "answer_idx": 0},
}


class SnapshotTest(unittest.TestCase):
    def test_exactly_one_snapshot_and_it_is_self_consistent(self):
        self.assertEqual(len(SNAPSHOT), 1, f"expected one upstream snapshot, found {SNAPSHOT}")
        data = json.load(open(SNAPSHOT[0], encoding="utf-8"))
        self.assertEqual(data["source"], "LumiOpen/lm-evaluation-harness")
        self.assertEqual(data["branch"], "finbench_v2")
        self.assertEqual(len(data["commit"]), 40)
        bad = [rel for rel, f in data["files"].items()
               if hashlib.sha256(f["text"].encode("utf-8")).hexdigest() != f["sha256"]]
        self.assertEqual(bad, [], f"snapshot files edited without refetching: {bad}")


class GridTest(unittest.TestCase):
    def test_every_cell_renders_without_unresolved_template_syntax(self):
        for family, formulation, variant in (
                (f, fo, v) for f in finbench.FAMILIES
                for fo in finbench.FORMULATIONS for v in finbench.VARIANTS):
            with self.subTest(family=family, formulation=formulation, variant=variant):
                prompt, choices, gold = finbench.item_for(
                    family, DOCS[family], formulation, variant)
                self.assertTrue(prompt.strip(), "empty prompt")
                self.assertNotIn("{{", prompt)
                self.assertNotIn("{%", prompt)
                self.assertGreaterEqual(len(choices), 2)
                self.assertIn(gold, range(len(choices)))

    def test_the_marker_field_survives_into_every_prompt(self):
        """Each family's prompt must carry the document: a template mix-up would drop it."""
        marker = {"arc_c": "MIKÄ-KYSYMYS", "belebele_fin": "KAPPALE",
                  "goldenswag": "Alku", "scandisent": "ARVOSTELU", "sib200": "UUTINEN"}
        for family, formulation, variant in (
                (f, fo, v) for f in finbench.FAMILIES
                for fo in finbench.FORMULATIONS for v in finbench.VARIANTS):
            with self.subTest(family=family, formulation=formulation, variant=variant):
                prompt = finbench.item_for(family, DOCS[family], formulation, variant)[0]
                self.assertIn(marker[family], prompt,
                              f"{family}/{formulation}/{variant} does not contain the doc")

    def test_cf_p0_prompts_are_the_upstream_strings(self):
        expected = {
            "arc_c": "Vastaus kysymykseen MIKÄ-KYSYMYS, on:",
            "belebele_fin": "Tässä on teksti: KAPPALE\nKysymys: KYSYMYS perustuen tekstiin.\nOikea vastaus:",
            "goldenswag": "Kerro loogisin jatko seuraavalle tekstille:\n\nAihe: Alku Loppu\n\nJatko:",
            "scandisent": "Arvostelijan teksti: ARVOSTELU\nTunne:",
            "sib200": "Päättele, mitä aihetta seuraava uutinen käsittelee. Uutinen: UUTINEN\nAihe:",
        }
        for family, want in expected.items():
            with self.subTest(family=family):
                self.assertEqual(finbench.item_for(family, DOCS[family], "cf", "p0")[0], want)

    def test_arc_mcf_lists_the_options_with_letters(self):
        prompt = finbench.item_for("arc_c", DOCS["arc_c"], "mcf", "p0")[0]
        self.assertEqual(
            prompt,
            "Mikä on paras vastaus kysymykseen MIKÄ-KYSYMYS?\n"
            " A VASTAUS-A\n B VASTAUS-B\nVastaus:")

    def test_upstream_quirks_are_replicated_not_corrected(self):
        """scandisent/mcf/p3 says "positivinen" upstream — a typo we keep verbatim."""
        prompt = finbench.item_for("scandisent", DOCS["scandisent"], "mcf", "p3")[0]
        self.assertIn("positivinen", prompt)
        self.assertNotIn("positiivinen", prompt)

    def test_belebele_mcf_numbers_options_but_scores_letters(self):
        """Upstream inconsistency: prompt numbers 1-4, doc_to_choice is A-D."""
        prompt, choices, _ = finbench.item_for(
            "belebele_fin", DOCS["belebele_fin"], "mcf", "p0")
        self.assertIn("1: VAST1", prompt)
        self.assertIn("4: VAST4", prompt)
        self.assertEqual(choices, ["A", "B", "C", "D"])


if __name__ == "__main__":
    unittest.main()
