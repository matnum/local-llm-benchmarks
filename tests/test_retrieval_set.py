"""Tests that pin the committed Finnish retrieval set to its own label rule.

The rule (PLAN.md D11) is: a passage is relevant to a query when it contains that
query's answer string. The qrels in `queries.json` are derived from it, so this
test re-derives them — a hand-edit of a label, a regenerated corpus, or a passage
that lost its answer string all fail here instead of quietly moving the axis's
published numbers.
"""

import json
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET_DIR = os.path.join(REPO, "docs", "retrieval", "fi-2026-10")
CORPUS = os.path.join(SET_DIR, "corpus.jsonl")
QUERIES = os.path.join(SET_DIR, "queries.json")
MANIFEST = os.path.join(SET_DIR, "manifest.json")

MIN_PASSAGE, MAX_PASSAGE = 60, 800


def load_corpus():
    with open(CORPUS, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_queries():
    with open(QUERIES, encoding="utf-8") as fh:
        return json.load(fh)


class CorpusTest(unittest.TestCase):
    def setUp(self):
        self.corpus = load_corpus()

    def test_set_is_large_enough_to_rank_against(self):
        # A retrieval axis over a handful of passages measures nothing.
        self.assertGreaterEqual(len(self.corpus), 100)

    def test_ids_are_unique_and_stable(self):
        ids = [row["id"] for row in self.corpus]
        self.assertEqual(len(ids), len(set(ids)))
        for row in self.corpus:
            self.assertRegex(row["id"], r"^[a-z0-9-]+#\d{3}$")

    def test_passages_respect_the_stated_bounds(self):
        for row in self.corpus:
            self.assertGreaterEqual(len(row["text"]), MIN_PASSAGE, row["id"])
            self.assertLessEqual(len(row["text"]), MAX_PASSAGE, row["id"])
            self.assertTrue(row["title"])
            self.assertTrue(row["section"])

    def test_corpus_covers_several_articles(self):
        self.assertGreaterEqual(len({row["title"] for row in self.corpus}), 20)

    def test_manifest_matches_the_corpus(self):
        with open(MANIFEST, encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest["passages"], len(self.corpus))
        self.assertIn("CC BY-SA", manifest["license"])
        self.assertEqual(
            len(manifest["articles"]), len({row["title"] for row in self.corpus})
        )
        for article in manifest["articles"]:
            self.assertEqual(len(article["sha256"]), 64)
            self.assertTrue(article["url"].startswith("https://fi.wikipedia.org/wiki/"))


class QueryTest(unittest.TestCase):
    def setUp(self):
        self.corpus = load_corpus()
        self.query_set = load_queries()
        self.by_id = {row["id"]: row for row in self.corpus}

    def test_set_carries_enough_queries(self):
        self.assertGreaterEqual(len(self.query_set["queries"]), 25)

    def test_query_ids_are_unique(self):
        ids = [item["id"] for item in self.query_set["queries"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_qrel_passes_exist_and_contain_the_answer(self):
        for item in self.query_set["queries"]:
            self.assertTrue(item["qrels"], item["id"])
            self.assertTrue(item["answer"].strip(), item["id"])
            for passage_id in item["qrels"]:
                self.assertIn(passage_id, self.by_id, item["id"])
                text = self.by_id[passage_id]["text"].lower()
                self.assertIn(item["answer"].lower(), text, f"{item['id']} -> {passage_id}")

    def test_labels_are_exactly_the_answer_rule(self):
        # The strongest form: no passage that contains the answer is left out, and
        # none that lacks it is labelled.
        for item in self.query_set["queries"]:
            answer = item["answer"].lower()
            expected = sorted(
                row["id"] for row in self.corpus if answer in row["text"].lower()
            )
            self.assertEqual(sorted(item["qrels"]), expected, item["id"])

    def test_queries_are_not_trivially_specific(self):
        # An answer string that occurs in most of the corpus makes the query a
        # keyword lookup; the axis is meant to rank passages.
        for item in self.query_set["queries"]:
            self.assertLessEqual(len(item["qrels"]), 4, item["id"])


if __name__ == "__main__":
    unittest.main()
