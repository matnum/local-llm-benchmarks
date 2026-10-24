"""Tests for the embeddings axis: metric primitives and response handling.

No GPU here — the server is exercised by the run itself. What is pinned is the
arithmetic (a metric that is quietly wrong is worse than no metric) and the
response parsing that would otherwise misalign vectors with passages silently.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import embed_eval  # noqa: E402


class MetricTest(unittest.TestCase):
    def test_normalize_makes_cosine_a_dot_product(self):
        vec = embed_eval.normalize([3.0, 4.0])
        self.assertAlmostEqual(vec[0], 0.6)
        self.assertAlmostEqual(vec[1], 0.8)
        self.assertAlmostEqual(embed_eval.cosine(vec, embed_eval.normalize([3.0, 4.0])), 1.0)
        self.assertAlmostEqual(embed_eval.cosine(vec, embed_eval.normalize([-3.0, -4.0])), -1.0)

    def test_rank_orders_by_score_and_breaks_ties_by_index(self):
        ranked = embed_eval.rank_passages([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(ranked, [1, 2, 0])

    def test_rank_is_deterministic_on_identical_scores(self):
        vectors = [[1.0, 0.0]] * 5
        self.assertEqual(embed_eval.rank_passages([1.0, 0.0], vectors), [0, 1, 2, 3, 4])

    def test_hit_recall_mrr_ndcg_on_a_known_ranking(self):
        # Relevant passages are at ranks 2 and 4 (1-based); top-3 catches one.
        ranked = [9, 1, 8, 2, 7]
        relevant = {1, 2}
        self.assertEqual(embed_eval.hit_at_k(ranked, relevant, 1), 0.0)
        self.assertEqual(embed_eval.hit_at_k(ranked, relevant, 3), 1.0)
        self.assertAlmostEqual(embed_eval.recall_at_k(ranked, relevant, 3), 0.5)
        self.assertAlmostEqual(embed_eval.recall_at_k(ranked, relevant, 5), 1.0)
        self.assertAlmostEqual(embed_eval.mrr_at_k(ranked, relevant, 10), 0.5)
        self.assertEqual(embed_eval.mrr_at_k(ranked, {42}, 10), 0.0)
        # DCG = 1/log2(3) + 1/log2(5); ideal = 1/log2(2) + 1/log2(3).
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        ideal = 1.0 + 1 / math.log2(3)
        self.assertAlmostEqual(embed_eval.ndcg_at_k(ranked, relevant, 10), dcg / ideal, places=9)

    def test_mrr_respects_the_cutoff(self):
        # First relevant passage is at rank 4, outside a top-3 cutoff.
        self.assertEqual(embed_eval.mrr_at_k([9, 8, 7, 1], {1}, 3), 0.0)
        self.assertAlmostEqual(embed_eval.mrr_at_k([9, 8, 7, 1], {1}, 4), 0.25)

    def test_ndcg_is_one_for_a_perfect_ranking(self):
        self.assertAlmostEqual(embed_eval.ndcg_at_k([1, 2, 3], {1, 2, 3}, 10), 1.0)


class PrefixTest(unittest.TestCase):
    def test_nomic_gets_its_task_prefixes(self):
        self.assertEqual(
            embed_eval.prefixes_for("nomic-embed-text-v1.5"),
            ("search_query: ", "search_document: "),
        )

    def test_bge_m3_is_not_given_prefixes_it_was_not_trained_with(self):
        self.assertEqual(embed_eval.prefixes_for("bge-m3"), ("", ""))

    def test_explicit_flags_win(self):
        self.assertEqual(
            embed_eval.prefixes_for("nomic", query_prefix="", doc_prefix=""),
            ("", ""),
        )


class ResponseTest(unittest.TestCase):
    def test_embeddings_are_placed_by_index_not_by_list_order(self):
        payload = {"data": [{"index": 1, "embedding": [0.0, 1.0]},
                            {"index": 0, "embedding": [1.0, 0.0]}]}
        self.assertEqual(embed_eval.parse_embeddings(payload, 2), [[1.0, 0.0], [0.0, 1.0]])

    def test_a_short_response_is_an_error_not_a_misalignment(self):
        payload = {"data": [{"index": 0, "embedding": [1.0]}]}
        with self.assertRaises(ValueError):
            embed_eval.parse_embeddings(payload, 2)

    def test_a_gap_in_indices_is_an_error(self):
        payload = {"data": [{"index": 0, "embedding": [1.0]},
                            {"index": 0, "embedding": [2.0]}]}
        with self.assertRaises(ValueError):
            embed_eval.parse_embeddings(payload, 2)

    def test_batches_covers_everything_once(self):
        items = list(range(5))
        self.assertEqual([b for b in embed_eval.batches(items, 2)], [[0, 1], [2, 3], [4]])


class TableTest(unittest.TestCase):
    def _summary(self, model, hit1, backend="vulkan", stamp="20261024-1200"):
        return {
            "model": model,
            "backend": backend,
            "llama_cpp": "5b59b83f4",
            "metrics": {"hit@1": hit1, "hit@5": 0.9, "hit@10": 0.95,
                        "mrr@10": 0.8, "ndcg@10": 0.85},
            "timings": {"passages_per_s": 42.0, "startup_s": 3.0},
            "measured": stamp,
        }

    def test_table_has_one_row_per_model_and_backend(self):
        table = embed_eval.render_table(
            [
                self._summary("nomic", 0.7),
                self._summary("bge-m3", 0.8),
                self._summary("bge-m3", 0.81, backend="rocm"),
            ]
        )
        rows = [line for line in table.splitlines() if line.startswith("| ") and "Model" not in line]
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0].startswith("| bge-m3 | rocm |"))
        self.assertTrue(rows[1].startswith("| bge-m3 | vulkan |"))
        self.assertTrue(rows[2].startswith("| nomic | vulkan |"))

    def test_newest_run_wins_for_the_same_model_and_backend(self):
        table = embed_eval.render_table(
            [self._summary("bge-m3", 0.10), self._summary("bge-m3", 0.90)]
        )
        self.assertIn("0.900", table)
        self.assertNotIn("0.100", table)

    def test_paired_counts_compare_the_rank_of_the_first_relevant(self):
        left = self._summary("bge-m3", 0.9)
        right = self._summary("nomic", 0.6)
        left["per_query"] = [
            {"id": "q01", "first_relevant_rank": 1},
            {"id": "q02", "first_relevant_rank": 4},
            {"id": "q03", "first_relevant_rank": None},
            {"id": "q04", "first_relevant_rank": 2},
        ]
        right["per_query"] = [
            {"id": "q01", "first_relevant_rank": 3},
            {"id": "q02", "first_relevant_rank": 4},
            {"id": "q03", "first_relevant_rank": 9},
            {"id": "q04", "first_relevant_rank": 1},
        ]
        # q01 left better, q02 equal, q03 right better (left never ranked it),
        # q04 right better.
        self.assertEqual(
            embed_eval.paired_counts(left, right), {"left": 1, "right": 2, "tie": 1}
        )


if __name__ == "__main__":
    unittest.main()
