"""Golden and invariant tests for tools/score_ll — the FIN-bench loglikelihood scorer.

The scorer was the one piece of the axis with no test, and it is the riskiest: one
context pass per item, then a short pass per choice with the choice dropped from the
KV cache (`llama_memory_seq_rm(mem, 0, ctxt.size(), -1)`) so the next choice reuses
the same context. If that rollback were wrong, every choice after the first would be
quietly contaminated and the scores would still look plausible. So the loop is pinned
structurally, not only by a number:

  * reversing the choice list reverses the loglikelihoods exactly;
  * repeating a choice scores it identically both times.

The absolute values are a golden recorded on CPU (-ngl 0): GPU logprobs move by up to
~0.12 across kernel/batch shapes (docs/RUNNING.md §9), which would make an absolute
golden meaningless, while the CPU path is stable.

Skipped when the scorer binary or the fixture GGUF is absent, so the suite still runs
on a machine without model weights.
"""
import os
import subprocess
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORER = os.path.join(REPO, "tools", "score_ll")
MODEL = os.path.expanduser("~/.local/share/models/Qwen3-4B-Q4_K_M.gguf")

CONTEXT = "Vastaus kysymykseen, on:"
C1 = " kyllä, varmasti"
C2 = " ei, ei todellakaan"
C3 = " ehkä joskus"

# Recorded with `score_ll -m Qwen3-4B-Q4_K_M.gguf -ngl 0 -c 512 -t 16` on this fixture.
GOLDEN = {C1: (-18.192119, 7), C2: (-19.611826, 7), C3: (-21.796371, 5)}


def score(items):
    """score([(id, [choice, ...]), ...]) -> {id: [(ll, ntok), ...]}."""
    stdin = "".join("\t".join([iid, CONTEXT, *choices]) + "\n"
                    for iid, choices in items)
    p = subprocess.run(
        [SCORER, "-m", MODEL, "-ngl", "0", "-c", "512", "-t", "16"],
        input=stdin, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise AssertionError(f"score_ll exited {p.returncode}: {p.stderr[-800:]}")
    out = {}
    for line in p.stdout.splitlines():
        f = line.split("\t")
        if not f or len(f) % 2 == 0:
            raise AssertionError(f"malformed output line: {line!r}")
        out[f[0]] = [(float(f[1 + 2 * i]), int(f[2 + 2 * i]))
                     for i in range((len(f) - 1) // 2)]
    return out


@unittest.skipUnless(os.path.isfile(SCORER),
                     f"scorer not built (see docs/RUNNING.md §9): {SCORER}")
@unittest.skipUnless(os.path.isfile(MODEL), f"fixture model absent: {MODEL}")
class ScoreLLTest(unittest.TestCase):

    def test_golden_loglikelihoods(self):
        got = score([("g", [C1, C2, C3])])["g"]
        self.assertEqual(len(got), 3)
        for choice, (ll, ntok) in zip([C1, C2, C3], got):
            exp_ll, exp_ntok = GOLDEN[choice]
            with self.subTest(choice=choice):
                self.assertEqual(ntok, exp_ntok, "choice tokenisation changed")
                self.assertAlmostEqual(ll, exp_ll, delta=0.05,
                                       msg="loglikelihood drifted from the golden")

    def test_choice_order_is_invariant(self):
        """Reversing the choices must reverse the loglikelihoods."""
        res = score([("f", [C1, C2, C3]), ("r", [C3, C2, C1])])
        fwd, rev = res["f"], res["r"]
        self.assertEqual([t for _, t in fwd], [t for _, t in reversed(rev)])
        for (ll_f, _), (ll_r, _) in zip(fwd, reversed(rev)):
            self.assertAlmostEqual(ll_f, ll_r, places=4)

    def test_repeated_choice_scores_identically(self):
        """A choice must be fully dropped from the KV before the next one.

        If the rollback leaked, the second copy would attend to the first and score
        differently — a contamination no single-choice fixture can see.
        """
        (a, n_a), (b, n_b) = score([("d", [C1, C1])])["d"]
        self.assertEqual(n_a, n_b)
        self.assertAlmostEqual(a, b, places=4)


if __name__ == "__main__":
    unittest.main()
