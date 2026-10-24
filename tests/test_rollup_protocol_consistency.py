"""The published TTFT column must be one quantity, on every backend.

Review finding B1 (2026-10-24): a Vulkan cold re-probe was consolidated while
ROCm still carried the warm column, so the tracked rollup had
``('vulkan','cold'): 10`` and ``('rocm','warm'): 10`` for ``ttft_ms_p50``. The
README renders those two rows per cell side by side, so Qwen3-Coder read
180.9 ms (Vulkan, cold) against 21.4 ms (ROCm, warm) — an apparent 8.4x latency
penalty that was entirely the protocol difference. Nothing failed: the only red
test was the README-vs-CSV value diff, and only once the README was stale.

This guard reads the tracked rollup and refuses a column whose per-cell backend
pairs disagree on either the KV type or the TTFT protocol. It is skipped when the
rollup is absent (a fresh checkout without results still runs the rest).
"""
import csv
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLUP = os.path.join(REPO, "results", "consolidated.csv")

# "... [f16 KV, cold TTFT, probe-only run]" -> ("f16", "cold")
LABEL = re.compile(r"\[(\w+) KV, (warm|cold) TTFT")

# The columns that carry the probe's protocol. Bench/power/memory do not.
STREAMING = ("ttft_ms", "ttft_ms_p50", "tg_server_tps")


@unittest.skipUnless(os.path.isfile(ROLLUP), "no tracked rollup")
class RollupProtocolConsistencyTest(unittest.TestCase):

    def setUp(self):
        self.latest = {}          # (cell, backend, metric) -> (kv, protocol, run_id)
        with open(ROLLUP, newline="") as f:
            for r in csv.DictReader(f):
                if r["metric"] not in STREAMING:
                    continue
                m = LABEL.search(r["note"] or "")
                if not m:
                    self.fail(f"{r['run_id']} {r['cell']}/{r['backend']} "
                              f"{r['metric']} has no protocol label in its note: "
                              f"{r['note']!r} — every streaming row must say which "
                              f"condition it was measured under")
                self.latest[(r["cell"], r["backend"], r["metric"])] = (
                    m.group(1), m.group(2), r["run_id"])

    def test_every_cell_agrees_across_backends(self):
        cells = sorted({c for c, _, _ in self.latest})
        self.assertTrue(cells, "the rollup has no streaming rows")
        for cell in cells:
            for metric in STREAMING:
                with self.subTest(cell=cell, metric=metric):
                    seen = {be: self.latest[(cell, be, metric)]
                            for be in ("vulkan", "rocm")
                            if (cell, be, metric) in self.latest}
                    self.assertEqual(len(seen), 2,
                                     f"{cell}/{metric} is missing a backend: {seen}")
                    conditions = {(kv, proto) for kv, proto, _ in seen.values()}
                    self.assertEqual(
                        len(conditions), 1,
                        f"{cell} {metric} mixes conditions across backends: "
                        f"{ {be: v[:2] for be, v in seen.items()} } — publish one "
                        f"protocol per column (run the missing backend with the same "
                        f"--kv and --ttft-protocol), or publish two labelled columns")

    def test_the_column_is_the_published_protocol(self):
        """The docs say f16 KV + cold; the rollup must say the same."""
        cells = sorted({c for c, _, _ in self.latest})
        for cell in cells:
            with self.subTest(cell=cell):
                kv, proto, run_id = self.latest[(cell, "vulkan", "ttft_ms_p50")]
                self.assertEqual((kv, proto), ("f16", "cold"),
                                 f"{cell} publishes {kv} KV / {proto} TTFT from "
                                 f"{run_id}, but matrix.yaml and the docs say f16/cold")


if __name__ == "__main__":
    unittest.main()
