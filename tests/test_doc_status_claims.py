"""Status text must follow the data: a shipped protocol may not be called unrun.

Review 5.6 (2026-10-24): nothing pinned the docs' status claims to the repository
state, so three documents said the cold protocol "has not been run" / "stays open
work" / "needs no protocol change" in the very commit that published cold rows —
and the earlier pilot table in RUNNING §3c quoted warm values that no artifact on
disk contained.

This guard reads the tracked rollup, decides what is actually published, and then
refuses stale status text in the documents a reader is invited to check. It also
pins `matrix.yaml`'s default protocol to the published one, because an all-default
sweep must reproduce the published column rather than drift from the docs (I1).
"""
import csv
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLUP = os.path.join(REPO, "results", "consolidated.csv")
MATRIX = os.path.join(REPO, "matrix.yaml")

DOCS = {
    "README.md": os.path.join(REPO, "README.md"),
    "PLAN.md": os.path.join(REPO, "PLAN.md"),
    "docs/RUNNING.md": os.path.join(REPO, "docs", "RUNNING.md"),
    "results/table.md": os.path.join(REPO, "results", "table.md"),
}

# Phrasings that assert a protocol has NOT been measured/shipped yet.
UNRUN = (
    "has not been run",
    "stays open work",
    "needs no protocol change",
    "a cold-request protocol would isolate",
)

LABEL = re.compile(r"\[(\w+) KV, (warm|cold) TTFT")


def _matrix_defaults():
    import yaml  # tests already depend on PyYAML via scripts/run_sweep.py
    with open(MATRIX) as f:
        return (yaml.safe_load(f) or {}).get("defaults") or {}


@unittest.skipUnless(os.path.isfile(ROLLUP), "no tracked rollup")
class DocStatusClaimTest(unittest.TestCase):

    def setUp(self):
        self.published = {}   # (cell, backend) -> (kv, protocol, run_id)
        with open(ROLLUP, newline="") as f:
            for r in csv.DictReader(f):
                if r["metric"] != "ttft_ms_p50":
                    continue
                m = LABEL.search(r["note"] or "")
                if m is None:
                    self.fail(f"{r['run_id']} {r['cell']}/{r['backend']} "
                              f"ttft_ms_p50 has no protocol label")
                self.published[(r["cell"], r["backend"])] = (m.group(1), m.group(2),
                                                             r["run_id"])

    def protocols(self):
        return {proto for _, proto, _ in self.published.values()}

    def test_matrix_default_is_the_published_protocol(self):
        """I1: an all-default sweep must reproduce the column the docs describe."""
        defaults = _matrix_defaults()
        self.assertEqual(defaults.get("ttft_protocol"), "cold",
                         "matrix.yaml defaults.ttft_protocol drifted from the published "
                         "protocol")
        self.assertEqual(defaults.get("kv"), "f16",
                         "matrix.yaml defaults.kv drifted from the published KV type")
        for (cell, backend), (kv, proto, run_id) in self.published.items():
            with self.subTest(cell=cell, backend=backend):
                self.assertEqual((kv, proto), (defaults.get("kv"),
                                               defaults.get("ttft_protocol")),
                                 f"{cell}/{backend} publishes {kv}/{proto} from {run_id}")

    def test_docs_do_not_call_a_published_protocol_unrun(self):
        live = self.protocols()
        self.assertIn("cold", live, "the cold protocol is no longer published — the "
                                    "status text below may legitimately say it is open")
        for name, path in DOCS.items():
            with open(path) as f:
                text = f.read().lower()
            for phrase in UNRUN:
                with self.subTest(doc=name, phrase=phrase):
                    self.assertNotIn(phrase, text,
                                     f"{name} still says {phrase!r} while the rollup "
                                     f"publishes {sorted(live)} TTFT rows — update the "
                                     f"status text in the same commit as the data")

    def test_status_prose_names_the_published_protocol(self):
        """README and RUNNING must say which protocol the column carries."""
        for name in ("README.md", "docs/RUNNING.md"):
            with open(DOCS[name]) as f:
                text = f.read().lower()
            with self.subTest(doc=name):
                self.assertIn("cold", text,
                              f"{name} never names the published cold protocol")


if __name__ == "__main__":
    unittest.main()
