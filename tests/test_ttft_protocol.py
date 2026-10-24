"""Tests for the cold-request TTFT protocol (2026-10-24).

The published TTFT column used to measure whatever the previous request left in
the KV cache: on Vulkan a 512-token generation inflated the next request's
first-token latency by 200-900 ms, so `ttft_ms` and `ttft_ms_p50` disagreed and
neither was comparable across cells (RUNNING §7, PLAN D6/D9).

The cold protocol erases the slot before every timed request, so each one
measures full prompt prefill + first token from an empty cache. These tests pin:

  * the flag defaults to the published protocol and is overridable,
  * the server gets --slot-save-path only for cold (the erase API needs it),
  * the probe erases before every TIMED request and not before the warmup,
    untimed, and records what it dropped,
  * partial-decode detection uses finish_reason, not the SSE chunk count,
  * the rollup note names the protocol, so a warm and a cold row can never be
    mistaken for the same quantity.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import run_sweep  # noqa: E402
import ttft_probe  # noqa: E402

CELL = {"id": "qwen3-4b-q8_0", "file": "/models/Qwen3-4B-Q8_0.gguf", "size_gib": 4.0}


class _FakeResponse:
    """Minimal stand-in for the HTTP response the probe iterates (SSE lines)."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return b"".join(self._lines)


class _FakeHTTP:
    """Records every urlopen call: (url, decoded body) plus the reply to send back."""

    def __init__(self, erase_reply=b'{"id_slot":0,"n_erased":557}'):
        self.calls = []
        self.erase_reply = erase_reply

    def __call__(self, req, timeout=None):
        url = getattr(req, "full_url", str(req))
        data = getattr(req, "data", b"")
        self.calls.append((url, json.loads(data.decode() or "{}")))
        if "/slots/" in url:
            return _FakeResponse([self.erase_reply])
        return _FakeResponse([
            b'data: {"choices":[{"text":"a","finish_reason":null}]}\n',
            b'data: {"choices":[{"text":"b","finish_reason":"length"}]}\n',
            b"data: [DONE]\n",
        ])

    @property
    def erases(self):
        return [c for c in self.calls if "/slots/" in c[0]]

    @property
    def completions(self):
        return [c for c in self.calls if c[0].endswith("/v1/completions")]


def _run_probe(protocol, n=3, warmup=1, fake=None):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ttft.json")
        fake = fake or _FakeHTTP()
        argv = ["ttft_probe.py", "http://127.0.0.1:11500", "qwen3-4b-q8_0",
                str(warmup), str(n), "512", out, protocol]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("urllib.request.urlopen", fake):
            ttft_probe.main()
        with open(out) as f:
            return json.load(f), fake


class ProbeProtocolTest(unittest.TestCase):

    def test_cold_erases_before_every_timed_request_only(self):
        result, fake = _run_probe("cold", n=3, warmup=1)
        self.assertEqual(result["protocol"], "cold")
        self.assertEqual(len(fake.erases), 3,
                         "one untimed erase per TIMED request, not per warmup request")
        self.assertEqual(len(fake.completions), 4, "1 warmup + 3 timed requests")
        self.assertEqual(result["requests"][0]["erased_tokens"], 557,
                         "the probe must record how many tokens the erase dropped")

    def test_warm_never_touches_the_slots_api(self):
        result, fake = _run_probe("warm", n=3, warmup=1)
        self.assertEqual(result["protocol"], "warm")
        self.assertEqual(fake.erases, [], "the warm arm must reproduce the old protocol")
        self.assertNotIn("erased_tokens", result["requests"][0])

    def test_protocol_does_not_change_the_request_body(self):
        """Cache control lives in the server call, not in a per-request override."""
        _, warm = _run_probe("warm", n=2, warmup=0)
        _, cold = _run_probe("cold", n=2, warmup=0)
        self.assertEqual(warm.completions[0][1], cold.completions[0][1])
        self.assertNotIn("cache_prompt", cold.completions[0][1])

    def test_unknown_protocol_is_refused(self):
        with self.assertRaises(SystemExit):
            _run_probe("tepid", n=1, warmup=0)

    def test_protocol_argument_is_required(self):
        """No default: an unlabelled run would publish warm numbers under a cold name."""
        with mock.patch.object(sys, "argv", ["ttft_probe.py", "http://h", "m", "1",
                                             "3", "512", "/tmp/x.json"]):
            with self.assertRaises(SystemExit):
                ttft_probe.main()

    def test_an_erase_that_drops_nothing_fails_instead_of_reporting_cold(self):
        fake = _FakeHTTP(erase_reply=b'{"id_slot":0,"n_erased":0}')
        with self.assertRaises(RuntimeError) as ctx:
            _run_probe("cold", n=2, warmup=1, fake=fake)
        self.assertIn("dropped no tokens", str(ctx.exception))

    def test_a_server_without_the_slots_api_fails_with_a_usable_message(self):
        import urllib.error

        class _Refusing(_FakeHTTP):
            def __call__(self, req, timeout=None):
                if "/slots/" in getattr(req, "full_url", str(req)):
                    raise urllib.error.HTTPError(
                        req.full_url, 501, "Not Supported", None,
                        io.BytesIO(b'{"error":"This server does not support slots action"}'))
                return super().__call__(req, timeout)

        with self.assertRaises(RuntimeError) as ctx:
            _run_probe("cold", n=1, warmup=0, fake=_Refusing())
        msg = str(ctx.exception)
        self.assertIn("501", msg)
        self.assertIn("--slot-save-path", msg)


class CompleteDecodesTest(unittest.TestCase):
    """The cold arm can report 508 chunks for a 512-token decode (tail merging)."""

    def test_finish_reason_length_counts_as_complete(self):
        self.assertTrue(ttft_probe._complete({"tokens": 508, "finish_reason": "length"}, 512))

    def test_early_stop_is_incomplete(self):
        self.assertFalse(ttft_probe._complete({"tokens": 512, "finish_reason": "stop"}, 512))

    def test_missing_finish_reason_falls_back_to_the_count(self):
        self.assertTrue(ttft_probe._complete({"tokens": 512, "finish_reason": None}, 512))
        self.assertFalse(ttft_probe._complete({"tokens": 40, "finish_reason": None}, 512))


class LauncherTest(unittest.TestCase):

    def test_cold_adds_the_slot_save_path_and_warm_does_not(self):
        warm = run_sweep.server_argv("/b", CELL, 8192, 11500, 16, 99, "f16", "warm", "/slots")
        self.assertNotIn("--slot-save-path", warm)
        cold = run_sweep.server_argv("/b", CELL, 8192, 11500, 16, 99, "f16", "cold", "/slots")
        self.assertEqual(cold[cold.index("--slot-save-path") + 1], "/slots")
        for argv in (warm, cold):
            self.assertEqual(argv[argv.index("-ctk") + 1], "f16")
            self.assertEqual(argv[argv.index("-ctv") + 1], "f16")

    def test_cold_without_a_slot_dir_is_refused(self):
        """M1: never render `--slot-save-path None`."""
        with self.assertRaises(ValueError):
            run_sweep.server_argv("/b", CELL, 8192, 11500, 16, 99, "f16", "cold", None)

    def test_probe_invocation_carries_the_protocol(self):
        argv = run_sweep.probe_argv("http://h", CELL, 1, 3, 512, "/o.json", "cold")
        self.assertEqual(argv[-1], "cold")
        self.assertIn("/scripts/ttft_probe.py", argv[1])


class DefaultsTest(unittest.TestCase):

    def test_matrix_defaults_to_the_published_protocol(self):
        defaults = run_sweep.load_matrix()["defaults"]
        self.assertEqual(defaults["ttft_protocol"], "cold",
                         "the default must match what is published, or the column "
                         "silently drifts from the docs (the 2026-10-23 lesson)")
        self.assertEqual(defaults["kv"], "f16",
                         "I1 (review 2026-10-24): --kv defaulted to q8_0 while every "
                         "doc says the published column is f16 KV")
        p = run_sweep.build_parser(defaults)
        self.assertEqual(p.parse_args([]).ttft_protocol, "cold")
        self.assertEqual(p.parse_args([]).kv, "f16")

    def test_matrix_default_is_wired_and_overridable(self):
        p = run_sweep.build_parser({"ttft_protocol": "warm", "kv": "q8_0"})
        self.assertEqual(p.parse_args([]).ttft_protocol, "warm")
        self.assertEqual(p.parse_args([]).kv, "q8_0")
        self.assertEqual(p.parse_args(["--ttft-protocol", "cold",
                                       "--kv", "f16"]).ttft_protocol, "cold")


class RunCompletionStampTest(unittest.TestCase):
    """I2: a run is stamped incomplete until its last cell finishes."""

    def test_mark_run_complete_clears_the_partial_flag(self):
        with tempfile.TemporaryDirectory() as rdir:
            with open(os.path.join(rdir, "run.json"), "w") as f:
                json.dump({"run_id": "run-x", "partial": True, "tag": "run-x-1"}, f)
            meta = run_sweep.mark_run_complete(rdir)
            self.assertFalse(meta["partial"])
            with open(os.path.join(rdir, "run.json")) as f:
                self.assertFalse(json.load(f)["partial"], "the flag must reach the file")
            self.assertEqual(meta["tag"], "run-x-1", "other stamps must survive")

    def test_a_run_that_never_finished_is_skipped_by_consolidate(self):
        """The crash path: partial stays true, so no rows enter the rollup."""
        with tempfile.TemporaryDirectory() as repo:
            bdir = os.path.join(repo, "results", "run-crashed", "vulkan")
            cdir = os.path.join(bdir, "qwen3-4b-q8_0")
            os.makedirs(cdir)
            with open(os.path.join(bdir, "run.json"), "w") as f:
                json.dump({"run_id": "run-crashed", "backend": "vulkan",
                           "partial": True, "covers_matrix": True}, f)
            with open(os.path.join(cdir, "cell.json"), "w") as f:
                json.dump({"cell": "qwen3-4b-q8_0", "backend": "vulkan",
                           "run": "run-crashed", "kv": "f16", "probe_only": True,
                           "model": "m", "quant": "Q8_0", "size_gib": 4.0,
                           "bench": {}, "ttft": {"gen_tps_mean": 1.0, "ttft_ms_mean": 1.0,
                                                 "ttft_ms_p50": 1.0, "requests": [{}]},
                           "resources": {"bench_window": None, "probe_window": {},
                                         "cell_window": {}}}, f)
            run_sweep.consolidate(repo=repo)
            with open(os.path.join(repo, "results", "consolidated.csv")) as f:
                self.assertEqual(len([r for r in f.read().splitlines()[1:] if r]),
                                 0, "an unfinished run must publish nothing")


class RollupNoteTest(unittest.TestCase):
    """A cold TTFT and a warm TTFT are different quantities; the note must say which."""

    def _consolidate(self, cell_json):
        with tempfile.TemporaryDirectory() as repo:
            bdir = os.path.join(repo, "results", "run-20261024-1000", "vulkan")
            cdir = os.path.join(bdir, "qwen3-4b-q8_0")
            os.makedirs(cdir)
            with open(os.path.join(bdir, "run.json"), "w") as f:
                json.dump({"run_id": "run-20261024-1000", "backend": "vulkan",
                           "partial": False}, f)
            with open(os.path.join(cdir, "cell.json"), "w") as f:
                json.dump(cell_json, f)
            run_sweep.consolidate(repo=repo)
            with open(os.path.join(repo, "results", "consolidated.csv"), newline="") as f:
                return {r["metric"]: r["note"] for r in
                        __import__("csv").DictReader(f)}

    @staticmethod
    def _cell(**over):
        base = {"cell": "qwen3-4b-q8_0", "backend": "vulkan", "run": "run-20261024-1000",
                "kv": "f16", "model": "Qwen3-4B", "quant": "Q8_0", "size_gib": 4.0,
                "bench": {}, "probe_only": True,
                "ttft": {"gen_tps_mean": 48.1, "ttft_ms_mean": 54.7, "ttft_ms_p50": 54.8,
                         "requests": [{}, {}, {}]},
                "resources": {"bench_window": None, "probe_window": {}, "cell_window": {}}}
        base.update(over)
        return base

    def test_cold_probe_only_note(self):
        notes = self._consolidate(self._cell(ttft_protocol="cold"))
        self.assertIn("f16 KV, cold TTFT, probe-only run", notes["ttft_ms_p50"])

    def test_warm_probe_only_note(self):
        notes = self._consolidate(self._cell(ttft_protocol="warm"))
        self.assertIn("f16 KV, warm TTFT, probe-only run", notes["ttft_ms_p50"])

    def test_legacy_cell_json_reads_as_q8_warm(self):
        # No ttft_protocol and no kv key: that is what every run before the flags
        # looks like — the driver hardcoded q8_0 KV and reused the prompt cache.
        cell = self._cell()
        del cell["kv"]
        notes = self._consolidate(cell)
        self.assertIn("[q8_0 KV, warm TTFT, probe-only run]", notes["ttft_ms_p50"])

    def test_full_warm_sweep_is_labelled_too(self):
        """I3: every streaming row names its condition, not just the probe-only ones."""
        cell = self._cell(ttft_protocol="warm", probe_only=False,
                          bench={"pp2048": {"mean": 1857.9, "repeats": 3}})
        notes = self._consolidate(cell)
        self.assertIn("[f16 KV, warm TTFT]", notes["ttft_ms"])
        self.assertEqual(notes["pp2048_bench_tps"], "llama-bench mean of 3")

    def test_cold_full_sweep_labels_the_streaming_rows_only(self):
        cell = self._cell(ttft_protocol="cold", probe_only=False,
                          bench={"pp2048": {"mean": 1857.9, "repeats": 3}})
        notes = self._consolidate(cell)
        self.assertEqual(notes["ttft_ms"], "mean of 3 [f16 KV, cold TTFT]")
        self.assertEqual(notes["pp2048_bench_tps"], "llama-bench mean of 3",
                         "the probe protocol cannot change bench tok/s — labelling "
                         "those rows would imply it did")


class TagMessageTest(unittest.TestCase):

    def test_message_names_the_protocol(self):
        msg = run_sweep.tag_message("vulkan", 10, "abcdef1234", "5b59b83f4e2",
                                    "k", "m", "f16", True, "cold")
        self.assertIn("KV f16, TTFT cold", msg)


if __name__ == "__main__":
    unittest.main()
