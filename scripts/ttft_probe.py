#!/usr/bin/env python3
"""TTFT + sustained generation probe against a running llama-server.

Usage: ttft_probe.py BASE_URL MODEL WARMUP N MAX_TOKENS OUT_JSON

Sends WARMUP untimed requests (primes CUDA/HIP graphs), then N timed
streaming /v1/completions requests of MAX_TOKENS tokens (temperature 0).
TTFT = time to first streamed token. gen_tps = tokens / (t_end - t_first),
where both are absolute perf_counter() readings.
stdlib only.
"""
import json
import statistics
import sys
import time
import urllib.request

PROMPT = ("You are a benchmark workload. Write a detailed, methodical technical "
          "explanation of how a unified-memory APU executes matrix "
          "multiplication for LLM inference. Cover memory bandwidth, "
          "quantized GEMM kernels, and attention. Be thorough.\n")


def one_request(base, model, max_tokens):
    body = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
        "stop": [],
        # Force the full decode budget. Without this, a model that ends its turn
        # immediately on a raw completion prompt (Qwen3.8 does) returns a single
        # chunk with text "" and finish_reason "stop": the sustained-generation
        # figure is meaningless and the probe dies with "no tokens received (n=0)".
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None          # ABSOLUTE clock reading of the first streamed token
    finish_reason = None
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:                      # SSE: one JSON per "data:" line
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            choice = d.get("choices", [{}])[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            text = choice.get("text", "")
            if text:
                n += 1
                if t_first is None:
                    t_first = time.perf_counter()
    t_end = time.perf_counter()
    if t_first is None or n == 0:
        raise RuntimeError(
            f"no tokens received (n={n}, finish_reason={finish_reason!r}). "
            f"finish_reason='stop' means the model ended its turn before emitting "
            f"any text — check that the server honours ignore_eos")
    ttft = t_first - t0
    # gen_tps must divide by the DECODE window (t_end - t_first) — two absolute
    # perf_counter readings. The old form subtracted `ttft`, a *duration*, from
    # the absolute t_end, yielding ~system uptime as the divisor: it reported
    # ~0.01 t/s for every model and silently zeroed tok_per_w downstream.
    decode_s = max(t_end - t_first, 1e-9)
    return {
        "ttft_ms": round(ttft * 1000, 1),
        "tokens": n,
        "tokens_requested": max_tokens,
        "finish_reason": finish_reason,
        "gen_tps": round(n / decode_s, 2),
        "total_s": round(t_end - t0, 2),
    }


def main():
    base, model = sys.argv[1], sys.argv[2]
    warmup, n, max_tokens = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    out = sys.argv[6]

    for i in range(warmup):
        one_request(base, model, min(32, max_tokens))   # short warmup is enough to prime
        print(f"warmup {i + 1}/{warmup} done", file=sys.stderr)

    reqs = [one_request(base, model, max_tokens) for _ in range(n)]
    ttfts = [r["ttft_ms"] for r in reqs]
    tps = [r["gen_tps"] for r in reqs]
    result = {
        "model": model,
        "prompt_chars": len(PROMPT),
        "max_tokens": max_tokens,
        "requests": reqs,
        # False => at least one timed decode stopped early, so gen_tps is a
        # partial-budget figure. Surfaced instead of being silently averaged in.
        "complete_decodes": all(r["tokens"] >= max_tokens for r in reqs),
        "ttft_ms_mean": round(statistics.fmean(ttfts), 1),
        "ttft_ms_p50": round(statistics.median(ttfts), 1),
        "gen_tps_mean": round(statistics.fmean(tps), 2),
        "gen_tps_p50": round(statistics.median(tps), 2),
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in
                      ("ttft_ms_mean", "ttft_ms_p50", "gen_tps_mean", "gen_tps_p50",
                       "complete_decodes")}, indent=1))


if __name__ == "__main__":
    main()
