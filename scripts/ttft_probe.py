#!/usr/bin/env python3
"""TTFT + sustained generation probe against a running llama-server.

Usage: ttft_probe.py BASE_URL MODEL WARMUP N MAX_TOKENS OUT_JSON

Sends WARMUP untimed requests (primes CUDA/HIP graphs), then N timed
streaming /v1/completions requests of MAX_TOKENS tokens (temperature 0).
TTFT = time to first streamed token. gen_tps = tokens / (last - first).
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
    }).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
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
            text = d.get("choices", [{}])[0].get("text", "")
            if text:
                n += 1
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
    t_end = time.perf_counter()
    if ttft is None or n == 0:
        raise RuntimeError(f"no tokens received (n={n})")
    return {
        "ttft_ms": round(ttft * 1000, 1),
        "tokens": n,
        "gen_tps": round(n / max(t_end - ttft, 1e-9), 2),
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
        "ttft_ms_mean": round(statistics.fmean(ttfts), 1),
        "ttft_ms_p50": round(statistics.median(ttfts), 1),
        "gen_tps_mean": round(statistics.fmean(tps), 2),
        "gen_tps_p50": round(statistics.median(tps), 2),
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in
                      ("ttft_ms_mean", "ttft_ms_p50", "gen_tps_mean", "gen_tps_p50")}, indent=1))


if __name__ == "__main__":
    main()
