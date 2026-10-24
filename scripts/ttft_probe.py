#!/usr/bin/env python3
"""TTFT + sustained generation probe against a running llama-server.

Usage: ttft_probe.py BASE_URL MODEL WARMUP N MAX_TOKENS OUT_JSON PROTOCOL

Sends WARMUP untimed requests (primes CUDA/HIP graphs), then N timed
streaming /v1/completions requests of MAX_TOKENS tokens (temperature 0).
TTFT = time to first streamed token. gen_tps = tokens / (t_end - t_first),
where both are absolute perf_counter() readings.
stdlib only.

PROTOCOL is required (no default: a silent warm run published under a cold
heading is the defect this argument exists to prevent) and says what the TIMED
requests start from:

  warm  the server's prompt cache is left alone, so a request that follows a
        512-token generation pays the KV trim/rollback that generation left
        behind — on Vulkan that cost lands in TTFT (RUNNING §7), which is why
        this arm is kept only to reproduce pre-2026-10-24 runs.
  cold  the slot's KV cache is erased before every timed request, so each
        request measures the same thing: full prompt prefill + first token from
        an empty cache. The erase call is untimed — it is protocol setup, not
        latency — and the result records how many tokens it dropped.
"""
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

PROMPT = ("You are a benchmark workload. Write a detailed, methodical technical "
          "explanation of how a unified-memory APU executes matrix "
          "multiplication for LLM inference. Cover memory bandwidth, "
          "quantized GEMM kernels, and attention. Be thorough.\n")


def erase_slot(base, id_slot=0):
    """Drop a slot's KV cache via the server's slots API.

    The call is untimed by design (see main): erasing is protocol setup, not
    work the request being measured should pay for. It needs the server started
    with --slot-save-path, which run_sweep.probe_cell() adds for `cold`.
    """
    req = urllib.request.Request(f"{base}/slots/{id_slot}?action=erase", data=b"{}",
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(
            f"slot erase failed: HTTP {e.code} {body} — POST /slots only exists when "
            f"llama-server was started with --slot-save-path") from e


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


def _complete(r, max_tokens):
    """Did this request spend its whole decode budget?

    finish_reason is the signal, not the chunk count: the server may merge the
    tail tokens into one SSE chunk (a cold run reported 508-510 chunks for a
    512-token decode), so `tokens >= max_tokens` produces false partial-decode
    flags. Fall back to the count only when the server sent no finish_reason.
    """
    finish = r.get("finish_reason")
    if finish is not None:
        return finish == "length"
    return r["tokens"] >= max_tokens


def main():
    if len(sys.argv) < 8:
        raise SystemExit("usage: ttft_probe.py BASE MODEL WARMUP N MAX_TOKENS OUT_JSON "
                         "PROTOCOL(warm|cold)")
    base, model = sys.argv[1], sys.argv[2]
    warmup, n, max_tokens = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    out = sys.argv[6]
    protocol = sys.argv[7]
    if protocol not in ("warm", "cold"):
        raise SystemExit(f"unknown TTFT protocol {protocol!r} (expected warm|cold)")

    for i in range(warmup):
        one_request(base, model, min(32, max_tokens))   # short warmup is enough to prime
        print(f"warmup {i + 1}/{warmup} done", file=sys.stderr)

    reqs = []
    for _ in range(n):
        # Cold protocol: every timed request starts from an empty KV cache, so
        # the KV work left by the previous 512-token generation is paid here,
        # outside the measured window, instead of inside TTFT.
        erased = None
        if protocol == "cold":
            reply = erase_slot(base)
            erased = reply.get("n_erased")
            # A cold measurement must start from an empty cache. If the erase
            # dropped nothing the server kept the sequence (wrong action, or a
            # build without the slots API) and this request would quietly measure
            # the warm protocol under a cold label. The one legitimate zero is the
            # first erase of a run with no warmup, where the cache is already empty.
            if not erased and not (not reqs and warmup == 0):
                raise RuntimeError(
                    f"slot erase dropped no tokens ({reply!r}), so the timed request "
                    f"would not start from an empty cache — refusing to label a warm "
                    f"measurement as cold")
        r = one_request(base, model, max_tokens)
        if erased is not None:
            r["erased_tokens"] = erased
        reqs.append(r)
    ttfts = [r["ttft_ms"] for r in reqs]
    tps = [r["gen_tps"] for r in reqs]
    result = {
        "model": model,
        "prompt_chars": len(PROMPT),
        "max_tokens": max_tokens,
        "protocol": protocol,
        "requests": reqs,
        # False => at least one timed decode stopped early, so gen_tps is a
        # partial-budget figure. Surfaced instead of being silently averaged in.
        "complete_decodes": all(_complete(r, max_tokens) for r in reqs),
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
