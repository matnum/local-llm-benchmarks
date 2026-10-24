#!/usr/bin/env python3
"""Measure Finnish retrieval quality of an embedding GGUF served by llama-server.

The axis is one question: given a Finnish question, does the model rank the
passage that answers it above 262 others? It runs the pinned llama.cpp
(`build-vulkan` = reference backend, `build` = ROCm), embeds the committed corpus
and the hand-authored queries, and writes one summary per run:

    results/embed-<stamp>-<name>/summary.json

Raw run directories are gitignored like the other axes; the published artifact is
`results/embeddings-table.md`, rendered from those summaries with `--table`.

Nomic models are trained with task prefixes and bge-m3 is not, so each model gets
its own prefixes (recorded in the summary): scoring nomic without
`search_query:`/`search_document:` would understate it and turn a
model-vs-model comparison into a protocol difference.

Usage:
    /usr/bin/python3 scripts/embed_eval.py --model ~/.local/share/models/bge-m3-f16.gguf --name bge-m3
    /usr/bin/python3 scripts/embed_eval.py --table        # rebuild the published table
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET_DIR = os.path.join(REPO, "docs", "retrieval", "fi-2026-10")
CORPUS = os.path.join(SET_DIR, "corpus.jsonl")
QUERIES = os.path.join(SET_DIR, "queries.json")
RESULTS = os.path.join(REPO, "results")
TABLE = os.path.join(RESULTS, "embeddings-table.md")

SERVERS = {
    "vulkan": os.path.expanduser("~/.local/opt/llama.cpp/build-vulkan/bin/llama-server"),
    "rocm": os.path.expanduser("~/.local/opt/llama.cpp/build/bin/llama-server"),
}
LLAMA_DIR = os.path.expanduser("~/.local/opt/llama.cpp")

# Trained-in task prefixes. Absent = the model was not trained with them.
MODEL_PREFIXES = {
    "nomic": ("search_query: ", "search_document: "),
}
METRIC_K = (1, 5, 10)


# --- metric primitives (pure, tested without a GPU) -------------------------


def normalize(vec):
    norm = math.sqrt(sum(value * value for value in vec))
    return [value / norm for value in vec] if norm else list(vec)


def cosine(left, right):
    """Cosine similarity of two already-normalised vectors."""
    return sum(a * b for a, b in zip(left, right))


def rank_passages(query_vec, passage_vecs):
    """Passage indices, best first. Ties break on index so runs are deterministic."""
    scores = [cosine(query_vec, vec) for vec in passage_vecs]
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def recall_at_k(ranked, relevant, k):
    """Fraction of the relevant passages found in the top k."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked, relevant, k):
    """1.0 when any relevant passage is in the top k."""
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr_at_k(ranked, relevant, k):
    """Reciprocal rank of the first relevant passage, 0 when none is in the top k."""
    for position, index in enumerate(ranked[:k], start=1):
        if index in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    """Binary-gain nDCG@k against the ideal ordering of the same qrels."""
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, index in enumerate(ranked[:k])
        if index in relevant
    )
    ideal = sum(1.0 / math.log2(position + 2) for position in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def prefixes_for(name, query_prefix=None, doc_prefix=None):
    """Task prefixes for a model: explicit flag > known family > none."""
    default = ("", "")
    for family, pair in MODEL_PREFIXES.items():
        if family in name.lower():
            default = pair
    return (
        default[0] if query_prefix is None else query_prefix,
        default[1] if doc_prefix is None else doc_prefix,
    )


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_embeddings(payload, expected):
    """Vectors from an OpenAI-style /v1/embeddings response, in input order.

    llama.cpp does not promise response order, so entries are placed by their
    `index`; a short response is an error rather than a silent misalignment.
    """
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected:
        raise ValueError(
            f"expected {expected} embeddings, got "
            f"{len(data) if isinstance(data, list) else type(data).__name__}"
        )
    vectors = [None] * expected
    for entry in data:
        vectors[int(entry["index"])] = entry["embedding"]
    if any(vector is None for vector in vectors):
        raise ValueError("embedding response has a gap in its indices")
    return vectors


# --- io ---------------------------------------------------------------------


def load_corpus(path=CORPUS):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_queries(path=QUERIES):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def llama_commit():
    try:
        return subprocess.run(
            ["git", "-C", LLAMA_DIR, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


class Server:
    """A llama-server in embedding mode, on a private port."""

    def __init__(self, binary, model, port, ctx=2048):
        self.url = f"http://127.0.0.1:{port}"
        self.startup_s = None
        self.proc = subprocess.Popen(
            [binary, "-m", model, "--embedding", "--host", "127.0.0.1",
             "--port", str(port), "--ctx-size", str(ctx), "--threads", "8"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def wait_ready(self, timeout=240):
        started = time.time()
        while time.time() - started < timeout:
            if self.proc.poll() is not None:
                stream = self.proc.stderr
                err = (stream.read() if stream else b"") or b""
                raise RuntimeError(
                    f"llama-server exited early:\n{err.decode('utf-8', 'replace')[-1500:]}"
                )
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                    if response.status == 200:
                        self.startup_s = time.time() - started
                        return self.startup_s
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("llama-server did not become healthy in time")

    def embed(self, texts, batch_size=16, timeout=300):
        vectors = []
        for batch in batches(texts, batch_size):
            body = json.dumps({"input": batch, "model": "embed"}).encode("utf-8")
            request = urllib.request.Request(
                f"{self.url}/v1/embeddings",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            vectors.extend(parse_embeddings(payload, len(batch)))
        return vectors

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# --- evaluation -------------------------------------------------------------


def evaluate(name, model, backend, server_bin, port, batch_size, corpus, query_set,
             query_prefix, doc_prefix):
    model = os.path.expanduser(model)
    if not os.path.isfile(model):
        # A literal "~" reaches llama-server as a path component when the caller
        # passes it unexpanded, and the server only reports "model loading error".
        raise SystemExit(f"model not found: {model}")
    corpus_pr = [doc_prefix + row["text"] for row in corpus]
    queries_pr = [query_prefix + item["query"] for item in query_set["queries"]]

    server = Server(server_bin, model, port)
    try:
        startup_s = server.wait_ready()
        started = time.time()
        passage_vecs = [normalize(vec) for vec in server.embed(corpus_pr, batch_size)]
        embed_s = time.time() - started
        query_vecs = [normalize(vec) for vec in server.embed(queries_pr, batch_size)]
    finally:
        server.close()

    index_of = {row["id"]: position for position, row in enumerate(corpus)}
    metrics = {f"hit@{k}": [] for k in METRIC_K}
    metrics.update({f"recall@{k}": [] for k in METRIC_K})
    metrics["mrr@10"] = []
    metrics["ndcg@10"] = []
    per_query = []
    for item, query_vec in zip(query_set["queries"], query_vecs):
        relevant = {index_of[pid] for pid in item["qrels"]}
        ranked = rank_passages(query_vec, passage_vecs)
        row = {"id": item["id"], "query": item["query"], "first_relevant_rank": None}
        for k in METRIC_K:
            metrics[f"hit@{k}"].append(hit_at_k(ranked, relevant, k))
            metrics[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
        metrics["mrr@10"].append(mrr_at_k(ranked, relevant, 10))
        metrics["ndcg@10"].append(ndcg_at_k(ranked, relevant, 10))
        for position, index in enumerate(ranked, start=1):
            if index in relevant:
                row["first_relevant_rank"] = position
                break
        per_query.append(row)

    summary = {
        "model": name,
        "model_path": model,
        "model_sha256": sha256(os.path.expanduser(model)),
        "model_bytes": os.path.getsize(os.path.expanduser(model)),
        "backend": backend,
        "llama_cpp": llama_commit(),
        "prefixes": {"query": query_prefix, "document": doc_prefix},
        "set": query_set["set"],
        "corpus": {"passages": len(corpus), "sha256": sha256(CORPUS)},
        "queries": {"count": len(query_set["queries"]), "sha256": sha256(QUERIES)},
        "timings": {
            "startup_s": round(startup_s, 2),
            "passages_embedded": len(corpus),
            "embed_s": round(embed_s, 2),
            "passages_per_s": round(len(corpus) / embed_s, 1) if embed_s else None,
        },
        "metrics": {key: sum(values) / len(values) for key, values in metrics.items()},
        "per_query": per_query,
        "measured": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    stamp = time.strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(RESULTS, f"embed-{stamp}-{name}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"{name}: {out}")
    print(render_row(summary))
    return summary


def render_row(summary):
    metrics = summary["metrics"]
    return (
        f"  {summary['model']:<12} hit@1 {metrics['hit@1']:.3f}  hit@5 {metrics['hit@5']:.3f}  "
        f"hit@10 {metrics['hit@10']:.3f}  MRR@10 {metrics['mrr@10']:.3f}  "
        f"nDCG@10 {metrics['ndcg@10']:.3f}  {summary['timings']['passages_per_s']} pass/s  "
        f"({summary['backend']}, startup {summary['timings']['startup_s']} s)"
    )


def paired_counts(left, right):
    """Per-query comparison of two summaries: where each put the first relevant passage.

    Averages hide the shape of a 30-query comparison; the paired count is what says
    whether one model wins across the set or just by a couple of large margins.
    """
    by_id = {row["id"]: row for row in right["per_query"]}
    wins = {"left": 0, "right": 0, "tie": 0}
    for row in left["per_query"]:
        other = by_id.get(row["id"])
        if other is None:
            continue
        mine, theirs = row["first_relevant_rank"] or 10**6, other["first_relevant_rank"] or 10**6
        if mine < theirs:
            wins["left"] += 1
        elif mine > theirs:
            wins["right"] += 1
        else:
            wins["tie"] += 1
    return wins


def compare(first, second):
    wins = paired_counts(first, second)
    return (
        f"{first['model']} ({first['backend']}) vs {second['model']} ({second['backend']}): "
        f"{wins['left']} queries better, {wins['right']} worse, {wins['tie']} equal "
        f"on the rank of the first relevant passage"
    )


HEADER = (
    "| Model | Backend | hit@1 | hit@5 | hit@10 | MRR@10 | nDCG@10 | Passages/s | llama.cpp |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def render_table(summaries):
    """Published table: one row per (model, backend), newest run winning."""
    latest = {}
    for summary in summaries:
        latest[(summary["model"], summary["backend"])] = summary  # oldest-first
    lines = [HEADER]
    for key in sorted(latest):
        s = latest[key]
        m = s["metrics"]
        lines.append(
            f"| {s['model']} | {s['backend']} | {m['hit@1']:.3f} | {m['hit@5']:.3f} | "
            f"{m['hit@10']:.3f} | {m['mrr@10']:.3f} | {m['ndcg@10']:.3f} | "
            f"{s['timings']['passages_per_s']} | {s['llama_cpp']} |\n"
        )
    return "".join(lines)


def load_summaries():
    summaries = []
    for entry in sorted(os.listdir(RESULTS)):
        if not entry.startswith("embed-"):
            continue
        path = os.path.join(RESULTS, entry, "summary.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                summaries.append(json.load(fh))
    return summaries


def _pick(latest, spec):
    """A summary by ``name`` or ``name@backend``; refuses an ambiguous spec."""
    if "@" in spec:
        name, backend = spec.split("@", 1)
        key = (name, backend)
        if key not in latest:
            raise SystemExit(f"no run for {spec!r}")
        return latest[key]
    candidates = [s for (model, _), s in latest.items() if model == spec]
    if not candidates:
        raise SystemExit(f"no run for {spec!r}")
    if len(candidates) > 1:
        raise SystemExit(f"{spec!r} ran on several backends; use {spec}@vulkan or {spec}@rocm")
    return candidates[0]


def write_table():
    summaries = load_summaries()
    if not summaries:
        raise SystemExit("no results/embed-*/summary.json runs to publish")
    table = render_table(summaries)
    with open(TABLE, "w", encoding="utf-8") as fh:
        fh.write(table)
    print(table)
    print(f"wrote {TABLE}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", help="path to the embedding GGUF")
    ap.add_argument("--name", help="model name used in the summary and table")
    ap.add_argument("--backend", default="vulkan", choices=sorted(SERVERS))
    ap.add_argument("--server", help="override the llama-server binary")
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--query-prefix", default=None)
    ap.add_argument("--doc-prefix", default=None)
    ap.add_argument("--table", action="store_true", help="rebuild results/embeddings-table.md")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="paired per-query comparison of two model names (latest run each)",
    )
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--queries", default=QUERIES)
    args = ap.parse_args(argv)

    if args.table:
        write_table()
        return 0
    if args.compare:
        latest = {}
        for summary in load_summaries():
            latest[(summary["model"], summary["backend"])] = summary
        picked = [_pick(latest, spec) for spec in args.compare]
        print(compare(*picked))
        return 0
    if not args.model or not args.name:
        ap.error("--model and --name are required unless --table is used")

    query_prefix, doc_prefix = prefixes_for(args.name, args.query_prefix, args.doc_prefix)
    evaluate(
        name=args.name,
        model=args.model,
        backend=args.backend,
        server_bin=args.server or SERVERS[args.backend],
        port=args.port,
        batch_size=args.batch_size,
        corpus=load_corpus(args.corpus),
        query_set=load_queries(args.queries),
        query_prefix=query_prefix,
        doc_prefix=doc_prefix,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
