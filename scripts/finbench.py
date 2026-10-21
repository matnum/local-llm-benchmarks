#!/usr/bin/env python3
"""FIN-bench-v2 (cf formulation, prompt variant p0) scorer for the local models.

Runs the Finnish capability axis: for each model, score every task's choices by
the summed loglikelihood of the choice text as a continuation, then report the
harness's two metrics.

Provenance — every prompt, dataset and transform below is copied from
github.com/LumiOpen/lm-evaluation-harness @ branch `finbench_v2`
(`lm_eval/tasks/finbench_v2/<family>/cf/*`), verified 2026-10-03; the two
`process_docs` helpers are replicas of the upstream `*_utils.py` modules and are
noted per task. Metrics follow `lm_eval/api/task.py` for
`OUTPUT_TYPE == "multiple_choice"`:

    completion_len = [float(len(choice_string)) for choice in choices]
    pred       = argmax(lls)                 -> acc
    pred_norm  = argmax(lls / completion_len) -> acc_norm

Note the two deliberate deviations from a straight `lm_eval` run, both forced by
the pinned llama.cpp build (which refuses `echo`, so no OpenAI-compatible server
can score continuations) and both documented in docs/RUNNING.md:
  1. scoring is done by `tools/score_ll` (libllama, logits) rather than by
     lm-evaluation-harness;
  2. the grid is run one (formulation, variant) pair per invocation — `--formulation
     cf|mcf --variant p0..p4` — rather than by lm-eval's task lookup. The prompts
     are the upstream `doc_to_text` strings, vendored from the `finbench_v2` branch
     into docs/finbench/upstream-<sha>.json and rendered with Jinja2, so a prompt is
     never paraphrased. Choice sets and gold indices follow the upstream base yaml
     (`doc_to_choice` / `doc_to_target`) for that formulation: note that `mcf` is
     not just a different prompt — for belebele_fin the choices become the letters
     A-D, while arc_c keeps the full option text (upstream does both).

Data comes from the HF datasets-server REST API (JSON, paginated) and is cached
locally, so no parquet/pyarrow dependency is needed. stdlib + PyYAML + Jinja2.

Usage:
  scripts/finbench.py --list
  scripts/finbench.py --model poro2-8b-q8_0 --pilot 150
  scripts/finbench.py --model poro2-8b-q8_0
  scripts/finbench.py --all --run-id <id>          # every model in MODELS
"""
import argparse
import collections
import csv
import functools
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import jinja2
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORER = f"{REPO}/tools/score_ll"
MODELS_DIR = os.path.expanduser("~/.local/share/models")
DATA_DIR = os.path.expanduser("~/.cache/finbench")
RESULTS = f"{REPO}/results"

FORMULATIONS = ("cf", "mcf")
VARIANTS = ("p0", "p1", "p2", "p3", "p4")

MODELS = {
    # cell id -> gguf, mirroring matrix.yaml (the Finnish question's four cells)
    "poro2-8b-q8_0": f"{MODELS_DIR}/Llama-Poro-2-8B-Instruct.Q8_0.gguf",
    # Quant-matched twin of qwen3-8b-q4_k_m, so the same-size Finnish-vs-English
    # comparison stops being confounded by quantisation.
    "poro2-8b-q4_k_m": f"{MODELS_DIR}/Llama-Poro-2-8B-Instruct.Q4_K_M.gguf",
    "poro2-70b-q4_k_m": f"{MODELS_DIR}/Llama-Poro-2-70B-Instruct.Q4_K_M.gguf",
    "eurollm-9b-q8_0": f"{MODELS_DIR}/EuroLLM-9B-Instruct-Q8_0.gguf",
    "qwen3-8b-q4_k_m": f"{MODELS_DIR}/Qwen3-8B-Q4_K_M.gguf",
}

# --------------------------------------------------------------------------
# tasks: prompt templates, dataset coordinates and doc transforms
# --------------------------------------------------------------------------
def _preprocess_endings(text):
    """Replica of ogx_goldenswagx_utils.preprocess()."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _gold_index(gold, choices):
    if isinstance(gold, str):
        return choices.index(gold) if gold in choices else -100
    return int(gold)


def _upstream_snapshot():
    """The vendored upstream task templates (see scripts/fetch_finbench_upstream.py)."""
    paths = sorted(glob.glob(os.path.join(REPO, "docs", "finbench", "upstream-*.json")))
    if len(paths) != 1:
        raise SystemExit("expected exactly one docs/finbench/upstream-*.json snapshot; "
                         "run scripts/fetch_finbench_upstream.py")
    with open(paths[0], encoding="utf-8") as f:
        return json.load(f)


UPSTREAM = _upstream_snapshot()
UPSTREAM_COMMIT = UPSTREAM["commit"]
# StrictUndefined: a template field the namespace does not provide must fail loudly
# rather than silently render an empty string into a prompt.
_ENV = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)


@functools.lru_cache(maxsize=None)
def _template(family, formulation, variant):
    """(upstream path, upstream task name, compiled template) for one grid cell."""
    prefix = f"{family}/{formulation}/"
    rel = next((r for r in sorted(UPSTREAM["files"])
                if r.startswith(prefix) and r.endswith(f"_{variant}.yaml")
                and not os.path.basename(r).startswith("_")), None)
    if rel is None:
        raise SystemExit(f"no upstream template for {family}/{formulation}/{variant}")
    spec = yaml.safe_load(UPSTREAM["files"][rel]["text"])
    # lm_eval prepends `description` to doc_to_text even at zero few-shot.
    text = (spec.get("description") or "") + spec["doc_to_text"]
    return rel, spec.get("task", ""), _ENV.from_string(text)


def _goldenswag_fields(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    query = _preprocess_endings(doc["activity_label"] + ": " + ctx)
    endings = [_preprocess_endings(e) for e in doc["endings"]]
    return query, endings


def _goldenswag_gold(doc):
    """Label is an int index upstream; tolerate a numeric string, never crash."""
    try:
        return int(doc["label"])
    except (TypeError, ValueError):
        return -100


def _scandisent_fields(doc):
    label_map = {"positive": "positiivinen", "negative": "negatiivinen"}
    return doc["text"], [label_map["positive"], label_map["negative"]], label_map[doc["label"]]


# Per-family data plumbing. `namespace` is what the upstream Jinja template sees
# (the process_docs-transformed doc where upstream has one), `choices` follows the
# formulation's doc_to_choice, `gold` is the doc_to_target index.
FAMILIES = collections.OrderedDict([
    ("arc_c", {
        "repo": "TurkuNLP/finbenchv2-arc-c-fi-ht", "config": "default", "split": "test",
        "namespace": lambda doc, formulation: doc,
        "choices": lambda doc, formulation: doc["choices"]["text"],
        "gold": lambda doc, choices: _gold_index(
            doc["choices"]["label"].index(doc["answerKey"]), choices),
    }),
    ("belebele_fin", {
        "repo": "TurkuNLP/finbenchv2-belebele-fi-og", "config": "default", "split": "test",
        "namespace": lambda doc, formulation: doc,
        "choices": lambda doc, formulation: (
            [doc["mc_answer1"], doc["mc_answer2"], doc["mc_answer3"], doc["mc_answer4"]]
            if formulation == "cf" else ["A", "B", "C", "D"]),
        "gold": lambda doc, choices: _gold_index(
            ["1", "2", "3", "4"].index(doc["correct_answer_num"]), choices),
    }),
    ("goldenswag", {
        "repo": "TurkuNLP/finbenchv2-goldenswag-fi-ht", "config": "default", "split": "train",
        "namespace": lambda doc, formulation: dict(
            zip(("query", "choices"), _goldenswag_fields(doc))),
        "choices": lambda doc, formulation: _goldenswag_fields(doc)[1],
        "gold": lambda doc, choices: _gold_index(_goldenswag_gold(doc), choices),
    }),
    ("scandisent", {
        "repo": "TurkuNLP/finbenchv2-scandisent-fi-mini", "config": "default", "split": "test",
        "namespace": lambda doc, formulation: {
            "query": _scandisent_fields(doc)[0], "choices": _scandisent_fields(doc)[1]},
        "choices": lambda doc, formulation: _scandisent_fields(doc)[1],
        "gold": lambda doc, choices: _gold_index(_scandisent_fields(doc)[2], choices),
    }),
    ("sib200", {
        "repo": "TurkuNLP/finbenchv2-sib-200-fi-og", "config": "default", "split": "test",
        "namespace": lambda doc, formulation: {
            "text": doc["text"], "choices": list(doc["choices"])},
        "choices": lambda doc, formulation: list(doc["choices"]),
        "gold": lambda doc, choices: _gold_index(int(doc["answer_idx"]), choices),
    }),
])
# squad_fi and truthfulqa have no cf/mcf variant upstream (gen / mc1 / mc2 only), so
# they are out of scope for this axis by construction, not by omission.


def item_for(family, doc, formulation="cf", variant="p0"):
    """doc -> (prompt, choices, gold), rendered from the upstream template."""
    spec = FAMILIES[family]
    _, _, template = _template(family, formulation, variant)
    prompt = template.render(**spec["namespace"](doc, formulation))
    choices = spec["choices"](doc, formulation)
    return prompt, choices, spec["gold"](doc, choices)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def fetch_rows(task, data_dir, limit=None, quiet=False):
    """Rows for a task, cached as JSONL. Paginates the datasets-server API.

    Resumable by design: this API fails transiently when a run bursts ~70
    requests at it (observed on the first attempt at sib200), and restarting a
    7k-row fetch from zero each time is wasteful. Each batch is appended to a
    `.partial` file as it arrives and promoted to the real cache only when the
    split is complete, so a rerun resumes at the row it reached.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, f"{task}.jsonl")
    part = cache + ".partial"
    if os.path.isfile(cache):
        rows = [json.loads(l) for l in open(cache, encoding="utf-8")]
        if not quiet:
            print(f"  {task}: {len(rows)} rows (cached)")
        return rows[:limit] if limit else rows

    spec = FAMILIES[task]
    rows = [json.loads(l) for l in open(part, encoding="utf-8")] if os.path.isfile(part) else []
    offset, total = len(rows), None
    if rows and not quiet:
        print(f"  {task}: resuming a partial fetch at row {offset}")
    out = open(part, "a", encoding="utf-8")
    backoff = 2
    while total is None or offset < total:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(spec['repo'])}"
               f"&config={urllib.parse.quote(spec['config'])}&split={spec['split']}"
               f"&offset={offset}&length=100")
        d = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    d = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code == 404 and offset == 0:
                    out.close()
                    raise SystemExit(f"{task}: dataset/config/split not found: {url}")
            except Exception:
                pass
            time.sleep(min(backoff, 30))
            backoff *= 2
        if d is None:
            out.close()
            raise SystemExit(f"{task}: API failed at offset {offset} after 6 attempts; "
                             f"partial kept at {part} — rerun to resume")
        backoff = 2
        total = d.get("num_rows_total", total)
        batch = [x["row"] for x in d.get("rows", [])]
        if not batch:
            break
        for row in batch:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        rows.extend(batch)
        offset += len(batch)
        time.sleep(0.3)                      # be polite to a public API
        if limit and len(rows) >= limit:
            break
    out.close()
    if limit:
        # A capped fetch is a sample: promoting it would silently shrink every
        # later run of this task (the cache is keyed by task alone).
        if not quiet:
            print(f"  {task}: {len(rows)} rows fetched (pilot, not cached)")
        return rows[:limit]
    os.replace(part, cache)
    if not quiet:
        print(f"  {task}: {len(rows)} rows fetched -> {cache}")
    return rows


def build_items(task, rows, limit=None, formulation="cf", variant="p0"):
    """doc -> (item id, prompt, choices, gold index); mirrors lm-eval's doc pipeline."""
    items, skipped = [], 0
    for i, doc in enumerate(rows):
        if limit and len(items) >= limit:
            break
        try:
            prompt, choices, gold = item_for(task, doc, formulation, variant)
        except (KeyError, ValueError, AttributeError) as e:
            skipped += 1
            print(f"    {task}: skipped row {i}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        items.append((f"{task}::{i}", prompt, choices, gold))
    return items, skipped


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def esc(s):
    """JSON-escape a field's content without the enclosing quotes (protocol)."""
    return json.dumps(s, ensure_ascii=False)[1:-1]


def score(model_path, items, ngl=99, n_ctx=4096, threads=16, fa=False, quiet=False):
    """Run tools/score_ll over all items; returns {id: [ll, ...]}, {id: [ntok, ...]}."""
    if not os.path.isfile(SCORER):
        raise SystemExit(f"{SCORER} missing — build it (see its header comment)")
    tsv = os.path.join(DATA_DIR, "_items.tsv")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(tsv, "w", encoding="utf-8") as f:
        for iid, prompt, choices, _ in items:
            f.write("\t".join([iid, esc(prompt)] + [esc(c) for c in choices]) + "\n")

    cmd = [SCORER, "-m", model_path, "-ngl", str(ngl), "-c", str(n_ctx), "-t", str(threads)]
    if fa:
        cmd.append("--fa")
    if not quiet:
        print(f"  scoring {len(items)} items with {os.path.basename(model_path)} ...")
    t0 = time.time()
    with open(tsv, encoding="utf-8") as fin, open(tsv + ".out", "w", encoding="utf-8") as fout:
        rc = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=None).returncode
    if rc != 0:
        raise SystemExit(f"score_ll exited {rc}")
    lls, ntoks = {}, {}
    for line in open(tsv + ".out", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        lls[parts[0]] = [float(parts[1 + 2 * i]) for i in range((len(parts) - 1) // 2)]
        ntoks[parts[0]] = [int(parts[2 + 2 * i]) for i in range((len(parts) - 1) // 2)]
    if not quiet:
        print(f"  scored in {time.time() - t0:.0f}s")
    return lls, ntoks


def evaluate(task, items, lls, ntoks=None):
    """acc / acc_norm exactly as lm-eval's multiple_choice branch computes them."""
    n = correct = correct_norm = 0
    gold_errors = overflow = 0
    per_item = []
    for iid, prompt, choices, gold in items:
        scores = lls.get(iid)
        if scores is None:
            continue
        if ntoks is not None and any(t < 0 for t in ntoks.get(iid, [])):
            overflow += 1                      # did not fit the context; excluded
            continue
        if gold < 0 or gold >= len(choices):
            gold_errors += 1
        pred = max(range(len(scores)), key=lambda i: scores[i])
        # completion_len is len() of the choice STRING (characters, not tokens/bytes)
        norms = [scores[i] / max(len(choices[i]), 1) for i in range(len(scores))]
        pred_norm = max(range(len(norms)), key=lambda i: norms[i])
        n += 1
        correct += int(pred == gold)
        correct_norm += int(pred_norm == gold)
        per_item.append({"id": iid, "gold": gold, "pred": pred, "pred_norm": pred_norm,
                         "lls": scores, "chars": [len(c) for c in choices]})
    return {"n": n, "acc": correct / n if n else 0.0,
            "acc_norm": correct_norm / n if n else 0.0,
            "gold_index_errors": gold_errors, "context_overflow": overflow,
            "items": per_item}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run_model(cell, model_path, run_id, tasks, pilot, data_dir, ngl, n_ctx, threads, fa,
              formulation="cf", variant="p0"):
    print(f"=== {cell} ({os.path.basename(model_path)}) ===")
    if not os.path.isfile(model_path):
        print(f"  SKIP: {model_path} not on disk")
        return None
    items, summaries, skipped = [], {}, {}
    for task in tasks:
        rows = fetch_rows(task, data_dir, limit=pilot)
        t_items, t_skipped = build_items(task, rows, limit=pilot,
                                         formulation=formulation, variant=variant)
        skipped[task] = t_skipped
        items.extend(t_items)
        print(f"  {task}: {len(t_items)} items"
              + (f" ({t_skipped} skipped)" if t_skipped else ""))
    lls, ntoks = score(model_path, items, ngl=ngl, n_ctx=n_ctx, threads=threads, fa=fa)
    for task in tasks:
        t_items = [it for it in items if it[0].startswith(task + "::")]
        res = evaluate(task, t_items, lls, ntoks)
        summaries[task] = {k: v for k, v in res.items() if k != "items"}
        summaries[task]["skipped"] = skipped[task]
        outdir = f"{RESULTS}/finbench-{run_id}/{cell}"
        os.makedirs(outdir, exist_ok=True)
        with open(f"{outdir}/{task}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print(f"    {task:<14} n={res['n']:<5} acc={res['acc']:.4f} acc_norm={res['acc_norm']:.4f}")
    outdir = f"{RESULTS}/finbench-{run_id}/{cell}"
    with open(f"{outdir}/summary.json", "w", encoding="utf-8") as f:
        json.dump({"cell": cell, "model": os.path.basename(model_path), "run_id": run_id,
                   "formulation": formulation, "prompt_variant": variant,
                   "upstream_commit": UPSTREAM_COMMIT,
                   "pilot_limit": pilot, "tasks": summaries}, f, indent=1, ensure_ascii=False)
    return summaries


def _collect_full_runs(results_dir):
    """Rows from every full run on disk: (run_id, formulation, variant, cell, task, ...).

    Pilots are skipped outright — they score the first N items per task and must
    never be published next to full-split numbers.
    """
    rows = []
    for base in sorted(glob.glob(os.path.join(results_dir, "finbench-*"))):
        if not os.path.isdir(base):
            continue
        for cell in sorted(os.listdir(base)):
            sp = os.path.join(base, cell, "summary.json")
            if not os.path.isfile(sp):
                continue
            with open(sp, encoding="utf-8") as f:
                s = json.load(f)
            if (s.get("pilot_limit") or 0) > 0:
                continue
            run_id = s.get("run_id") or os.path.basename(base)[len("finbench-"):]
            formulation = s.get("formulation", "cf")      # older runs predate the stamp
            variant = s.get("prompt_variant", "p0")
            for task, d in s["tasks"].items():
                rows.append({"run_id": run_id, "formulation": formulation,
                             "variant": variant, "cell": cell, "task": task,
                             "n": d["n"], "acc": round(d["acc"], 4),
                             "acc_norm": round(d["acc_norm"], 4), "_raw": d["acc"]})
    return rows


def consolidate(_run_id=None):
    """Rebuild the tracked FIN-bench rollups from every full run on disk.

    One table per (formulation, variant) pair and one CSV keyed by
    run_id/formulation/variant/cell/task, so publishing a second pair adds rows
    instead of replacing the first. `_run_id` is accepted for CLI compatibility
    and ignored.
    """
    rows = _collect_full_runs(RESULTS)
    if not rows:
        print("no full FIN-bench runs on disk — tracked rollups left untouched")
        return
    path = f"{RESULTS}/finbench-consolidated.csv"
    md_path = f"{RESULTS}/finbench-table.md"
    fields = ["run_id", "formulation", "variant", "cell", "task", "n", "acc", "acc_norm"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["formulation"], r["variant"],
                                                r["run_id"], r["cell"], r["task"])))
    tasks = sorted({r["task"] for r in rows})
    ns = sorted({r["n"] for r in rows})
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[(r["formulation"], r["variant"])].append(r)
    md = ["# FIN-bench-v2 — Finnish capability", "",
          "Scored by `tools/score_ll` (pinned llama.cpp logits) from the upstream "
          "templates vendored in `docs/finbench/`. Cells are **acc / acc_norm**; `acc` = "
          "argmax loglikelihood of the choice text, `acc_norm` divides by the choice's "
          "character length, exactly as `lm_eval/api/task.py` does. Items per task: "
          + (f"{ns[0]}" if len(ns) == 1 else ", ".join(str(n) for n in ns))
          + ". Pilots are excluded.", ""]
    for (formulation, variant), combo_rows in sorted(grouped.items()):
        run_ids = sorted({r["run_id"] for r in combo_rows})
        md += [f"## `{formulation}` / `{variant}`", "",
               f"Run `{', '.join(run_ids)}`.", "",
               "| model | " + " | ".join(tasks) + " | mean acc |",
               "|---" * (len(tasks) + 2) + "|"]
        by_cell = collections.defaultdict(dict)
        for r in combo_rows:
            by_cell[r["cell"]][r["task"]] = r
        ranked = []
        for cell, vals in by_cell.items():
            # average the unrounded accuracies (the CSV stores 4-decimal values)
            accs = [vals[t]["_raw"] for t in tasks if t in vals]
            ranked.append((sum(accs) / len(accs), cell, vals))
        for mean, cell, vals in sorted(ranked, reverse=True):
            md.append(f"| {cell} | " + " | ".join(
                f"{vals[t]['acc']:.3f} / {vals[t]['acc_norm']:.3f}" if t in vals else "–"
                for t in tasks) + f" | {mean:.3f} |")
        md.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md).rstrip() + "\n")
    print(f"wrote {path} and {md_path} "
          f"({len(grouped)} formulation/variant pairs, "
          f"{len({r['run_id'] for r in rows})} runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="cell id from MODELS, or a path to a .gguf")
    ap.add_argument("--all", action="store_true", help="run every model in MODELS")
    ap.add_argument("--tasks", default="", help="comma-separated task subset (default: all)")
    ap.add_argument("--pilot", type=int, default=0, help="cap items per task (fast check)")
    ap.add_argument("--formulation", choices=FORMULATIONS, default="cf",
                    help="FIN-bench formulation (default cf)")
    ap.add_argument("--variant", choices=VARIANTS, default="p0",
                    help="prompt variant (default p0)")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M"))
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--fa", action="store_true", help="flash attention (off: matches the "
                                                      "no-FA loglikelihood numerics)")
    ap.add_argument("--consolidate-only", action="store_true")
    ap.add_argument("--list", action="store_true", help="list tasks and models, exit")
    args = ap.parse_args()

    if args.list:
        print(f"FIN-bench-v2 grid from {UPSTREAM['source']} @ {UPSTREAM_COMMIT[:12]} "
              f"({os.path.basename(sorted(glob.glob(os.path.join(REPO, 'docs', 'finbench', 'upstream-*.json')))[0])})")
        for family, spec in FAMILIES.items():
            print(f"  {family:<14} {spec['repo']} [{spec['split']}]")
            for formulation in FORMULATIONS:
                cells = []
                for variant in VARIANTS:
                    rel, task, _ = _template(family, formulation, variant)
                    cells.append(task or variant)
                print(f"    {formulation:<4} " + ", ".join(cells))
        print("models:")
        for c, p in MODELS.items():
            print(f"  {c:<18} {'on disk' if os.path.isfile(p) else 'MISSING'}  {p}")
        return
    if args.consolidate_only:
        consolidate(args.run_id)
        return

    tasks = list(FAMILIES) if not args.tasks else [t.strip() for t in args.tasks.split(",")]
    for t in tasks:
        if t not in FAMILIES:
            raise SystemExit(f"unknown task {t!r}; --list shows the options")
    targets = (list(MODELS.items()) if args.all
               else [(args.model, MODELS.get(args.model, args.model))] if args.model
               else [])
    if not targets:
        raise SystemExit("pass --model <cell|path>, --all, --list or --consolidate-only")

    print(f"FIN-bench-v2 {args.formulation}/{args.variant} | run {args.run_id} | "
          f"tasks: {', '.join(tasks)}"
          + (f" | PILOT {args.pilot} items/task" if args.pilot else ""))
    for cell, path in targets:
        run_model(cell, path, args.run_id, tasks, args.pilot, args.data_dir,
                  args.ngl, args.ctx, args.threads, args.fa,
                  formulation=args.formulation, variant=args.variant)
    consolidate(args.run_id)


if __name__ == "__main__":
    main()
