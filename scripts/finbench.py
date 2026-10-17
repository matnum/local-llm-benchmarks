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
  2. only the `cf` formulation at prompt variant `p0` is run, not the full
     mcf/cf/gen × p0..p4 grid.

Data comes from the HF datasets-server REST API (JSON, paginated) and is cached
locally, so no parquet/pyarrow dependency is needed. stdlib + PyYAML only.

Usage:
  scripts/finbench.py --list
  scripts/finbench.py --model poro2-8b-q8_0 --pilot 150
  scripts/finbench.py --model poro2-8b-q8_0
  scripts/finbench.py --all --run-id <id>          # every model in MODELS
"""
import argparse
import collections
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORER = f"{REPO}/tools/score_ll"
MODELS_DIR = os.path.expanduser("~/.local/share/models")
DATA_DIR = os.path.expanduser("~/.cache/finbench")
RESULTS = f"{REPO}/results"

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


def _arc_c(doc):
    choices = doc["choices"]["text"]
    gold = doc["choices"]["label"].index(doc["answerKey"])
    return f"Vastaus kysymykseen {doc['question']}, on:", choices, _gold_index(gold, choices)


def _belebele_fin(doc):
    choices = [doc["mc_answer1"], doc["mc_answer2"], doc["mc_answer3"], doc["mc_answer4"]]
    prompt = (f"Tässä on teksti: {doc['flores_passage']}\n"
              f"Kysymys: {doc['question']} perustuen tekstiin.\nOikea vastaus:")
    gold = ["1", "2", "3", "4"].index(doc["correct_answer_num"])
    return prompt, choices, _gold_index(gold, choices)


def _goldenswag(doc):
    # process_docs replica: query = preprocess(label + ": " + ctx_a + " " + ctx_b.capitalize())
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    query = _preprocess_endings(doc["activity_label"] + ": " + ctx)
    choices = [_preprocess_endings(e) for e in doc["endings"]]
    prompt = f"Kerro loogisin jatko seuraavalle tekstille:\n\n{query}\n\nJatko:"
    return prompt, choices, _gold_index(int(doc["label"]), choices)


def _scandisent(doc):
    # process_docs replica: label_map maps to Finnish sentiment words
    label_map = {"positive": "positiivinen", "negative": "negatiivinen"}
    choices = [label_map["positive"], label_map["negative"]]
    prompt = f"Arvostelijan teksti: {doc['text']}\nTunne:"
    return prompt, choices, _gold_index(label_map[doc["label"]], choices)


def _sib200(doc):
    prompt = f"Päättele, mitä aihetta seuraava uutinen käsittelee. Uutinen: {doc['text']}\nAihe:"
    choices = list(doc["choices"])
    return prompt, choices, _gold_index(int(doc["answer_idx"]), choices)


TASKS = collections.OrderedDict([
    ("arc_c", {
        "repo": "TurkuNLP/finbenchv2-arc-c-fi-ht", "config": "default", "split": "test",
        "upstream": "arc_c/cf/arc_challenge_fi_cf_fbv2_p0.yaml (_arc_challenge_fi_cf_fbv2_yaml)",
        "doc_to_text": "Vastaus kysymykseen {{ question }}, on:",
        "transform": _arc_c,
    }),
    ("belebele_fin", {
        "repo": "TurkuNLP/finbenchv2-belebele-fi-og", "config": "default", "split": "test",
        "upstream": "belebele_fin/cf/belebele_fin_cf_fbv2_p0.yaml (_belebele_fin_cf_yaml)",
        "doc_to_text": "Tässä on teksti: {{flores_passage}}\nKysymys: {{question}} perustuen tekstiin.\nOikea vastaus:",
        "transform": _belebele_fin,
    }),
    ("goldenswag", {
        "repo": "TurkuNLP/finbenchv2-goldenswag-fi-ht", "config": "default", "split": "train",
        "upstream": "goldenswag/cf/goldenswag_fi_cf_fbv2_p0.yaml (_goldenswag_fi_cf_yaml + ogx_goldenswagx_utils.py)",
        "doc_to_text": "Kerro loogisin jatko seuraavalle tekstille:\n\n{{ query }}\n\nJatko:",
        "transform": _goldenswag,
    }),
    ("scandisent", {
        "repo": "TurkuNLP/finbenchv2-scandisent-fi-mini", "config": "default", "split": "test",
        "upstream": "scandisent/cf/scandisent_fi_cf_fbv2_p0.yaml (_scandisent_fi_cf_yaml + scandisent_fi_utils.py)",
        "doc_to_text": "Arvostelijan teksti: {{ query }}\nTunne:",
        "transform": _scandisent,
    }),
    ("sib200", {
        "repo": "TurkuNLP/finbenchv2-sib-200-fi-og", "config": "default", "split": "test",
        "upstream": "sib200/cf/sib200_fi_cf_fbv2_p0.yaml (_sib200_fi_cf_yaml)",
        "doc_to_text": "Päättele, mitä aihetta seuraava uutinen käsittelee. Uutinen: {{ text }}\nAihe:",
        "transform": _sib200,
    }),
])
# squad_fi and truthfulqa have no cf variant upstream (gen / mc1 / mc2 only), so
# they are out of scope for a cf-only run by construction, not by omission.


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

    spec = TASKS[task]
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


def build_items(task, rows, limit=None):
    """doc -> (item id, prompt, choices, gold index); mirrors lm-eval's doc pipeline."""
    items, skipped = [], 0
    for i, doc in enumerate(rows):
        if limit and len(items) >= limit:
            break
        try:
            prompt, choices, gold = TASKS[task]["transform"](doc)
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
def run_model(cell, model_path, run_id, tasks, pilot, data_dir, ngl, n_ctx, threads, fa):
    print(f"=== {cell} ({os.path.basename(model_path)}) ===")
    if not os.path.isfile(model_path):
        print(f"  SKIP: {model_path} not on disk")
        return None
    items, summaries, skipped = [], {}, {}
    for task in tasks:
        rows = fetch_rows(task, data_dir, limit=pilot)
        t_items, t_skipped = build_items(task, rows, limit=pilot)
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
                   "formulation": "cf", "prompt_variant": "p0",
                   "pilot_limit": pilot, "tasks": summaries}, f, indent=1, ensure_ascii=False)
    return summaries


def consolidate(run_id):
    """Roll every model of this run into a CSV + markdown table (one row per model/task)."""
    rows, models, pilot = [], [], 0
    base = f"{RESULTS}/finbench-{run_id}"
    if not os.path.isdir(base):
        return
    for cell in sorted(os.listdir(base)):
        sp = os.path.join(base, cell, "summary.json")
        if not os.path.isfile(sp):
            continue
        s = json.load(open(sp, encoding="utf-8"))
        models.append(cell)
        pilot = max(pilot, s.get("pilot_limit") or 0)
        for task, d in s["tasks"].items():
            rows.append({"run_id": run_id, "cell": cell, "task": task,
                         "n": d["n"], "acc": round(d["acc"], 4),
                         "acc_norm": round(d["acc_norm"], 4)})
    if not rows:
        return
    path = f"{RESULTS}/finbench-consolidated.csv"
    fields = ["run_id", "cell", "task", "n", "acc", "acc_norm"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tasks = sorted({r["task"] for r in rows})
    ns = sorted({r["n"] for r in rows})
    md = [f"# FIN-bench-v2 — Finnish capability (run `{run_id}`)", ""]
    if pilot:
        md += [f"> **PILOT — first {pilot} items per task only.** Not comparable with a full "
               f"run; a task's score can move several points on the full split.", ""]
    md += [f"`cf` formulation, prompt variant `p0`, scored by `tools/score_ll` (pinned "
           f"llama.cpp logits). Cells are **acc / acc_norm**; `acc` = argmax loglikelihood of "
           f"the choice text, `acc_norm` divides by the choice's character length, exactly as "
           f"`lm_eval/api/task.py` does. Items per task: "
           + (f"{ns[0]}" if len(ns) == 1 else ", ".join(str(n) for n in ns)) + ".", "",
           "| model | " + " | ".join(tasks) + " | mean acc |",
           "|---" * (len(tasks) + 2) + "|"]
    # ranked by mean acc — a capability table is read as a ranking, and the
    # alphabetical order (as cell ids happen to be named) hides exactly that.
    ranked = []
    for cell in models:
        vals = {r["task"]: r for r in rows if r["cell"] == cell}
        accs = [vals[t]["acc"] for t in tasks if t in vals]
        ranked.append((sum(accs) / len(accs), cell, vals))
    for mean, cell, vals in sorted(ranked, reverse=True):
        md.append(f"| {cell} | " + " | ".join(
            f"{vals[t]['acc']:.3f} / {vals[t]['acc_norm']:.3f}" if t in vals else "–"
            for t in tasks)
            + f" | {mean:.3f} |")
    open(f"{RESULTS}/finbench-table.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"\nwrote {path} and {RESULTS}/finbench-table.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="cell id from MODELS, or a path to a .gguf")
    ap.add_argument("--all", action="store_true", help="run every model in MODELS")
    ap.add_argument("--tasks", default="", help="comma-separated task subset (default: all)")
    ap.add_argument("--pilot", type=int, default=0, help="cap items per task (fast check)")
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
        print("tasks (cf / p0):")
        for t, spec in TASKS.items():
            print(f"  {t:<14} {spec['repo']} [{spec['split']}]")
            print(f"                 upstream: {spec['upstream']}")
        print("models:")
        for c, p in MODELS.items():
            print(f"  {c:<18} {'on disk' if os.path.isfile(p) else 'MISSING'}  {p}")
        return
    if args.consolidate_only:
        consolidate(args.run_id)
        return

    tasks = list(TASKS) if not args.tasks else [t.strip() for t in args.tasks.split(",")]
    for t in tasks:
        if t not in TASKS:
            raise SystemExit(f"unknown task {t!r}; --list shows the options")
    targets = (list(MODELS.items()) if args.all
               else [(args.model, MODELS.get(args.model, args.model))] if args.model
               else [])
    if not targets:
        raise SystemExit("pass --model <cell|path>, --all, --list or --consolidate-only")

    print(f"FIN-bench-v2 cf/p0 | run {args.run_id} | tasks: {', '.join(tasks)}"
          + (f" | PILOT {args.pilot} items/task" if args.pilot else ""))
    for cell, path in targets:
        run_model(cell, path, args.run_id, tasks, args.pilot, args.data_dir,
                  args.ngl, args.ctx, args.threads, args.fa)
    consolidate(args.run_id)


if __name__ == "__main__":
    main()
