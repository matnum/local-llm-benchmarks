#!/usr/bin/env python3
"""Vendor the upstream FIN-bench-v2 task templates this driver replicates.

The Finnish axis follows `LumiOpen/lm-evaluation-harness` (branch `finbench_v2`)
rather than paraphrasing it: the prompt for every (family, formulation, variant)
is the upstream `doc_to_text`, rendered with Jinja2. This script snapshots the
templates into `docs/finbench/upstream-<sha>.json` (path -> sha256 + text) so the
driver and its tests never need the network.

Usage:
    /usr/bin/python3 scripts/fetch_finbench_upstream.py [--branch finbench_v2]
"""
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "docs", "finbench")
API = "https://api.github.com/repos/LumiOpen/lm-evaluation-harness"
RAW = "https://raw.githubusercontent.com/LumiOpen/lm-evaluation-harness"

FAMILIES = ("arc_c", "belebele_fin", "goldenswag", "scandisent", "sib200")
FORMULATIONS = ("cf", "mcf")
VARIANTS = ("p0", "p1", "p2", "p3", "p4")


def _get(url, as_json=True, tries=4):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read()
            return json.loads(body) if as_json else body.decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise SystemExit(f"not found upstream: {url}")
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    raise SystemExit(f"failed to fetch {url}")


def _list(path, branch):
    d = _get(f"{API}/contents/{path}?ref={branch}")
    return [x["name"] for x in d]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", default="finbench_v2")
    args = ap.parse_args()

    branch = args.branch
    sha = _get(f"{API}/branches/{branch}")["commit"]["sha"]
    base = f"lm_eval/tasks/finbench_v2"
    paths = []
    for family in FAMILIES:
        for formulation in FORMULATIONS:
            d = f"{base}/{family}/{formulation}"
            for name in _list(d, branch):
                # variant yamls + the base/include yaml + process_docs helpers
                if name.endswith((".yaml", ".yml", ".py")) and not name.startswith("_group"):
                    paths.append((family, formulation, f"{d}/{name}"))

    files = {}
    for family, formulation, path in paths:
        text = _get(f"{RAW}/{sha}/{path}", as_json=False)
        files[os.path.relpath(path, base)] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"upstream-{sha[:12]}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"branch": branch, "commit": sha,
                   "source": "LumiOpen/lm-evaluation-harness",
                   "fetched": time.strftime("%Y-%m-%d"), "files": files},
                  f, indent=1, sort_keys=True)
    print(f"wrote {out} ({len(files)} files, commit {sha[:12]})")


if __name__ == "__main__":
    main()
