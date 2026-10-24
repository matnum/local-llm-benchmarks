#!/usr/bin/env python3
"""Build the Finnish retrieval set the embeddings axis is measured on.

The axis answers one question — does bge-m3 or nomic-embed-text retrieve a
Finnish passage better — so the set has to be Finnish, business-document shaped
and committed, not fetched at run time. This script derives it from Finnish
Wikipedia (CC BY-SA 4.0) and writes:

    docs/retrieval/<set>/corpus.jsonl   one passage per line: id, title, section, text
    docs/retrieval/<set>/manifest.json  article titles + sha256 of each extract,
                                        licence, date, passage size bounds

Queries and relevance labels are **hand-authored** in `queries.json` (one entry
per question: the question, the answer string that proves the passage answers it,
and the passage ids). A test re-checks every label against the committed corpus,
so a query can never drift from the text it claims to be answerable from.

Usage:
    /usr/bin/python3 scripts/make_retrieval_set.py [--set fi-2026-10] [--out-dir docs/retrieval]
"""
import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://fi.wikipedia.org/w/api.php"
LICENSE = "CC BY-SA 4.0"
UA = "local-llm-benchmarks/1.0 (retrieval-set builder; contact: repo owner)"

# SMB-document topics: tax, bookkeeping, invoicing, employment, contracts,
# privacy. Chosen so a passage answers the kind of question an on-prem assistant
# for a Finnish small business would be asked.
ARTICLES = (
    "Arvonlisävero",
    "Kirjanpito",
    "Kirjanpitolaki",
    "Tilinpäätös",
    "Lasku",
    "Laskutus",
    "Perintä",
    "Osakeyhtiö",
    "Toiminimi",
    "Yrittäjä",
    "Työsopimus",
    "Työturvallisuus",
    "Työeläke",
    "Ennakonpidätys",
    "Verohallinto",
    "Kuitti",
    "Sähköinen allekirjoitus",
    "Tietosuoja",
    "Yleinen tietosuoja-asetus",
    "Sopimus",
    "Tilintarkastus",
    "Kassavirta",
    "Budjetti",
    "Palkka",
    "Lomaraha",
    "Vahingonkorvaus",
)

# Reference apparatus, not prose: a passage from here would be a false gold.
DROP_SECTIONS = {
    "lähteet",
    "viitteet",
    "viitteet ja lähteet",
    "kirjallisuus",
    "aiheesta muualla",
    "katso myös",
    "ulkoiset linkit",
    "huomautukset",
    "lähdekirjallisuus",
}

SECTION_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1$")
MIN_PARAGRAPH = 60
MAX_PASSAGE = 800


def _get(url, tries=6):
    """GET JSON, honouring Retry-After on 429; None when it never succeeds.

    Wikimedia returns 429 with a Retry-After header when requests come too fast,
    so a fixed short backoff just burns the retries.
    """
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise SystemExit(f"not found: {url}")
            wait = exc.headers.get("retry-after")
            wait = float(wait) if wait else 2.0 * (attempt + 1)
            time.sleep(min(wait + 0.5, 60))
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return None


def slug(title):
    """ASCII path-safe id stem: arvonlisavero."""
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")


def fetch_extracts(titles):
    """Title -> plaintext extract; missing titles come back as ''.

    One request per title on purpose: the API lowers ``exlimit`` to 1 for
    whole-article extracts (only ``exintro`` allows batches), so a multi-title
    request returns a single article and silently omits the rest — which reads as
    "every article is missing".
    """
    extracts = {}
    for title in titles:
        query = urllib.parse.urlencode(
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "format": "json",
                "redirects": "1",
                "exlimit": "1",
                "titles": title,
            }
        )
        data = _get(f"{API}?{query}")
        if data is None:
            print(f"  skip (rate-limited or unreachable): {title}")
            continue
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            resolved = page.get("title", "")
            if "missing" in page or not page.get("extract"):
                print(f"  skip (missing or empty): {resolved or title}")
                break
            if resolved != title:
                print(f"  note: redirect resolved {title!r} -> {resolved!r}")
            extracts[resolved] = page["extract"]
        time.sleep(1.0)  # be polite to the API
    return extracts


def sections(extract):
    """Split an extract into (section, paragraphs), dropping reference sections."""
    current = "Johdanto"
    dropped = False
    paragraphs = []
    for raw_line in extract.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = SECTION_RE.match(line)
        if match:
            current = match.group(2).strip()
            dropped = current.lower() in DROP_SECTIONS
            continue
        if dropped:
            continue
        paragraphs.append((current, line))
    return paragraphs


def split_long(text):
    """Hard-bound one over-long paragraph, preferring sentence boundaries."""
    pieces, start = [], 0
    while len(text) - start > MAX_PASSAGE:
        window = text[start : start + MAX_PASSAGE]
        cut = window.rfind(". ")
        cut = cut + 1 if cut > MAX_PASSAGE * 0.5 else MAX_PASSAGE
        pieces.append(text[start : start + cut].strip())
        start += cut
    pieces.append(text[start:].strip())
    return [p for p in pieces if p]


def passages(title, extract):
    """Merge paragraphs into passages of at most MAX_PASSAGE characters."""
    out, current_section, current, buffer = [], None, [], []
    for section, paragraph in sections(extract):
        if len(paragraph) < MIN_PARAGRAPH:
            continue
        if section != current_section and current:
            out.append((current_section, " ".join(current)))
            current = []
        current_section = section
        for piece in split_long(paragraph):
            candidate = (" ".join(current + [piece])).strip()
            if current and len(candidate) > MAX_PASSAGE:
                out.append((current_section, " ".join(current)))
                current = [piece]
            else:
                current.append(piece)
    if current:
        out.append((current_section, " ".join(current)))
    return [(section, text) for section, text in out if text.strip()]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--set", default="fi-2026-10", help="set directory name")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "docs", "retrieval"))
    args = ap.parse_args()

    extracts = fetch_extracts(ARTICLES)
    if not extracts:
        raise SystemExit("no articles fetched; refusing to write an empty corpus")

    out_dir = os.path.join(args.out_dir, args.set)
    os.makedirs(out_dir, exist_ok=True)
    records, article_meta = [], []
    for title in sorted(extracts):
        stem = slug(title)
        article_passages = passages(title, extracts[title])
        for index, (section, text) in enumerate(article_passages):
            records.append(
                {"id": f"{stem}#{index:03d}", "title": title, "section": section, "text": text}
            )
        article_meta.append(
            {
                "title": title,
                "url": "https://fi.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                "chars": len(extracts[title]),
                "sha256": hashlib.sha256(extracts[title].encode("utf-8")).hexdigest(),
                "passages": len(article_passages),
            }
        )

    with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "set": args.set,
        "source": "https://fi.wikipedia.org",
        "license": LICENSE,
        "fetched": time.strftime("%Y-%m-%d"),
        "builder": "scripts/make_retrieval_set.py",
        "passage_chars": [min(len(r["text"]) for r in records), max(len(r["text"]) for r in records)],
        "passages": len(records),
        "articles": article_meta,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False, sort_keys=True)

    longest = max(len(r["text"]) for r in records)
    print(
        f"wrote {len(records)} passages from {len(article_meta)} articles "
        f"to {out_dir} (longest {longest} chars)"
    )


if __name__ == "__main__":
    main()
