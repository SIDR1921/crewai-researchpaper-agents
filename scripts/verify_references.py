#!/usr/bin/env python
"""Verify every reference before it is cited, against OpenAlex then Semantic Scholar.

Reads ``docs/references_candidates.txt`` (one title per line, ``#`` for comments,
an optional ``| group`` suffix) and writes ``results/references_verified.json``
with the resolved title, authors, venue, year and DOI for each hit, and a clear
``found: false`` for each miss. The report builder refuses to cite anything that
is not in this file with ``found: true``.

    python scripts/verify_references.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from litreview.config import paths
from litreview.mining.openalex import OpenAlexCache, verify_reference


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="docs/references_candidates.txt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    P = paths()
    src = Path(args.candidates)
    if not src.exists():
        print(f"{src} missing")
        return 1
    entries = []
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        title, _, group = line.partition("|")
        entries.append((title.strip(), (group.strip() or "journal-conference")))

    oa = OpenAlexCache(P.data / "openalex_cache.json")
    out = []
    for title, group in tqdm(entries, desc="verify"):
        rec = verify_reference(title, oa)
        rec["group"] = group
        out.append(rec)
    oa.save()

    found = [r for r in out if r["found"]]
    report = {"n_candidates": len(out), "n_found": len(found),
              "n_missing": len(out) - len(found),
              "by_group": {g: sum(1 for r in found if r["group"] == g)
                           for g in {r["group"] for r in out}},
              "references": out}
    dest = Path(args.out or (P.results / "references_verified.json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(f"{len(found)}/{len(out)} verified -> {dest}")
    for r in out:
        if not r["found"]:
            print("  NOT FOUND:", r["query"][:90])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
