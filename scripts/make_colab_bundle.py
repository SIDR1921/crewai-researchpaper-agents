#!/usr/bin/env python
"""Package the project for Google Colab as a single zip to drop on Drive.

Includes the code, the mined datasets and the cached arXiv LaTeX sources, so a
Colab run starts from the data this machine already produced rather than spending
hours re-mining arXiv. The sources cache matters: ``write_datasets`` rewrites
``citation_contexts.jsonl`` from scratch each run, so without the cache a re-run
of Stage 1 would re-download every paper. Left out is only what Colab can
re-download cheaply or must build itself: the virtualenv, the git history, the
SciCite/ACL-ARC corpora and the trained checkpoints.

    python scripts/make_colab_bundle.py
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Everything the notebooks need to install and run the package.
CODE = ["pyproject.toml", "README.md", "RUN_ON_COLAB.md", ".env.example",
        "src", "scripts", "notebooks", "docs", "app", "tests"]

#: Mined outputs worth carrying, so Colab can resume rather than restart.
DATA_FILES = ["rw_pairs.jsonl", "citation_contexts.jsonl", "llm_labels.jsonl",
              "gold_blind.jsonl", "gold_labelled.jsonl", "gold_key.json",
              "verifier_dev.jsonl", "multixscience_test.jsonl",
              "papers_train.jsonl", "papers_test.jsonl",
              "openalex_cache.json", "arxiv_meta_cache.json", "doi_meta_cache.json",
              "extractions_train_api.jsonl", "claim_verdicts_train_api.json"]

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", "_debug", "_jobs"}
SKIP_SUFFIX = {".pyc", ".pyo", ".tmp"}


def _add(zf: zipfile.ZipFile, path: Path, arc: str) -> int:
    if path.is_dir():
        n = 0
        for child in sorted(path.rglob("*")):
            if any(part in SKIP_DIRS for part in child.parts):
                continue
            if child.is_file() and child.suffix not in SKIP_SUFFIX:
                zf.write(child, f"{arc}/{child.relative_to(path)}")
                n += 1
        return n
    if path.is_file():
        zf.write(path, arc)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "litreview_colab.zip"))
    ap.add_argument("--with-results", action="store_true", default=True,
                    help="include results/ and figures/ so the report can be rebuilt on Colab")
    ap.add_argument("--no-results", dest="with_results", action="store_false")
    ap.add_argument("--no-data", dest="with_data", action="store_false", default=True)
    ap.add_argument("--no-sources", dest="with_sources", action="store_false", default=True,
                    help="omit data/sources (makes Stage 1 re-download every paper)")
    args = ap.parse_args()

    out = Path(args.out)
    counts: dict[str, int] = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in CODE:
            src = REPO / name
            if src.exists():
                counts[name] = _add(zf, src, f"repo/{name}")
        if args.with_data:
            n = 0
            for f in DATA_FILES:
                src = REPO / "data" / f
                if src.exists():
                    zf.write(src, f"data/{f}")
                    n += 1
            counts["data"] = n
        if args.with_data and args.with_sources:
            counts["data/sources"] = _add(zf, REPO / "data" / "sources", "data/sources")
        if args.with_results:
            for name in ("results", "figures", "report"):
                src = REPO / name
                if src.exists():
                    counts[name] = _add(zf, src, name)

    size = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size:.1f} MB)")
    for k, v in counts.items():
        print(f"  {k:12s} {v:4d} files")
    print("\nUpload this file into Google Drive at:  MyDrive/litreview/")
    print("Then open notebooks/01_mine_arxiv.ipynb in Colab and run the setup cell;")
    print("it finds the zip, unpacks it and installs the package.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
