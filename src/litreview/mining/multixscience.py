"""Multi-XScience loaded into the same schema as ``rw_pairs.jsonl``.

Source: Lu, Dong and Charlin (EMNLP 2020), https://github.com/yaolu/Multi-XScience.
Each example is a paper's related-work paragraph plus the abstracts of the papers
it cites, which is exactly our benchmark task, so it serves as an external check
that the numbers are not an artefact of our own mining.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

#: The repository ships the splits gzipped; the plain .json paths 404.
RAW = ("https://raw.githubusercontent.com/yaolu/Multi-XScience/master/data/"
       "{split}.json.gz")
SPLITS = {"train": "train", "val": "val", "test": "test"}


def download(split: str, dest: Path) -> Path:
    """Cache one split, decompressing the repository's gzipped JSON."""
    import gzip

    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"multixscience_{split}_raw.json"
    if out.exists() and out.stat().st_size > 1000:
        return out
    r = requests.get(RAW.format(split=SPLITS[split]), timeout=300)
    r.raise_for_status()
    blob = r.content
    if blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    out.write_bytes(blob)
    return out


def to_rw_pairs(raw_path: Path, split_name: str = "test", limit: int | None = None,
                min_cited: int = 3, min_coverage: float = 0.6) -> list[dict]:
    """Convert to our ``rw_pairs`` schema, applying the same filters as Stage 1."""
    records = json.loads(Path(raw_path).read_text())
    out = []
    for r in records:
        refs = r.get("ref_abstract") or {}
        cited = [{"cite_key": k, "title": None,
                  "abstract": (v.get("abstract") or "").strip() or None,
                  "year": None, "venue": None, "openalex_id": None}
                 for k, v in refs.items()]
        if len(cited) < min_cited:
            continue
        coverage = sum(c["abstract"] is not None for c in cited) / len(cited)
        if coverage < min_coverage:
            continue
        out.append({
            "arxiv_id": r.get("aid") or r.get("paper_id") or "",
            "split": split_name, "published": None,
            "title": None, "abstract": (r.get("abstract") or "").strip(),
            "rw_title": "Related Work", "rw_level": "section", "rw_tier": 0,
            "rw_text": (r.get("related_work") or "").strip(),
            "rw_paragraphs": [(r.get("related_work") or "").strip()],
            "cited": cited, "abstract_coverage": round(coverage, 3),
            "source": "multi-xscience",
        })
        if limit and len(out) >= limit:
            break
    return out


def build(dest: Path, split: str = "test", limit: int | None = None) -> dict:
    """Download and write ``multixscience_<split>.jsonl``; return statistics."""
    raw = download(split, dest)
    rows = to_rw_pairs(raw, split_name=split, limit=limit)
    out = dest / f"multixscience_{split}.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = len(json.loads(raw.read_text()))
    return {"split": split, "raw_examples": total, "kept": len(rows), "path": str(out),
            "median_cited": sorted(len(r["cited"]) for r in rows)[len(rows) // 2] if rows else 0}
