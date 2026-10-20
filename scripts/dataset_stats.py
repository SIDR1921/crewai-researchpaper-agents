#!/usr/bin/env python
"""Descriptive statistics for the mined datasets, written to results/.

Separate from the mining run so statistics can be recomputed without re-mining,
and so every number in Chapter 4's dataset table has a file behind it.

    python scripts/dataset_stats.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from litreview.config import paths


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _hist(values, bins: int = 18) -> dict:
    v = [x for x in values if x is not None]
    if not v:
        return {}
    counts, edges = np.histogram(v, bins=bins)
    return {"counts": counts.tolist(), "edges": [round(float(e), 3) for e in edges],
            "centers": [round(float((edges[i] + edges[i + 1]) / 2), 3)
                        for i in range(len(edges) - 1)]}


def _summary(values) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}
    return {"n": len(v), "min": round(float(v[0]), 3),
            "p25": round(float(np.percentile(v, 25)), 3),
            "median": round(float(np.median(v)), 3),
            "mean": round(float(np.mean(v)), 3),
            "p75": round(float(np.percentile(v, 75)), 3),
            "max": round(float(v[-1]), 3)}


def main() -> int:
    P = paths()
    rw = _read(P.data / "rw_pairs.jsonl")
    ctx = _read(P.data / "citation_contexts.jsonl")
    mx = _read(P.data / "multixscience_test.jsonl")

    rep: dict = {"rw_pairs": {}, "citation_contexts": {}, "multixscience": {}}

    for name, rows in (("rw_pairs", rw), ("multixscience", mx)):
        if not rows:
            rep[name] = {"available": False}
            continue
        words = [len((r.get("rw_text") or "").split()) for r in rows]
        ncited = [len(r.get("cited") or []) for r in rows]
        cov = [r.get("abstract_coverage") for r in rows]
        sents = [len([s for s in (r.get("rw_text") or "").split(". ") if len(s) > 20])
                 for r in rows]
        rep[name] = {
            "available": True, "n_papers": len(rows),
            "by_split": dict(Counter(r.get("split") for r in rows)),
            "rw_words": _summary(words), "n_cited": _summary(ncited),
            "abstract_coverage": _summary(cov), "rw_sentences": _summary(sents),
            "rw_level": dict(Counter(r.get("rw_level") for r in rows)),
            "rw_title_examples": sorted({(r.get("rw_title") or "")[:40] for r in rows})[:12],
            "date_range": {
                "min": min((r.get("published") for r in rows if r.get("published")), default=None),
                "max": max((r.get("published") for r in rows if r.get("published")), default=None)},
        }
        if name == "rw_pairs":
            rep["histograms"] = {"rw_words": _hist(words), "n_cited": _hist(ncited),
                                 "abstract_coverage": _hist(cov, bins=12)}

    if ctx:
        rep["citation_contexts"] = {
            "available": True, "n_rows": len(ctx),
            "n_citing_papers": len({c.get("citing_id") for c in ctx}),
            "n_cited_titles": len({c.get("cited_title") for c in ctx}),
            "resolved_to_openalex": sum(1 for c in ctx if c.get("cited_openalex_id")),
            "with_cited_abstract": sum(1 for c in ctx if c.get("cited_abstract")),
            "in_related_work": sum(1 for c in ctx if c.get("in_related_work")),
            "by_split": dict(Counter(c.get("split") for c in ctx)),
            "cites_per_sentence": _summary([c.get("n_cites_in_sentence") for c in ctx]),
            "context_words": _summary([len((c.get("context") or "").split()) for c in ctx]),
            "contexts_per_citing_paper": _summary(
                list(Counter(c.get("citing_id") for c in ctx).values())),
        }
    else:
        rep["citation_contexts"] = {"available": False}

    # leakage check, recorded rather than asserted, so the report can state it
    train_max = max((r.get("published") for r in rw if r.get("split") == "train"
                     and r.get("published")), default=None)
    test_min = min((r.get("published") for r in rw if r.get("split") == "test"
                    and r.get("published")), default=None)
    rep["time_split"] = {"train_latest": train_max, "test_earliest": test_min,
                         "disjoint": bool(train_max and test_min and test_min > train_max)}

    out = P.results / "stage1_dataset_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "histograms"}, indent=2))
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
