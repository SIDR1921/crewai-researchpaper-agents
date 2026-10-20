"""Stage 1 end to end: arXiv -> parsed papers -> the two datasets, with yield stats.

The notebook and the local CLI both call :func:`run_stage1`, so there is exactly
one implementation. Every step is cached, so re-running continues where it
stopped -- which is what makes the Colab notebook survive a disconnect.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import paths
from . import arxiv, latex_mining as lm
from .openalex import OpenAlexCache

#: Default windows. The test window is deliberately later than the knowledge
#: cutoff of the evaluated models, so no system can have memorised the real
#: related-work section (see ``docs/leakage.md``).
TRAIN_WINDOW = ("202301010000", "202412312359")
TEST_WINDOW = ("202606010000", "202609302359")
CATEGORIES = ["cs.CL", "cs.IR"]
MIN_COVERAGE = 0.6
MIN_CITED = 3


def parse_corpus(papers: dict[str, list[dict]], src_root: Path | None = None,
                 progress=None) -> tuple[dict, dict]:
    """Parse every paper whose LaTeX source is cached. Returns (mined, yield counts)."""
    mined: dict[str, dict] = {}
    counts = {"listed": 0, "no_source": 0, "no_main": 0, "parse_error": 0,
              "parsed": 0, "no_related_work": 0, "with_related_work": 0}
    for split, ps in papers.items():
        it = progress(ps, split) if progress else ps
        for p in it:
            counts["listed"] += 1
            files = arxiv.fetch_source(p["arxiv_id"], root=src_root)
            if not files:
                counts["no_source"] += 1
                continue
            try:
                res = lm.mine_paper(files)
            except Exception:
                counts["parse_error"] += 1
                continue
            if res is None:
                counts["no_main"] += 1
                continue
            counts["parsed"] += 1
            if res["related_work"] is None:
                counts["no_related_work"] += 1
            else:
                counts["with_related_work"] += 1
            mined[p["arxiv_id"]] = {**p, "split": split, **res}
    return mined, counts


def write_datasets(mined: dict, oa: OpenAlexCache, data_dir: Path,
                   min_coverage: float = MIN_COVERAGE, min_cited: int = MIN_CITED) -> dict:
    """Write ``citation_contexts.jsonl`` and ``rw_pairs.jsonl``; return statistics."""
    data_dir.mkdir(parents=True, exist_ok=True)

    def cited_meta(m: dict, key: str):
        title = m["keymap"].get(key)
        return title, (oa.cache.get(title) if title else None)

    n_ctx, n_ctx_resolved = 0, 0
    with (data_dir / "citation_contexts.jsonl").open("w") as f:
        for aid, m in mined.items():
            rw_text = (m["related_work"] or {}).get("text", "")
            for c in m["all_contexts"]:
                title, meta = cited_meta(m, c["cite_key"])
                if title is None:
                    continue
                n_ctx += 1
                n_ctx_resolved += meta is not None
                f.write(json.dumps({
                    "citing_id": aid, "citing_title": m["title"],
                    "citing_date": m["published"], "split": m["split"],
                    "cite_key": c["cite_key"], "cited_title": title,
                    "cited_openalex_id": meta and meta["openalex_id"],
                    "cited_year": meta and meta["year"],
                    "cited_abstract": meta and meta["abstract"],
                    "cite_sentence": c["cite_sentence"], "context": c["context"],
                    "n_cites_in_sentence": c["n_cites_in_sentence"],
                    "in_related_work": c["cite_sentence"] in rw_text,
                }) + "\n")

    kept, dropped_cov, dropped_few = 0, 0, 0
    with (data_dir / "rw_pairs.jsonl").open("w") as f:
        for aid, m in mined.items():
            rw = m["related_work"]
            if not rw:
                continue
            if len(rw["cited_keys"]) < min_cited:
                dropped_few += 1
                continue
            cited = []
            for k in rw["cited_keys"]:
                title, meta = cited_meta(m, k)
                cited.append({"cite_key": k, "title": title,
                              "abstract": meta and meta["abstract"],
                              "year": meta and meta["year"],
                              "venue": meta and meta.get("venue"),
                              "openalex_id": meta and meta["openalex_id"]})
            coverage = sum(c["abstract"] is not None for c in cited) / len(cited)
            if coverage < min_coverage:
                dropped_cov += 1
                continue
            f.write(json.dumps({
                "arxiv_id": aid, "split": m["split"], "published": m["published"],
                "title": m["title"], "abstract": m["abstract"],
                "rw_title": rw["title"], "rw_level": rw["level"], "rw_tier": rw["tier"],
                "rw_text": rw["text"], "rw_paragraphs": rw["paragraphs"],
                "cited": cited, "abstract_coverage": round(coverage, 3),
                "source": "arxiv",
            }) + "\n")
            kept += 1
    return {"contexts_rows": n_ctx, "contexts_resolved_to_openalex": n_ctx_resolved,
            "rw_pairs_kept": kept, "rw_pairs_dropped_low_coverage": dropped_cov,
            "rw_pairs_dropped_too_few_cites": dropped_few}


def run_stage1(n_train: int, n_test: int, categories: list[str] | None = None,
               train_window: tuple[str, str] = TRAIN_WINDOW,
               test_window: tuple[str, str] = TEST_WINDOW,
               root: Path | None = None, progress=None, fetch_progress=None,
               oa_progress=None, seed: int = 0) -> dict:
    """List, download, parse, resolve and write. Returns the yield report."""
    P = paths(root).mkdirs()
    cats = categories or CATEGORIES
    papers = {
        "train": arxiv.sample_window("train", train_window, n_train, cats, seed=seed, root=P.data),
        "test": arxiv.sample_window("test", test_window, n_test, cats, seed=seed, root=P.data),
    }
    for split, ps in papers.items():
        it = fetch_progress(ps, split) if fetch_progress else ps
        for p in it:
            arxiv.fetch_source(p["arxiv_id"], root=P.sources)

    mined, counts = parse_corpus(papers, src_root=P.sources, progress=progress)

    oa = OpenAlexCache(P.data / "openalex_cache.json")
    titles = sorted({t for m in mined.values() for t in m["keymap"].values() if t})
    oa.resolve_many(titles, progress=oa_progress)
    resolved = sum(oa.cache.get(t) is not None for t in titles)
    with_abs = sum(1 for t in titles if (oa.cache.get(t) or {}).get("abstract"))

    stats = write_datasets(mined, oa, P.data)
    report = {
        "config": {"n_train": n_train, "n_test": n_test, "categories": cats,
                   "train_window": train_window, "test_window": test_window, "seed": seed},
        "yield": counts,
        "openalex": {"unique_cited_titles": len(titles), "resolved": resolved,
                     "resolved_pct": round(100 * resolved / max(1, len(titles)), 1),
                     "with_abstract": with_abs,
                     "with_abstract_pct": round(100 * with_abs / max(1, len(titles)), 1)},
        "datasets": stats,
        "windows_by_split": {
            s: {"min": min((p["published"] for p in ps), default=None),
                "max": max((p["published"] for p in ps), default=None), "n": len(ps)}
            for s, ps in papers.items()},
    }
    (P.results / "stage1_yield.json").write_text(json.dumps(report, indent=2))
    return report
