"""Topic or seed papers -> a ranked corpus (Flow step 1).

Deterministic by default: OpenAlex topic search, one hop of reference snowballing
from the strongest hits, Semantic Scholar as a top-up, then dedupe and rank. The
Librarian agent can drive the same sources through its tools, but the Flow uses
this function so a corpus is reproducible from a seed.

Ranking combines relevance to the query with *citation velocity* (citations per
year since publication) rather than raw citation count, which would always favour
older papers.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Sequence

from rapidfuzz import fuzz

from .mining.openalex import OpenAlexCache, semantic_scholar_search

STOP = set("a an the of for on in with and or to from using via by new novel towards toward "
           "approach approaches method methods model models study analysis based".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP and len(w) > 2}


def relevance(query: str, paper: dict) -> float:
    """0-1 blend of title/abstract term overlap and fuzzy title similarity."""
    q = _tokens(query)
    if not q:
        return 0.0
    title_t, abs_t = _tokens(paper.get("title")), _tokens(paper.get("abstract"))
    title_overlap = len(q & title_t) / len(q)
    abs_overlap = len(q & abs_t) / len(q)
    fuzzy = fuzz.token_set_ratio(query.lower(), (paper.get("title") or "").lower()) / 100
    return round(0.5 * title_overlap + 0.2 * abs_overlap + 0.3 * fuzzy, 4)


def velocity(paper: dict, today: date | None = None) -> float:
    """Citations per year since publication."""
    y = paper.get("year")
    c = paper.get("cited_by_count") or 0
    if not y:
        return 0.0
    age = max(1.0, ((today or date.today()).year - int(y)) + 0.5)
    return round(c / age, 2)


def _key(paper: dict) -> str:
    return (paper.get("openalex_id") or paper.get("doi")
            or re.sub(r"[^a-z0-9]", "", (paper.get("title") or "").lower())[:80])


def collect_corpus(topic: str, n: int = 25, seeds: Sequence[str] | None = None,
                   from_year: int | None = None, to_year: int | None = None,
                   snowball_from: int = 5, require_abstract: bool = True,
                   cache: OpenAlexCache | None = None, pages: int = 2,
                   use_semantic_scholar: bool = True) -> dict:
    """Build and rank a corpus. Returns ``{papers, stats}``; ``papers`` is length ``n``."""
    oa = cache or OpenAlexCache()
    pool: dict[str, dict] = {}
    stats = {"topic": topic, "from_openalex": 0, "from_snowball": 0,
             "from_semantic_scholar": 0, "from_seeds": 0}

    for p in oa.search_topic(topic, per_page=50, from_year=from_year, to_year=to_year,
                             pages=pages):
        pool.setdefault(_key(p), p | {"origin": "openalex-search"})
    stats["from_openalex"] = len(pool)

    for s in seeds or []:
        hit = oa.resolve_title(s, floor=80) if not str(s).startswith("http") else None
        if hit:
            pool.setdefault(_key(hit), hit | {"origin": "seed"})
            stats["from_seeds"] += 1

    strongest = sorted(pool.values(), key=lambda p: -(p.get("cited_by_count") or 0))[:snowball_from]
    before = len(pool)
    for p in strongest:
        if not p.get("openalex_id"):
            continue
        for r in oa.related_works(p["openalex_id"], direction="references", limit=40):
            pool.setdefault(_key(r), r | {"origin": "snowball"})
    stats["from_snowball"] = len(pool) - before

    if use_semantic_scholar and len(pool) < n * 3:
        before = len(pool)
        for p in semantic_scholar_search(topic, limit=40):
            pool.setdefault(_key(p), p | {"origin": "semantic-scholar"})
        stats["from_semantic_scholar"] = len(pool) - before

    cands = list(pool.values())
    if require_abstract:
        cands = [p for p in cands if (p.get("abstract") or "").strip()]
    if from_year:
        cands = [p for p in cands if (p.get("year") or 0) >= from_year]
    if to_year:
        cands = [p for p in cands if (p.get("year") or 9999) <= to_year]

    for p in cands:
        p["relevance"] = relevance(topic, p)
        p["velocity"] = velocity(p)
    vmax = max((p["velocity"] for p in cands), default=1.0) or 1.0
    for p in cands:
        p["rank_score"] = round(0.7 * p["relevance"] + 0.3 * (p["velocity"] / vmax), 4)
    cands.sort(key=lambda p: -p["rank_score"])

    chosen = cands[:n]
    for i, p in enumerate(chosen, 1):
        p["paper_id"] = (p.get("openalex_id") or "").rstrip("/").split("/")[-1] or f"P{i}"
    stats |= {"pool_size": len(pool), "after_filters": len(cands), "selected": len(chosen),
              "origins": {o: sum(1 for p in chosen if p.get("origin") == o)
                          for o in {p.get("origin") for p in chosen}},
              "mean_relevance": round(sum(p["relevance"] for p in chosen) / max(1, len(chosen)), 4),
              "year_range": [min((p.get("year") or 9999) for p in chosen) if chosen else None,
                             max((p.get("year") or 0) for p in chosen) if chosen else None]}
    oa.save()
    return {"papers": chosen, "stats": stats}


def save_corpus(corpus: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(corpus, indent=2))
    return p


def corpus_table(papers: Sequence[dict]) -> list[dict]:
    """Compact rows for the human approval checkpoint."""
    return [{"#": i, "paper_id": p.get("paper_id"), "year": p.get("year"),
             "title": (p.get("title") or "")[:90], "venue": (p.get("venue") or "")[:40],
             "cites": p.get("cited_by_count"), "cites/yr": p.get("velocity"),
             "relevance": p.get("relevance"), "origin": p.get("origin")}
            for i, p in enumerate(papers, 1)]
