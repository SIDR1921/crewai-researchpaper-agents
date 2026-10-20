"""CrewAI tools. Each agent is given only the ones its role needs.

The tools are thin wrappers over the deterministic code in the package, so an
agent cannot do anything the pipeline cannot do without it -- and the heavy
lifting (fuzzy quote matching, NLI scoring) stays in Python where it is testable.
"""
from __future__ import annotations

import json

from crewai.tools import tool

from .config import paths
from .mining.openalex import OpenAlexCache, semantic_scholar_search

_oa: OpenAlexCache | None = None
_verifier = None


def _cache() -> OpenAlexCache:
    global _oa
    if _oa is None:
        _oa = OpenAlexCache(paths().data / "openalex_cache.json")
    return _oa


@tool("search_literature")
def search_literature(query: str, limit: int = 20) -> str:
    """Search OpenAlex (and Semantic Scholar as a fallback) for papers matching a topic
    query. Returns JSON: a list of {title, year, venue, cited_by_count, openalex_id,
    abstract_excerpt}. Use this to find candidate papers for a corpus."""
    hits = _cache().search_topic(query, per_page=min(int(limit), 50))
    if not hits:
        hits = semantic_scholar_search(query, limit=int(limit))
    return json.dumps([{
        "title": h.get("title"), "year": h.get("year"), "venue": h.get("venue"),
        "cited_by_count": h.get("cited_by_count"), "openalex_id": h.get("openalex_id"),
        "abstract_excerpt": (h.get("abstract") or "")[:300],
    } for h in hits[: int(limit)]], indent=1)


@tool("snowball_references")
def snowball_references(openalex_id: str, limit: int = 25) -> str:
    """Given an OpenAlex work id, return the papers it cites, as JSON. Use this to expand a
    corpus outward from seed papers."""
    refs = _cache().related_works(openalex_id, direction="references", limit=int(limit))
    return json.dumps([{"title": r.get("title"), "year": r.get("year"),
                        "cited_by_count": r.get("cited_by_count"),
                        "openalex_id": r.get("openalex_id")} for r in refs], indent=1)


@tool("verify_reference_exists")
def verify_reference_exists(title: str) -> str:
    """Check that a paper with this title really exists, via OpenAlex then Semantic Scholar.
    Returns JSON with found=true/false and the resolved metadata. Never cite a paper this
    reports as not found."""
    from .mining.openalex import verify_reference
    return json.dumps(verify_reference(title, _cache()), indent=1)


@tool("check_quote_and_entailment")
def check_quote_and_entailment(claim: str, passage: str, source_text: str) -> str:
    """Check whether `passage` really occurs in `source_text` and whether `claim` is entailed
    by `passage`. Returns JSON {quote_score, entail_prob, contradiction_prob, verdict}. This is
    the only authority on whether a claim is supported; do not judge it yourself."""
    global _verifier
    from .verify import Thresholds, quote_score
    th = Thresholds.load()
    q = quote_score(passage, source_text)
    if _verifier is None:
        from .verify import NLI
        _verifier = NLI()
    e, c = _verifier.score([(passage, claim)])[0]
    return json.dumps({"quote_score": round(q, 2), "entail_prob": e, "contradiction_prob": c,
                       "verdict": th.verdict(q, e)})


LIBRARIAN_TOOLS = [search_literature, snowball_references, verify_reference_exists]
VERIFIER_TOOLS = [check_quote_and_entailment]
