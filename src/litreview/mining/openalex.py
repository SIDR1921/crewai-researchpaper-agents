"""Title -> metadata resolution against OpenAlex, with an on-disk cache.

A fuzzy-similarity floor is applied to the title match so a wrong paper is
rarely attached to a cite key. Misses are cached as ``None`` so repeat runs are
free; transient errors are *not* cached, so they are retried.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import requests
from rapidfuzz import fuzz

from ..config import paths, secret

OPENALEX = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/search"
SELECT = "id,doi,title,publication_year,cited_by_count,abstract_inverted_index,authorships,primary_location,type"
MATCH_FLOOR = 88
USER_AGENT = "litreview-research (student project; contact via mailto param)"


def rebuild_abstract(inv: dict | None) -> str | None:
    """OpenAlex stores abstracts as an inverted index; put the words back in order."""
    if not inv:
        return None
    pos = [(i, w) for w, idxs in inv.items() for i in idxs]
    return " ".join(w for _, w in sorted(pos)) or None


def _venue(work: dict) -> str | None:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name")


def _authors(work: dict, limit: int = 8) -> list[str]:
    return [(a.get("author") or {}).get("display_name")
            for a in (work.get("authorships") or [])[:limit]
            if (a.get("author") or {}).get("display_name")]


def normalise(work: dict, match_score: float | None = None) -> dict:
    return {
        "openalex_id": work.get("id"),
        "title": work.get("title"),
        "year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count"),
        "doi": work.get("doi"),
        "venue": _venue(work),
        "authors": _authors(work),
        "type": work.get("type"),
        "abstract": rebuild_abstract(work.get("abstract_inverted_index")),
        "match_score": match_score,
    }


class OpenAlexCache:
    """Thread-safe JSON cache keyed by the raw query title."""

    def __init__(self, path: str | Path | None = None, mailto: str | None = None,
                 delay: float = 0.12) -> None:
        self.path = Path(path) if path else paths().data / "openalex_cache.json"
        self.mailto = mailto or secret("OPENALEX_MAILTO") or "litreview@example.com"
        self.delay = delay
        self._lock = threading.Lock()
        self.cache: dict[str, dict | None] = {}
        if self.path.exists():
            try:
                self.cache = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.cache = {}
        self._dirty = 0

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache))
            tmp.replace(self.path)
            self._dirty = 0

    # ------------------------------------------------------------------ lookups

    def resolve_title(self, title: str | None, floor: int = MATCH_FLOOR) -> dict | None:
        title = (title or "").strip()
        if len(title) < 12:
            return None
        if title in self.cache:
            return self.cache[title]
        try:
            r = requests.get(OPENALEX, timeout=30, headers={"User-Agent": USER_AGENT}, params={
                "search": title[:300], "per_page": 3, "mailto": self.mailto, "select": SELECT})
            if r.status_code == 429:
                time.sleep(5)
                return None                       # retried on the next run
            results = r.json().get("results", [])
        except Exception:
            return None                           # transient: do not cache
        best, best_score = None, -1.0
        for w in results:
            score = fuzz.token_sort_ratio(title.lower(), (w.get("title") or "").lower())
            if score >= floor and score > best_score:
                best, best_score = normalise(w, score), score
        with self._lock:
            self.cache[title] = best
            self._dirty += 1
        if self._dirty >= 100:
            self.save()
        time.sleep(self.delay)
        return best

    def resolve_many(self, titles: list[str], progress=None, workers: int = 8
                     ) -> dict[str, dict | None]:
        """Resolve a list of titles, in parallel.

        OpenAlex's "polite pool" (entered by sending a mailto) allows roughly ten
        requests a second, so a thread pool is the difference between ten minutes
        and five hours on a pilot-sized corpus. The per-request sleep is dropped
        when running in parallel, because the pool size is what bounds the rate.
        """
        todo = [t for t in titles if t and t not in self.cache]
        if not todo:
            return {t: self.cache.get(t) for t in titles}
        if workers <= 1:
            it = progress(todo) if progress else todo
            for t in it:
                self.resolve_title(t)
            self.save()
            return {t: self.cache.get(t) for t in titles}

        from concurrent.futures import ThreadPoolExecutor, as_completed
        delay, self.delay = self.delay, 0.0
        done = 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self.resolve_title, t): t for t in todo}
                it = progress(as_completed(futures)) if progress else as_completed(futures)
                for fut in it:
                    try:
                        fut.result()
                    except Exception:
                        pass
                    done += 1
                    if done % 500 == 0:
                        self.save()
        finally:
            self.delay = delay
        self.save()
        return {t: self.cache.get(t) for t in titles}

    # ---------------------------------------------------- topic / seed searches

    def search_topic(self, query: str, per_page: int = 50, from_year: int | None = None,
                     to_year: int | None = None, pages: int = 1) -> list[dict]:
        """Relevance-ranked OpenAlex search, used by the corpus builder."""
        out: list[dict] = []
        for page in range(1, pages + 1):
            params = {"search": query, "per_page": per_page, "page": page,
                      "mailto": self.mailto, "select": SELECT}
            filt = []
            if from_year:
                filt.append(f"from_publication_date:{from_year}-01-01")
            if to_year:
                filt.append(f"to_publication_date:{to_year}-12-31")
            if filt:
                params["filter"] = ",".join(filt)
            try:
                r = requests.get(OPENALEX, timeout=45, headers={"User-Agent": USER_AGENT}, params=params)
                res = r.json().get("results", [])
            except Exception:
                break
            if not res:
                break
            out += [normalise(w) for w in res]
            time.sleep(self.delay)
        return out

    def related_works(self, openalex_id: str, direction: str = "references",
                      limit: int = 50) -> list[dict]:
        """One hop of snowballing: a work's references, or works citing it."""
        wid = (openalex_id or "").rstrip("/").split("/")[-1]
        if not wid:
            return []
        key = f"__{direction}__{wid}"
        if key in self.cache:
            ids = self.cache[key] or []
        else:
            try:
                r = requests.get(f"{OPENALEX}/{wid}", timeout=30,
                                 headers={"User-Agent": USER_AGENT},
                                 params={"mailto": self.mailto, "select": "id,referenced_works"})
                ids = (r.json().get("referenced_works") or [])[:limit] \
                    if direction == "references" else []
            except Exception:
                return []
            with self._lock:
                self.cache[key] = ids
            time.sleep(self.delay)
        if direction == "citations":
            try:
                r = requests.get(OPENALEX, timeout=45, headers={"User-Agent": USER_AGENT},
                                 params={"filter": f"cites:{wid}", "per_page": min(limit, 50),
                                         "mailto": self.mailto, "select": SELECT})
                return [normalise(w) for w in r.json().get("results", [])]
            except Exception:
                return []
        out = []
        for chunk_start in range(0, len(ids), 50):
            chunk = [i.rstrip("/").split("/")[-1] for i in ids[chunk_start:chunk_start + 50]]
            try:
                r = requests.get(OPENALEX, timeout=45, headers={"User-Agent": USER_AGENT},
                                 params={"filter": f"openalex_id:{'|'.join(chunk)}",
                                         "per_page": 50, "mailto": self.mailto, "select": SELECT})
                out += [normalise(w) for w in r.json().get("results", [])]
            except Exception:
                pass
            time.sleep(self.delay)
        return out


def semantic_scholar_search(query: str, limit: int = 20) -> list[dict]:
    """Secondary source for the corpus builder. Degrades to [] when rate-limited."""
    key = secret("S2_API_KEY")
    headers = {"User-Agent": USER_AGENT} | ({"x-api-key": key} if key else {})
    try:
        r = requests.get(SEMANTIC_SCHOLAR, timeout=30, headers=headers, params={
            "query": query, "limit": limit,
            "fields": "title,abstract,year,citationCount,externalIds,venue,authors"})
        if r.status_code != 200:
            return []
        papers = r.json().get("data", []) or []
    except Exception:
        return []
    return [{
        "openalex_id": None,
        "s2_id": p.get("paperId"),
        "title": p.get("title"),
        "abstract": p.get("abstract"),
        "year": p.get("year"),
        "cited_by_count": p.get("citationCount"),
        "venue": p.get("venue"),
        "authors": [a.get("name") for a in (p.get("authors") or [])],
        "doi": (p.get("externalIds") or {}).get("DOI"),
        "arxiv_id": (p.get("externalIds") or {}).get("ArXiv"),
        "match_score": None,
    } for p in papers if p.get("title")]


def verify_reference(title: str, cache: OpenAlexCache | None = None) -> dict:
    """Check a reference really exists before it is cited in the report."""
    cache = cache or OpenAlexCache()
    hit = cache.resolve_title(title, floor=85)
    if hit:
        return {"query": title, "found": True, "source": "openalex", **hit}
    for p in semantic_scholar_search(title, limit=3):
        if fuzz.token_sort_ratio(title.lower(), (p["title"] or "").lower()) >= 85:
            return {"query": title, "found": True, "source": "semantic_scholar", **p}
    for p in arxiv_search_title(title, limit=5):
        if fuzz.token_sort_ratio(title.lower(), (p["title"] or "").lower()) >= 85:
            return {"query": title, "found": True, "source": "arxiv", **p}
    return {"query": title, "found": False, "source": None}


ARXIV_API = "https://export.arxiv.org/api/query"


def arxiv_search_title(title: str, limit: int = 5) -> list[dict]:
    """Third fallback for reference verification: the arXiv API.

    Some venue-published preprints (ICLR in particular) are indexed badly by
    OpenAlex, and Semantic Scholar rate-limits aggressively without a key.
    """
    import feedparser
    try:
        r = requests.get(ARXIV_API, timeout=40, headers={"User-Agent": USER_AGENT},
                         params={"search_query": f"all:{title[:200]}", "max_results": limit})
        feed = feedparser.parse(r.text)
    except Exception:
        return []
    out = []
    for e in feed.entries:
        t = " ".join(e.title.split())
        out.append({
            "openalex_id": None,
            "arxiv_id": e.id.split("/abs/")[-1].split("v")[0],
            "title": t, "abstract": " ".join(e.summary.split()),
            "year": int(e.published[:4]) if getattr(e, "published", "") else None,
            "venue": "arXiv preprint", "doi": getattr(e, "arxiv_doi", None),
            "authors": [a.get("name") for a in getattr(e, "authors", [])],
            "cited_by_count": None, "match_score": None,
        })
    return out
