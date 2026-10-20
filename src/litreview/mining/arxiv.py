"""arXiv listing and LaTeX source download, cached on disk.

arXiv asks for at least three seconds between requests; that is enforced here
rather than left to the caller. Every paper's extracted source files are cached
as one JSON file, and an empty dict is cached for PDF-only submissions so they
are never retried.
"""
from __future__ import annotations

import gzip
import io
import json
import random
import re
import tarfile
import time
from pathlib import Path

import feedparser
import requests

from ..config import paths

ARXIV_API = "https://export.arxiv.org/api/query"
EPRINT = "https://arxiv.org/e-print/{}"
DELAY = 3.1
KEEP_EXT = (".tex", ".bbl", ".bib", ".bst")
MAX_FILE = 2_000_000
HEADERS = {"User-Agent": "litreview-research (student project)"}

_last_call = [0.0]


def _throttle(delay: float = DELAY) -> None:
    wait = delay - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def list_arxiv(window: tuple[str, str], n: int, cats: list[str], page: int = 200,
               progress=None) -> list[dict]:
    """Papers in ``cats`` submitted inside ``window`` (``YYYYMMDDHHMM`` strings)."""
    cat_q = " OR ".join(f"cat:{c}" for c in cats)
    q = f"({cat_q}) AND submittedDate:[{window[0]} TO {window[1]}]"
    out: list[dict] = []
    start, empty_pages = 0, 0
    while len(out) < n and empty_pages < 3:
        _throttle()
        try:
            r = requests.get(ARXIV_API, timeout=90, headers=HEADERS, params={
                "search_query": q, "start": start, "max_results": page,
                "sortBy": "submittedDate", "sortOrder": "descending"})
            feed = feedparser.parse(r.text)
        except Exception:
            empty_pages += 1
            continue
        if not feed.entries:
            empty_pages += 1
            start += page
            continue
        empty_pages = 0
        for e in feed.entries:
            out.append({
                "arxiv_id": e.id.split("/abs/")[-1].split("v")[0],
                "title": " ".join(e.title.split()),
                "abstract": " ".join(e.summary.split()),
                "published": e.published[:10],
                "updated": getattr(e, "updated", e.published)[:10],
                "primary_cat": e.arxiv_primary_category["term"],
                "all_cats": [t["term"] for t in getattr(e, "tags", [])],
            })
        start += page
        if progress:
            progress(len(out))
    return out[:n]


def month_windows(window: tuple[str, str]) -> list[tuple[str, str]]:
    """Split a submission window into calendar months."""
    from datetime import datetime
    start = datetime.strptime(window[0][:8], "%Y%m%d")
    end = datetime.strptime(window[1][:8], "%Y%m%d")
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        lo = max(start, datetime(y, m, 1))
        hi_dt = datetime(ny, nm, 1)
        hi = min(end, hi_dt)
        if lo < hi_dt:
            out.append((lo.strftime("%Y%m%d") + "0000",
                        (hi.strftime("%Y%m%d") + "2359") if hi < hi_dt
                        else (hi_dt.strftime("%Y%m%d") + "0000")))
        y, m = ny, nm
    return out


def sample_window(split: str, window: tuple[str, str], n: int, cats: list[str],
                  oversample: int = 3, seed: int = 0, root: Path | None = None,
                  progress=None) -> list[dict]:
    """Cached, seeded sample spread across the whole window.

    The arXiv API returns newest-first, so simply over-fetching and sampling gives
    the *last* fortnight of a two-year window rather than the window. The window is
    therefore split into calendar months and a share drawn from each, which is what
    makes "the training window is 2023-2024" a true statement about the sample.
    """
    p = (root or paths().data) / f"papers_{split}.jsonl"
    if p.exists():
        return [json.loads(l) for l in p.open()]

    months = month_windows(window) or [window]
    per_month = max(4, -(-n * oversample // len(months)))
    rng = random.Random(seed)
    pool: list[dict] = []
    for w in months:
        got = list_arxiv(w, per_month, cats, page=min(200, per_month), progress=progress)
        rng.shuffle(got)
        pool += got[: max(1, -(-n // len(months)) * oversample)]

    seen, uniq = set(), []
    for rec in pool:
        if rec["arxiv_id"] in seen:
            continue
        seen.add(rec["arxiv_id"])
        uniq.append(rec)
    rng.shuffle(uniq)
    papers = uniq[:n]
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for rec in papers:
            f.write(json.dumps(rec) + "\n")
    return papers


def fetch_source(arxiv_id: str, root: Path | None = None) -> dict | None:
    """``{relative_path: text}`` for a paper's LaTeX files, or ``None``.

    ``None`` means "no usable LaTeX" (PDF-only submission, or an archive we could
    not read); that outcome is cached as ``{}`` so it is not retried.
    """
    src = root or paths().sources
    src.mkdir(parents=True, exist_ok=True)
    cache = src / f"{arxiv_id.replace('/', '_')}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text()) or None
        except json.JSONDecodeError:
            cache.unlink()
    _throttle()
    try:
        r = requests.get(EPRINT.format(arxiv_id), timeout=180, headers=HEADERS)
    except Exception:
        return None                              # transient: not cached
    files: dict[str, str] = {}
    blob = r.content
    if r.ok and not blob.startswith(b"%PDF"):
        try:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tar:
                for m in tar.getmembers():
                    if m.isfile() and m.name.lower().endswith(KEEP_EXT) and m.size < MAX_FILE:
                        fh = tar.extractfile(m)
                        if fh is not None:
                            files[m.name] = fh.read().decode("utf-8", "ignore")
        except (tarfile.ReadError, EOFError):
            try:                                  # single gzipped .tex file
                files["main.tex"] = gzip.decompress(blob).decode("utf-8", "ignore")
            except OSError:
                pass
    cache.write_text(json.dumps(files))
    return files or None


ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def arxiv_id_from_any(s: str) -> str | None:
    m = ARXIV_ID_RE.search(s or "")
    return m.group(1) if m else None


# --------------------------------------------------------------- metadata by id

ABS_BATCH = 100


def fetch_metadata(ids: list[str], chunk: int = ABS_BATCH, progress=None) -> dict[str, dict]:
    """``arxiv_id -> {title, abstract, year, authors, venue}`` via the API's ``id_list``.

    One request returns up to a hundred papers, which makes this far cheaper than a
    search per title and, unlike OpenAlex, costs nothing against a daily budget.
    For a corpus mined from arXiv it is also the more authoritative source: these
    are the abstracts of the exact versions being cited.
    """
    out: dict[str, dict] = {}
    clean = [i.strip() for i in ids if i and i.strip()]
    batches = [clean[i:i + chunk] for i in range(0, len(clean), chunk)]
    it = progress(batches) if progress else batches
    for batch in it:
        _throttle()
        try:
            r = requests.get(ARXIV_API, timeout=120, headers=HEADERS,
                             params={"id_list": ",".join(batch), "max_results": len(batch)})
            feed = feedparser.parse(r.text)
        except Exception:
            continue
        for e in feed.entries:
            raw = e.id.split("/abs/")[-1]
            aid = raw.split("v")[0]
            out[aid] = {
                "arxiv_id": aid,
                "openalex_id": None,
                "title": " ".join(e.title.split()),
                "abstract": " ".join(e.summary.split()),
                "year": int(e.published[:4]) if getattr(e, "published", "") else None,
                "venue": "arXiv preprint",
                "doi": getattr(e, "arxiv_doi", None),
                "authors": [a.get("name") for a in getattr(e, "authors", [])],
                "cited_by_count": None,
                "match_score": 100.0,          # matched by identifier, not by title
                "source": "arxiv",
            }
    return out
