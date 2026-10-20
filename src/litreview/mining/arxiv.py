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

ARXIV_API = "http://export.arxiv.org/api/query"
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


def sample_window(split: str, window: tuple[str, str], n: int, cats: list[str],
                  oversample: int = 4, seed: int = 0, root: Path | None = None,
                  progress=None) -> list[dict]:
    """Cached, seeded sample of a window.

    The API returns newest-first, so we over-fetch and then sample; otherwise a
    "window" would really be its last few days.
    """
    p = (root or paths().data) / f"papers_{split}.jsonl"
    if p.exists():
        return [json.loads(l) for l in p.open()]
    papers = list_arxiv(window, n * oversample, cats, progress=progress)
    random.Random(seed).shuffle(papers)
    papers = papers[:n]
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
