"""Tests for the bugs the first pilot run exposed.

Each of these was a silent failure: the data looked plausible and the numbers were
wrong, which is exactly the class of bug a results-driven report cannot tolerate.
"""
import json

import pytest

from litreview.mining import multixscience
from litreview.mining.arxiv import month_windows
from litreview.mining.openalex import OpenAlexCache


# ------------------------------- the window collapsed to its last fortnight

def test_month_windows_covers_a_two_year_span():
    ws = month_windows(("202301010000", "202412312359"))
    assert len(ws) == 24
    assert ws[0][0].startswith("20230101") and ws[-1][0].startswith("20241201")


def test_month_windows_is_contiguous_and_ordered():
    ws = month_windows(("202606010000", "202609302359"))
    assert len(ws) == 4
    for a, b in zip(ws, ws[1:]):
        assert a[1][:8] == b[0][:8], "months must join without a gap"
        assert a[0] < b[0]


def test_month_windows_handles_a_window_inside_one_month():
    ws = month_windows(("202605100000", "202605202359"))
    assert len(ws) == 1 and ws[0][0].startswith("20260510")


# --------------------- a rate-limited request must not be cached as a miss

class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {"results": []}
        self.headers = headers or {}

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._payload


def test_rate_limited_search_is_not_cached(tmp_path, monkeypatch):
    oa = OpenAlexCache(tmp_path / "cache.json")
    monkeypatch.setattr("litreview.mining.openalex.requests.get",
                        lambda *a, **k: _Resp(429, headers={"Retry-After": "0"}))
    monkeypatch.setattr("litreview.mining.openalex.time.sleep", lambda *_: None)
    assert oa.resolve_title("A perfectly ordinary paper title here") is None
    assert oa.cache == {}, "a 429 must never become a permanent cache miss"


def test_genuine_miss_is_cached(tmp_path, monkeypatch):
    oa = OpenAlexCache(tmp_path / "cache.json")
    monkeypatch.setattr("litreview.mining.openalex.requests.get",
                        lambda *a, **k: _Resp(200, {"results": []}))
    monkeypatch.setattr("litreview.mining.openalex.time.sleep", lambda *_: None)
    title = "A paper that genuinely does not exist anywhere"
    assert oa.resolve_title(title) is None
    assert oa.cache == {title: None}, "a real miss should be cached so it is not retried"


def test_match_below_the_floor_is_rejected(tmp_path, monkeypatch):
    oa = OpenAlexCache(tmp_path / "cache.json")
    payload = {"results": [{"id": "W1", "title": "Something else altogether",
                            "publication_year": 2020}]}
    monkeypatch.setattr("litreview.mining.openalex.requests.get",
                        lambda *a, **k: _Resp(200, payload))
    monkeypatch.setattr("litreview.mining.openalex.time.sleep", lambda *_: None)
    assert oa.resolve_title("Structural scaffolds for citation intent classification") is None


def test_good_match_is_normalised(tmp_path, monkeypatch):
    oa = OpenAlexCache(tmp_path / "cache.json")
    payload = {"results": [{
        "id": "https://openalex.org/W42", "title": "SciBERT: A Pretrained Language Model",
        "publication_year": 2019, "cited_by_count": 100,
        "abstract_inverted_index": {"A": [0], "pretrained": [1], "model": [2]},
        "authorships": [{"author": {"display_name": "Iz Beltagy"}}],
        "primary_location": {"source": {"display_name": "EMNLP"}}}]}
    monkeypatch.setattr("litreview.mining.openalex.requests.get",
                        lambda *a, **k: _Resp(200, payload))
    monkeypatch.setattr("litreview.mining.openalex.time.sleep", lambda *_: None)
    hit = oa.resolve_title("SciBERT: A Pretrained Language Model")
    assert hit["year"] == 2019 and hit["venue"] == "EMNLP"
    assert hit["abstract"] == "A pretrained model"
    assert hit["authors"] == ["Iz Beltagy"] and hit["match_score"] >= 88


# ------------------------------------------- Multi-XScience schema conversion

def test_multixscience_converts_into_the_rw_pairs_schema(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps([
        {"aid": "a1", "abstract": "we do things",
         "related_work": "Earlier work @cite_1 did X. Other work @cite_2 did Y.",
         "ref_abstract": {"cite_1": {"abstract": "the first cited abstract"},
                          "cite_2": {"abstract": "the second cited abstract"},
                          "cite_3": {"abstract": "the third cited abstract"}}},
        {"aid": "too_few", "abstract": "x", "related_work": "y",
         "ref_abstract": {"cite_1": {"abstract": "only one"}}},
        {"aid": "low_cov", "abstract": "x", "related_work": "y",
         "ref_abstract": {f"c{i}": {"abstract": ""} for i in range(4)}},
    ]))
    rows = multixscience.to_rw_pairs(raw, split_name="test")
    assert [r["arxiv_id"] for r in rows] == ["a1"], "filters must drop the other two"
    r = rows[0]
    assert r["source"] == "multi-xscience" and r["split"] == "test"
    assert len(r["cited"]) == 3 and r["abstract_coverage"] == 1.0
    assert set(r) >= {"arxiv_id", "rw_text", "cited", "abstract_coverage", "rw_paragraphs"}


# -------------------------------- identifier-first resolution (DOI / arXiv id)

def test_parse_identifiers_from_bib_bbl_and_inlined():
    from litreview.mining.latex_mining import parse_identifiers
    files = {
        "r.bib": "@article{a, title={X}, doi={10.18653/v1/2022.acl-demo.9}, year={2022}}\n"
                 "@misc{b, title={Y}, eprint={2204.01691}, archivePrefix={arXiv}}\n"
                 "@article{c, title={Z}, journal={Nature}}\n",
        "r.bbl": r"\bibitem{d} A. Author. \newblock Some title. \newblock arXiv:2111.10952.",
    }
    ids = parse_identifiers(files)
    assert ids["a"]["doi"] == "10.18653/v1/2022.acl-demo.9"
    assert ids["b"]["arxiv_id"] == "2204.01691"
    assert ids["d"]["arxiv_id"] == "2111.10952"
    assert "c" not in ids, "an entry with no identifier must not appear"


def test_resolver_prefers_an_arxiv_identifier_over_a_title_guess():
    from litreview.mining.pipeline import Resolver

    class _Cache:
        cache = {"A fuzzy title": {"abstract": "from openalex", "openalex_id": "W1"}}

    paper = {"keymap": {"k1": "A fuzzy title", "k2": "Another title"},
             "identifiers": {"k1": {"arxiv_id": "2204.01691"}}}
    r = Resolver(_Cache(), {"2204.01691": {"title": "Exact arXiv title",
                                           "abstract": "from arxiv", "source": "arxiv"}})
    title, meta = r(paper, "k1")
    assert meta["abstract"] == "from arxiv", "identifier match must win"
    title2, meta2 = r(paper, "k2")
    assert meta2 is None, "no identifier and no cached title -> unresolved"


def test_arxiv_ids_for_collects_only_related_work_citations():
    from litreview.mining.pipeline import arxiv_ids_for
    mined = {"p1": {"related_work": {"cited_keys": ["k1", "k2"]},
                    "identifiers": {"k1": {"arxiv_id": "1111.1111"},
                                    "k2": {"doi": "10.1/x"},
                                    "k3": {"arxiv_id": "9999.9999"}}},
             "p2": {"related_work": None, "identifiers": {"z": {"arxiv_id": "2222.2222"}}}}
    assert arxiv_ids_for(mined) == ["1111.1111"]
