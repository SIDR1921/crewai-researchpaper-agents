"""Schema coercion and the Extractor's repair of common small-model mistakes."""
import pytest

from litreview.extract import MAX_INPUT_CHARS, build_input_text, build_prompt, coerce_record
from litreview.schemas import PaperRecord


def test_year_coercion_from_strings_and_junk():
    assert PaperRecord(paper_id="a", title="t", year="2024-05-01").year == 2024
    assert PaperRecord(paper_id="a", title="t", year="unknown").year is None
    assert PaperRecord(paper_id="a", title="t", year=1700).year is None


def test_string_fields_are_listified():
    r = PaperRecord(paper_id="a", title="t", methods="BERT", datasets=None, limitations="")
    assert r.methods == ["BERT"] and r.datasets == [] and r.limitations == []


def test_coerce_record_fills_ids_and_fixes_evidence_shape():
    rec = coerce_record({
        "claims": [
            {"text": "X improves Y", "type": "FINDING", "evidence": "a quoted passage"},
            {"text": "no evidence", "type": "finding"},
        ],
        "metrics": [{"name": "ROUGE-L", "value": "44.2%", "setting": "XSum"},
                    {"value": 1.0}],
    }, {"paper_id": "p1", "title": "A paper", "published": "2024-03-01"})
    assert rec.paper_id == "p1" and rec.year == 2024
    assert [c.claim_id for c in rec.claims] == ["c1"]        # claim without evidence dropped
    assert rec.claims[0].type.value == "finding"
    assert rec.claims[0].evidence.section == "unknown"
    assert rec.metrics[0].value == pytest.approx(44.2)       # "44.2%" -> 44.2
    assert len(rec.metrics) == 1                             # nameless metric dropped


def test_build_input_text_respects_priority_and_budget():
    paper = {"abstract": "A " * 200,
             "sections": {"introduction": "I " * 3000, "results": "R " * 3000,
                          "method": "M " * 3000}}
    text = build_input_text(paper)
    assert len(text) <= MAX_INPUT_CHARS + 60
    assert "[abstract]" in text and "[results]" in text
    assert "[method]" not in text          # method is not in the Extractor's budget list


def test_build_input_text_falls_back_to_abstract_only():
    text = build_input_text({"abstract": "Short but usable abstract. " * 10, "sections": {}})
    assert text.startswith("[abstract]")


def test_build_prompt_quotes_safely():
    p = build_prompt({"paper_id": "x", "title": 'A "quoted" title', "year": 2024,
                      "abstract": "z" * 300, "sections": {}})
    assert '"x"' in p and "PAPER TEXT" in p
