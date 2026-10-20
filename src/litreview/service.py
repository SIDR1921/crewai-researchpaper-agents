"""Read-only views over everything the pipeline has produced.

One module so the web app and the MCP connector show the same numbers as the
report: all three read ``results/*.json`` and ``data/*.jsonl`` through here, and
none of them computes anything of its own. A stage that has not run is reported
as absent rather than faked, exactly as in the report builder.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config import SETTINGS, paths


def _jsonl(path: Path, limit: int | None = None, offset: int = 0) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit and len(out) >= limit:
                break
    return out


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------- pipeline

STAGES = [
    {"id": "mine", "n": 1, "name": "Mine arXiv",
     "blurb": "Related-work sections and citation contexts from LaTeX sources",
     "result": "stage1_yield.json"},
    {"id": "label", "n": 2, "name": "Label relations",
     "blurb": "Seven-class citation relations, labelled by the configured LLM",
     "result": "stage2_labelling.json"},
    {"id": "classify", "n": 2, "name": "Fine-tune SciBERT",
     "blurb": "Component A: the citation-relation classifier",
     "result": "stage2_benchmark.json"},
    {"id": "extract", "n": 3, "name": "Extract claims",
     "blurb": "Structured records whose every claim quotes its evidence",
     "result": "stage3_extract_train_api.json"},
    {"id": "verify", "n": 3, "name": "Verify claims",
     "blurb": "Quote match plus NLI entailment, on calibrated thresholds",
     "result": "stage3_verify_train_api.json"},
    {"id": "flow", "n": "4-5", "name": "Run the Flow",
     "blurb": "Corpus, map, gaps and a grounded review, end to end",
     "result": "flow_summary.json"},
    {"id": "distill", "n": 6, "name": "Distil the extractor",
     "blurb": "Component B: QLoRA on Verifier-approved extractions",
     "result": "stage6_distill.json"},
    {"id": "eval", "n": 7, "name": "Evaluate S1-S4",
     "blurb": "Against human related-work sections, with bootstrap CIs",
     "result": "eval_arxiv.json"},
]


def _headline(stage_id: str, P) -> list[dict]:
    """A couple of real numbers per stage, or [] when it has not run."""
    g = lambda name: _json(P.results / name) or {}
    if stage_id == "mine":
        d = g("stage1_yield.json")
        if not d:
            return []
        y, ds = d.get("yield", {}), d.get("datasets", {})
        return [{"label": "papers listed", "value": y.get("listed")},
                {"label": "with related work", "value": y.get("with_related_work")},
                {"label": "benchmark examples", "value": ds.get("rw_pairs_kept")},
                {"label": "citation contexts", "value": ds.get("contexts_rows")}]
    if stage_id == "label":
        d = g("stage2_labelling.json")
        if not d:
            n = _count_lines(paths().data / "llm_labels.jsonl")
            return [{"label": "labelled so far", "value": n}] if n else []
        return [{"label": "contexts labelled", "value": d.get("n_labelled_total")}]
    if stage_id == "classify":
        d = g("stage2_benchmark.json")
        if not d:
            return []
        out = []
        for k, nice in (("scibert", "SciBERT macro-F1"), ("tfidf_lr", "TF-IDF macro-F1"),
                        ("zero_shot_llm", "zero-shot macro-F1")):
            if d.get(k):
                out.append({"label": nice, "value": round(d[k]["macro_f1"], 3)})
        return out
    if stage_id == "extract":
        d = g("stage3_extract_train_api.json")
        if not d:
            return []
        return [{"label": "papers extracted", "value": d.get("ok")},
                {"label": "claims / paper", "value": d.get("claims_per_paper")},
                {"label": "JSON valid", "value": _pct(d.get("json_validity_rate"))}]
    if stage_id == "verify":
        d = g("stage3_verify_train_api.json")
        if not d:
            return []
        r = d.get("verdict_rates", {})
        return [{"label": "claims checked", "value": d.get("n_claims")},
                {"label": "supported", "value": _pct(r.get("supported"))},
                {"label": "unsupported", "value": _pct(r.get("unsupported"))}]
    if stage_id == "flow":
        d = g("flow_summary.json")
        if not d:
            return []
        fa = (d.get("faithfulness") or {}).get("after_revision") or {}
        return [{"label": "papers in corpus", "value": d.get("n_records")},
                {"label": "contradictions", "value": d.get("n_contradictions")},
                {"label": "faithfulness", "value": _pct(fa.get("faithfulness"))}]
    if stage_id == "distill":
        d = g("stage6_distill.json")
        if not d:
            return []
        return [{"label": "distilled JSON valid",
                 "value": _pct((d.get("distilled") or {}).get("json_validity_rate"))}]
    if stage_id == "eval":
        d = g("eval_arxiv.json")
        if not d:
            return []
        aggs = d.get("aggregates", {})
        out = [{"label": "examples", "value": (d.get("benchmark") or {}).get("n_examples")}]
        for s in ("S1", "S3"):
            m = ((aggs.get(s) or {}).get("faithfulness") or {}).get("mean")
            if m is not None:
                out.append({"label": f"{s} faithfulness", "value": round(m, 3)})
        return out
    return []


def _pct(x) -> str | None:
    return None if x is None else f"{100 * float(x):.0f}%"


def pipeline_status() -> dict:
    """Which stages have run, with a couple of real numbers each."""
    P = paths()
    stages = []
    for st in STAGES:
        done = (P.results / st["result"]).exists()
        stages.append({**st, "done": done, "headline": _headline(st["id"], P)})
    return {
        "stages": stages,
        "done": sum(1 for s in stages if s["done"]),
        "total": len(stages),
        "provider": SETTINGS.describe(),
        "root": str(P.root),
    }


# -------------------------------------------------------------------- datasets

def dataset_summary() -> dict:
    P = paths()
    stats = _json(P.results / "stage1_dataset_stats.json") or {}
    return {
        "rw_pairs": _count_lines(P.data / "rw_pairs.jsonl"),
        "contexts": _count_lines(P.data / "citation_contexts.jsonl"),
        "labels": _count_lines(P.data / "llm_labels.jsonl"),
        "multixscience": _count_lines(P.data / "multixscience_test.jsonl"),
        "stats": stats,
        "yield": _json(P.results / "stage1_yield.json"),
    }


def benchmark_examples(limit: int = 20, offset: int = 0, split: str | None = None) -> dict:
    P = paths()
    rows = _jsonl(P.data / "rw_pairs.jsonl")
    if split:
        rows = [r for r in rows if r.get("split") == split]
    total = len(rows)
    page = rows[offset: offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "items": [
        {"arxiv_id": r["arxiv_id"], "title": r.get("title"), "split": r.get("split"),
         "published": r.get("published"), "rw_title": r.get("rw_title"),
         "rw_level": r.get("rw_level"), "words": len((r.get("rw_text") or "").split()),
         "n_cited": len(r.get("cited") or []),
         "coverage": r.get("abstract_coverage")} for r in page]}


def benchmark_example(arxiv_id: str) -> dict | None:
    for r in _jsonl(paths().data / "rw_pairs.jsonl"):
        if r.get("arxiv_id") == arxiv_id:
            return r
    return None


def label_summary() -> dict:
    """Distribution and sample rationales, straight from the label cache."""
    from .relations.taxonomy import DEFINITIONS, LABELS
    rows = _jsonl(paths().data / "llm_labels.jsonl")
    good = [r for r in rows if r.get("llm_label") in LABELS]
    counts = Counter(r["llm_label"] for r in good)
    n = max(1, len(good))
    examples: dict[str, list[dict]] = {}
    for label in LABELS:
        picks = [r for r in good if r["llm_label"] == label][:3]
        examples[label] = [{"context": r.get("marked_context", "")[:420],
                            "cited_title": r.get("cited_title"),
                            "rationale": r.get("llm_rationale"),
                            "confidence": r.get("llm_confidence")} for r in picks]
    conf = [r.get("llm_confidence") or 0 for r in good]
    return {
        "n_labelled": len(good), "n_rows": len(rows),
        "n_citing_papers": len({r.get("citing_id") for r in good}),
        "mean_confidence": round(sum(conf) / n, 3),
        "labels": [{"label": l, "definition": DEFINITIONS[l], "count": counts[l],
                    "share": round(counts[l] / n, 4), "examples": examples[l]}
                   for l in LABELS],
    }


# ------------------------------------------------------------------ the review

def _flow_runs() -> list[Path]:
    root = paths().results / "flow_runs"
    return sorted([p for p in root.glob("*") if p.is_dir()]) if root.exists() else []


def runs() -> list[dict]:
    out = []
    for d in _flow_runs():
        summary = _json(d / "summary.json") or {}
        out.append({"id": d.name, "topic": summary.get("topic") or d.name,
                    "n_records": summary.get("n_records"),
                    "n_claims": summary.get("n_claims"),
                    "has_review": (d / "review.md").exists(),
                    "has_graph": (d / "citation_graph.html").exists()})
    return out


def review(run_id: str | None = None) -> dict | None:
    """A grounded review with, for each sentence, its verdict and its evidence.

    This is the view the whole project exists to produce: every sentence carries
    the claims it rests on, each claim's verbatim passage, and the entailment
    score the Verifier gave it.
    """
    dirs = _flow_runs()
    if not dirs:
        return None
    d = next((x for x in dirs if x.name == run_id), dirs[-1])
    summary = _json(d / "summary.json") or {}
    verdicts = _json(d / "review_verdicts.json") or {}
    records = _json(d / "records.json") or []
    claim_ev: dict[str, dict] = {}
    by_paper: dict[str, dict] = {}
    for rec in records:
        pid = rec.get("paper_id")
        by_paper[pid] = {"title": rec.get("title"), "year": rec.get("year")}
        for c in rec.get("claims") or []:
            claim_ev[f"{pid}:{c['claim_id']}"] = {
                "paper_id": pid, "paper_title": rec.get("title"),
                "text": c.get("text"), "type": c.get("type"),
                "passage": (c.get("evidence") or {}).get("passage"),
                "section": (c.get("evidence") or {}).get("section")}

    sentences = []
    for v in (verdicts.get("sentence_verdicts") or []):
        sentences.append({
            "text": v.get("text"),
            "verdict": v.get("verdict"),
            "entail_prob": v.get("entail_prob"),
            "best_evidence": v.get("best_evidence"),
            "cited_paper_ids": v.get("cited_paper_ids") or [],
        })
    return {
        "run": d.name,
        "topic": summary.get("topic"),
        "markdown": (d / "review.md").read_text() if (d / "review.md").exists() else "",
        "sentences": sentences,
        "faithfulness": verdicts.get("after_revision") or verdicts.get("before_revision") or {},
        "revised": verdicts.get("revised"),
        "papers": by_paper,
        "claims": claim_ev,
        "contradictions": (_json(d / "contradictions.json") or {}).get("contradictions") or [],
        "gaps": (_json(d / "gaps.json") or {}).get("gaps") or [],
        "summary": summary,
        "reference": _json(d / "reference.json"),
    }


# ----------------------------------------------------------------------- usage

def usage() -> dict:
    from .llm import daily_tokens, usage_summary
    P = paths()
    u = usage_summary(P.results / "llm_calls.jsonl")
    try:
        u["daily"] = daily_tokens(path=P.results / "llm_calls.jsonl")
    except Exception:
        u["daily"] = {}
    rows = _jsonl(P.results / "llm_calls.jsonl")
    recent = rows[-40:]
    return {**u, "recent": [{"tag": r.get("tag"), "model": r.get("model"),
                             "total_tokens": r.get("total_tokens"),
                             "latency_s": r.get("latency_s"), "ok": r.get("ok")}
                            for r in recent]}


def references() -> dict:
    d = _json(paths().results / "references_verified.json") or {}
    return {"n_found": d.get("n_found"), "n_candidates": d.get("n_candidates"),
            "items": [r for r in (d.get("references") or []) if r.get("found")]}


def report_status() -> dict:
    P = paths()
    d = _json(P.results / "report_build.json") or {}
    docx, pdf = P.root / "report" / "LitReview_Report.docx", P.root / "report" / "LitReview_Report.pdf"
    return {**d, "docx": docx.exists(), "pdf": pdf.exists(),
            "docx_path": str(docx), "pdf_path": str(pdf)}
