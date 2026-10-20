"""The systems compared in Stage 7.

Each takes one benchmark example -- a topic plus the abstracts of the papers the
real related-work section cites -- and returns a review plus the artefacts the
metrics need. The cited papers *are* the corpus, so corpus collection is out of
scope here; that keeps the comparison about synthesis quality rather than
retrieval luck.

* ``S1`` single well-prompted LLM call
* ``S2`` crew without the Verifier (extract -> write)
* ``S3`` full crew (extract -> verify -> relations -> map -> gaps -> write -> verify/revise)
* ``S4`` full crew with the distilled extractor
* ``S3_no_relations`` ablation: S3 with the relation classifier switched off
"""
from __future__ import annotations

import time
from typing import Sequence

from ..extract import ApiExtractor, coerce_record
from ..normalise import Normaliser
from ..review import Review, single_call_review, split_sentences, to_markdown, write_review
from ..schemas import ReviewSentence

SYSTEMS = ["S1", "S2", "S3", "S4", "S3_no_relations"]
SYSTEM_LABELS = {
    "S1": "S1 single LLM call",
    "S2": "S2 crew, no Verifier",
    "S3": "S3 full crew",
    "S4": "S4 full crew + distilled extractor",
    "S3_no_relations": "S3 ablation: no relation classifier",
}


def _papers_from_pair(pair: dict) -> list[dict]:
    """The cited papers of one benchmark example, as Extractor inputs."""
    out = []
    for i, c in enumerate(pair.get("cited") or [], 1):
        if not (c.get("abstract") or "").strip():
            continue
        pid = c.get("cite_key") or c.get("openalex_id") or f"p{i}"
        out.append({"paper_id": pid, "title": c.get("title") or pid,
                    "year": c.get("year"), "abstract": c["abstract"],
                    "sections": {"abstract": c["abstract"]}})
    return out


def topic_of(pair: dict) -> str:
    """The topic string given to every system: the citing paper's own title/abstract.

    Deliberately *not* the gold related-work text -- that would leak the target.
    """
    t = (pair.get("title") or "").strip()
    a = (pair.get("abstract") or "").strip()
    if t and a:
        return f"{t}. {a[:600]}"
    return t or a[:600] or "this research area"


def run_s1(pair: dict, **kw) -> dict:
    t0 = time.time()
    rev = single_call_review(topic_of(pair), _papers_from_pair(pair))
    return {"system": "S1", "review": rev.model_dump(),
            "records": [], "evidence_by_paper": {},
            "wall_s": round(time.time() - t0, 2), "notes": []}


def _extract_all(papers: Sequence[dict], extractor) -> tuple[list[dict], dict, list[str]]:
    records, notes = [], []
    norm = Normaliser()
    for p in papers:
        rec, meta = extractor.extract(p)
        if rec is None:
            notes.append(f"extraction failed for {p['paper_id']}: {str(meta.get('error'))[:80]}")
            continue
        d = rec.model_dump(mode="json")
        records.append(norm.normalise_record(d) | {"_input_text": meta.get("input_text", "")})
    ev = {r["paper_id"]: [c["evidence"]["passage"] for c in (r.get("claims") or [])]
          for r in records}
    return records, ev, notes


def run_crew(pair: dict, verifier=None, classifier=None, extractor=None,
             use_verifier: bool = True, use_relations: bool = True,
             system: str = "S3", adjudicate_limit: int = 3,
             gap_limit: int = 3, **kw) -> dict:
    """S2, S3, S4 and the ablation, which differ only in these switches."""
    t0 = time.time()
    extractor = extractor or ApiExtractor()
    papers = _papers_from_pair(pair)
    records, ev_by_paper, notes = _extract_all(papers, extractor)
    if not records:
        return {"system": system, "review": Review(topic=topic_of(pair)).model_dump(),
                "records": [], "evidence_by_paper": {}, "wall_s": round(time.time() - t0, 2),
                "notes": notes + ["no records extracted"]}

    verdicts = {}
    verify_report = {}
    if use_verifier and verifier is not None:
        rows = {r["paper_id"]: {"record": r, "meta": {"input_text": r.get("_input_text", "")}}
                for r in records}
        verify_report = verifier.verify_corpus(rows)
        verdicts = {pid: [v.model_dump() for v in
                          verifier.verify_claims(row["record"], row["meta"]["input_text"])]
                    for pid, row in rows.items()}

    relations: list[dict] = []
    if use_relations and classifier is not None:
        pairs = []
        for r in records:
            src = (r.get("_input_text") or "").lower()
            for other in records:
                if other is r:
                    continue
                title = (other.get("title") or "")[:60]
                if len(title) > 20 and title.lower() in src:
                    sent = next((s for s in (r.get("_input_text") or "").split(". ")
                                 if title.lower() in s.lower()), title)
                    pairs.append({"citing_id": r["paper_id"], "cited_id": other["paper_id"],
                                  "sentence": sent[:400],
                                  "marked_context": sent[:400].replace(title, "[[CITED]]"),
                                  "cited_title": other.get("title") or ""})
        if pairs:
            for p, (lab, prob) in zip(pairs, classifier.classify_batch(pairs)):
                p["relation"], p["prob"] = lab, prob
            relations = pairs

    contradictions, gaps = [], []
    if system in ("S3", "S4", "S3_no_relations"):
        from ..gaps import build_matrix, candidate_gaps, rank_gaps
        from ..synthesis import adjudicate, find_contradictions
        cons = find_contradictions(records, relations)
        rows = {r["paper_id"]: {"record": r} for r in records}
        contradictions = [c.model_dump() for c in adjudicate(cons, rows, limit=adjudicate_limit)]
        matrix = build_matrix(records)
        cands = candidate_gaps(matrix, relations, records, max_candidates=10)
        gaps = [g.model_dump() for g in rank_gaps(cands, matrix, records, limit=gap_limit)]

    rev, evidence = write_review(topic_of(pair), records, verdicts or None,
                                 contradictions, gaps, relations,
                                 only_supported=use_verifier)

    faith = {}
    if use_verifier and verifier is not None:
        sents = [s.model_dump() for s in rev.sentences]
        vs = verifier.verify_sentences(sents, evidence, ev_by_paper)
        before = verifier.faithfulness(vs)
        bad = [v.model_dump() for v in vs if v.verdict == "unsupported"]
        after, flags = before, []
        if bad:
            from ..review import revise_review
            rev2 = revise_review(rev, bad, records, verdicts or None)
            v2 = verifier.verify_sentences([s.model_dump() for s in rev2.sentences],
                                           evidence, ev_by_paper)
            after = verifier.faithfulness(v2)
            if after["faithfulness"] >= before["faithfulness"]:
                rev, vs = rev2, v2
            else:
                after = before
                notes.append("revision did not improve faithfulness; kept draft")
            flags = [v.text for v in vs if v.verdict == "unsupported"]
        rev.markdown = to_markdown(rev.topic, rev.sentences, flags)
        faith = {"before_revision": before, "after_revision": after,
                 "n_flagged_after": len(flags)}

    return {"system": system, "review": rev.model_dump(), "records": records,
            "evidence_by_paper": ev_by_paper, "verify_report": verify_report,
            "faithfulness_internal": faith, "n_relations": len(relations),
            "relation_counts": {k: sum(1 for e in relations if e.get("relation") == k)
                                for k in {e.get("relation") for e in relations}},
            "n_contradictions": len(contradictions),
            "n_gaps_real": sum(1 for g in gaps if g.get("verdict") == "real-gap"),
            "wall_s": round(time.time() - t0, 2), "notes": notes}


def run_system(system: str, pair: dict, verifier=None, classifier=None,
               distilled=None, **kw) -> dict:
    if system == "S1":
        return run_s1(pair)
    if system == "S2":
        return run_crew(pair, verifier=None, classifier=None, use_verifier=False,
                        use_relations=False, system="S2", **kw)
    if system == "S3":
        return run_crew(pair, verifier=verifier, classifier=classifier,
                        use_verifier=True, use_relations=True, system="S3", **kw)
    if system == "S4":
        if distilled is None:
            return {"system": "S4", "skipped": True,
                    "reason": "distilled extractor not available", "review": {},
                    "records": [], "evidence_by_paper": {}, "wall_s": 0.0, "notes": []}
        return run_crew(pair, verifier=verifier, classifier=classifier, extractor=distilled,
                        use_verifier=True, use_relations=True, system="S4", **kw)
    if system == "S3_no_relations":
        return run_crew(pair, verifier=verifier, classifier=None, use_verifier=True,
                        use_relations=False, system="S3_no_relations", **kw)
    raise ValueError(f"unknown system {system!r}")
