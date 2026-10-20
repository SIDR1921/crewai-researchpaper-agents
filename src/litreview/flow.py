"""The pipeline as a CrewAI ``Flow`` (Stage 5).

``collect_corpus -> approve -> extract -> verify_claims -> classify_relations ->
build_map -> find_gaps -> write_review -> verify_review (revise once) -> export``

Everything is cached under ``run_dir``, so a Flow that dies mid-way continues
where it stopped on the next ``kickoff()``. The approval checkpoint is a callable
so the same Flow works in a notebook (prints the table, asks), in a script
(auto-approve) and in the Streamlit app (a button).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel, Field

from .config import paths
from .schemas import Contradiction, Gap


class FlowState(BaseModel):
    topic: str = ""
    seeds: list[str] = Field(default_factory=list)
    n_papers: int = 15
    run_dir: str = ""
    approved: bool = False
    corpus: list[dict] = Field(default_factory=list)
    corpus_stats: dict = Field(default_factory=dict)
    records: list[dict] = Field(default_factory=list)
    claim_verdicts: dict = Field(default_factory=dict)
    verify_report: dict = Field(default_factory=dict)
    relations: list[dict] = Field(default_factory=list)
    consensus: dict = Field(default_factory=dict)
    contradictions: list[dict] = Field(default_factory=list)
    matrix: dict = Field(default_factory=dict)
    gaps: list[dict] = Field(default_factory=list)
    review: dict = Field(default_factory=dict)
    faithfulness: dict = Field(default_factory=dict)
    exports: dict = Field(default_factory=dict)
    timings: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def auto_approve(table: list[dict]) -> bool:
    return True


class LitReviewFlow:
    """The orchestrated pipeline.

    Implemented as a plain class with a CrewAI ``Flow`` subclass generated in
    :meth:`as_crewai_flow`, so the steps stay unit-testable without a Flow
    runtime and the Flow decorators stay a thin wiring layer.
    """

    def __init__(self, topic: str, n_papers: int = 15, seeds: Sequence[str] | None = None,
                 run_dir: str | Path | None = None,
                 approve: Callable[[list[dict]], bool] = auto_approve,
                 extractor=None, classifier=None, verifier=None,
                 use_relations: bool = True, use_verifier: bool = True,
                 progress=None) -> None:
        P = paths().mkdirs()
        self.state = FlowState(topic=topic, n_papers=n_papers, seeds=list(seeds or []),
                               run_dir=str(run_dir or (P.results / "flow_runs" /
                                                       _slug(topic))))
        Path(self.state.run_dir).mkdir(parents=True, exist_ok=True)
        self.approve = approve
        self._extractor = extractor
        self._classifier = classifier
        self._verifier = verifier
        self.use_relations = use_relations
        self.use_verifier = use_verifier
        self.progress = progress

    # ------------------------------------------------------------------ helpers

    def _p(self, name: str) -> Path:
        return Path(self.state.run_dir) / name

    def _timed(self, name: str, fn, *a, **kw):
        t0 = time.time()
        out = fn(*a, **kw)
        self.state.timings[name] = round(time.time() - t0, 2)
        return out

    @property
    def extractor(self):
        if self._extractor is None:
            from .extract import ApiExtractor
            self._extractor = ApiExtractor()
        return self._extractor

    @property
    def verifier(self):
        if self._verifier is None:
            from .verify import Verifier
            self._verifier = Verifier()
        return self._verifier

    # -------------------------------------------------------------- 1. corpus

    def collect_corpus(self) -> list[dict]:
        cache = self._p("corpus.json")
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            from .corpus import collect_corpus, save_corpus
            data = self._timed("collect_corpus", collect_corpus, self.state.topic,
                               n=self.state.n_papers, seeds=self.state.seeds)
            save_corpus(data, cache)
        self.state.corpus = data["papers"]
        self.state.corpus_stats = data["stats"]
        return self.state.corpus

    # ------------------------------------------------------------ 2. approval

    def approve_corpus(self) -> bool:
        from .corpus import corpus_table
        self.state.approved = bool(self.approve(corpus_table(self.state.corpus)))
        if not self.state.approved:
            self.state.notes.append("corpus rejected at the approval checkpoint")
        return self.state.approved

    # ------------------------------------------------------------- 3. extract

    def extract(self) -> list[dict]:
        from .extract import extract_corpus, load_records
        from .normalise import Normaliser
        cache = self._p("extractions.jsonl")
        self._timed("extract", extract_corpus, self.state.corpus, cache,
                    extractor=self.extractor, progress=self.progress)
        rows = load_records(cache)
        norm = Normaliser()
        self.state.records = [norm.normalise_record(r["record"]) |
                              {"_input_text": (r.get("meta") or {}).get("input_text", "")}
                              for r in rows.values()]
        self._p("normalisation_report.json").write_text(json.dumps(norm.report(), indent=2))
        return self.state.records

    # -------------------------------------------------------- 4. verify claims

    def verify_claims(self) -> dict:
        if not self.use_verifier:
            self.state.claim_verdicts = {}
            self.state.verify_report = {"skipped": True,
                                        "reason": "system configured without the Verifier"}
            return self.state.verify_report
        from .extract import load_records
        rows = load_records(self._p("extractions.jsonl"))
        rep = self._timed("verify_claims", self.verifier.verify_corpus, rows,
                          self._p("claim_verdicts.json"))
        saved = json.loads(self._p("claim_verdicts.json").read_text())
        self.state.claim_verdicts = saved["per_paper"]
        self.state.verify_report = rep
        return rep

    # ---------------------------------------------------- 5. classify relations

    def classify_relations(self) -> list[dict]:
        if not self.use_relations or self._classifier is None:
            self.state.relations = []
            self.state.notes.append("relation classifier not used in this configuration")
            return []
        # Edges come from citation sentences between corpus papers. Where the Flow
        # is run on an OpenAlex corpus we have no LaTeX, so relations are derived
        # from each paper's own claims mentioning another corpus paper's method.
        pairs = []
        for r in self.state.records:
            src = r.get("_input_text") or ""
            for other in self.state.records:
                if other is r:
                    continue
                title = (other.get("title") or "")[:60]
                if len(title) > 20 and title.lower() in src.lower():
                    sent = next((s for s in src.split(". ") if title.lower() in s.lower()), title)
                    pairs.append({"citing_id": r["paper_id"], "cited_id": other["paper_id"],
                                  "sentence": sent[:400],
                                  "marked_context": sent[:400].replace(title, "[[CITED]]"),
                                  "cited_title": other.get("title") or ""})
        if pairs:
            preds = self._timed("classify_relations",
                                self._classifier.classify_batch, pairs)
            for p, (lab, prob) in zip(pairs, preds):
                p["relation"], p["prob"] = lab, prob
        self.state.relations = pairs
        self._p("relations.json").write_text(json.dumps(pairs, indent=2))
        return pairs

    # ------------------------------------------------------------ 6. build map

    def build_map(self) -> dict:
        from .synthesis import (adjudicate, build_graph, consensus_map, contradictions_table,
                                export_graph_html, find_contradictions)
        self.state.consensus = self._timed("consensus_map", consensus_map, self.state.records)
        cons = find_contradictions(self.state.records, self.state.relations)
        from .extract import load_records
        rows = load_records(self._p("extractions.jsonl"))
        cons = self._timed("adjudicate", adjudicate, cons, rows, limit=10,
                           progress=self.progress)
        self.state.contradictions = [c.model_dump() for c in cons]
        g = build_graph(self.state.records, self.state.relations, cons)
        try:
            export_graph_html(g, self._p("citation_graph.html"))
        except Exception as e:  # noqa: BLE001 - a missing renderer must not kill the run
            self.state.notes.append(f"graph HTML not written: {str(e)[:120]}")
        self._p("contradictions.json").write_text(json.dumps(
            {"consensus": self.state.consensus,
             "contradictions": self.state.contradictions,
             "table": contradictions_table(cons)}, indent=2))
        return {"n_groups": self.state.consensus["n_groups"],
                "n_contradictions": len(self.state.contradictions)}

    # ------------------------------------------------------------- 7. find gaps

    def find_gaps(self) -> list[dict]:
        from .gaps import build_matrix, candidate_gaps, rank_gaps, save
        self.state.matrix = build_matrix(self.state.records)
        cands = candidate_gaps(self.state.matrix, self.state.relations, self.state.records)
        ranked = self._timed("rank_gaps", rank_gaps, cands, self.state.matrix,
                             self.state.records, limit=10, progress=self.progress)
        self.state.gaps = [g.model_dump() for g in ranked]
        save(self.state.matrix, ranked, self._p("gaps.json"))
        return self.state.gaps

    # --------------------------------------------------------- 8/9. write+verify

    def write_review(self) -> dict:
        from .review import to_markdown, write_review
        rev, evidence = self._timed("write_review", write_review,
                                    self.state.topic, self.state.records,
                                    self.state.claim_verdicts or None,
                                    self.state.contradictions, self.state.gaps,
                                    self.state.relations,
                                    only_supported=self.use_verifier)
        self._evidence = evidence
        self.state.review = rev.model_dump()
        return self.state.review

    def verify_review(self) -> dict:
        from .review import Review, to_markdown
        rev = Review.model_validate(self.state.review)
        if not self.use_verifier:
            self.state.faithfulness = {"skipped": True,
                                       "reason": "system configured without the Verifier"}
            return self.state.faithfulness
        ev_by_paper: dict[str, list[str]] = {}
        for r in self.state.records:
            ev_by_paper[r["paper_id"]] = [c["evidence"]["passage"] for c in (r.get("claims") or [])]
        sents = [s.model_dump() for s in rev.sentences]
        verdicts = self._timed("verify_review", self.verifier.verify_sentences,
                               sents, getattr(self, "_evidence", {}), ev_by_paper)
        bad = [v.model_dump() for v in verdicts if v.verdict == "unsupported"]
        before = self.verifier.faithfulness(verdicts)
        after, flags = before, []
        if bad:
            from .review import revise_review
            rev2 = self._timed("revise_review", revise_review, rev, bad, self.state.records,
                               self.state.claim_verdicts or None)
            v2 = self.verifier.verify_sentences([s.model_dump() for s in rev2.sentences],
                                                getattr(self, "_evidence", {}), ev_by_paper)
            after = self.verifier.faithfulness(v2)
            if after["faithfulness"] >= before["faithfulness"]:
                rev, verdicts = rev2, v2
            else:
                after = before
                self.state.notes.append("revision did not improve faithfulness; kept draft")
            flags = [v.text for v in verdicts if v.verdict == "unsupported"]
        rev.markdown = to_markdown(rev.topic, rev.sentences, flags)
        self.state.review = rev.model_dump()
        self.state.faithfulness = {
            "before_revision": before, "after_revision": after,
            "revised": bool(bad), "n_flagged_after": len(flags),
            "sentence_verdicts": [v.model_dump() for v in verdicts]}
        self._p("review_verdicts.json").write_text(json.dumps(self.state.faithfulness, indent=2))
        return self.state.faithfulness

    # ---------------------------------------------------------------- 10. export

    def export(self) -> dict:
        from .synthesis import contradictions_table
        d = Path(self.state.run_dir)
        (d / "review.md").write_text(self.state.review.get("markdown", ""))
        (d / "records.json").write_text(json.dumps(self.state.records, indent=2))
        _csv(d / "contradictions.csv", contradictions_table(
            [Contradiction.model_validate(c) for c in self.state.contradictions]))
        _csv(d / "gaps.csv", [{k: v for k, v in g.items() if k != "values"}
                              for g in self.state.gaps])
        summary = {
            "topic": self.state.topic, "run_dir": str(d),
            "corpus_stats": self.state.corpus_stats,
            "n_records": len(self.state.records),
            "n_claims": sum(len(r.get("claims") or []) for r in self.state.records),
            "verify_report": self.state.verify_report,
            "n_relations": len(self.state.relations),
            "relation_counts": {k: sum(1 for e in self.state.relations
                                       if e.get("relation") == k)
                                for k in {e.get("relation") for e in self.state.relations}},
            "consensus": {k: v for k, v in (self.state.consensus or {}).items()
                          if k != "groups"},
            "n_contradictions": len(self.state.contradictions),
            "gap_verdicts": {k: sum(1 for g in self.state.gaps if g.get("verdict") == k)
                             for k in {g.get("verdict") for g in self.state.gaps}},
            "faithfulness": self.state.faithfulness,
            "timings_s": self.state.timings, "notes": self.state.notes,
            "config": {"use_verifier": self.use_verifier, "use_relations": self.use_relations,
                       "extractor": getattr(self.extractor, "name", "?")},
        }
        (d / "summary.json").write_text(json.dumps(summary, indent=2))
        self.state.exports = {
            "review_md": str(d / "review.md"), "records": str(d / "records.json"),
            "graph_html": str(d / "citation_graph.html"),
            "contradictions_csv": str(d / "contradictions.csv"),
            "gaps_csv": str(d / "gaps.csv"), "summary": str(d / "summary.json")}
        return summary

    # ------------------------------------------------------------------- runner

    def run(self) -> dict:
        self.collect_corpus()
        if not self.approve_corpus():
            return {"stopped": "corpus not approved", "corpus_stats": self.state.corpus_stats}
        self.extract()
        self.verify_claims()
        self.classify_relations()
        self.build_map()
        self.find_gaps()
        self.write_review()
        self.verify_review()
        return self.export()

    # ---------------------------------------------- CrewAI Flow wiring (Stage 5)

    def as_crewai_flow(self):
        """Wrap the steps in a real CrewAI ``Flow`` with ``@start`` / ``@listen``."""
        from crewai.flow.flow import Flow, listen, start
        outer = self

        class _LitFlow(Flow[FlowState]):
            @start()
            def collect_corpus(self):
                outer.collect_corpus()
                return "corpus"

            @listen(collect_corpus)
            def approve(self, _):
                return "approved" if outer.approve_corpus() else "rejected"

            @listen(approve)
            def extract(self, prev):
                if prev == "rejected":
                    return "stopped"
                outer.extract()
                return "extracted"

            @listen(extract)
            def verify_claims(self, prev):
                if prev == "stopped":
                    return prev
                outer.verify_claims()
                return "verified"

            @listen(verify_claims)
            def classify_relations(self, prev):
                if prev == "stopped":
                    return prev
                outer.classify_relations()
                return "classified"

            @listen(classify_relations)
            def build_map(self, prev):
                if prev == "stopped":
                    return prev
                outer.build_map()
                return "mapped"

            @listen(build_map)
            def find_gaps(self, prev):
                if prev == "stopped":
                    return prev
                outer.find_gaps()
                return "gaps"

            @listen(find_gaps)
            def write_review(self, prev):
                if prev == "stopped":
                    return prev
                outer.write_review()
                return "written"

            @listen(write_review)
            def verify_review(self, prev):
                if prev == "stopped":
                    return prev
                outer.verify_review()
                return "review-verified"

            @listen(verify_review)
            def export(self, prev):
                if prev == "stopped":
                    return {"stopped": True}
                return outer.export()

        return _LitFlow()


def _slug(s: str) -> str:
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "-", (s or "run").lower()).strip("-")[:60] or "run"


def _csv(path: Path, rows: Sequence[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
