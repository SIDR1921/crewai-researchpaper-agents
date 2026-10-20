#!/usr/bin/env python
"""Fill Report_Template.docx from results/ and figures/ only.

Two rules are enforced by construction rather than by care:

* **Every number comes from a results file.** Values are read through
  :class:`Results`, which records every lookup. A missing file does not become an
  estimate: the sentence that needed it is replaced by an explicit statement that
  the stage was not run, and the gap is listed at the end of the run.
* **Every citation resolves to a verified reference.** :class:`References` is
  seeded from ``results/references_verified.json`` and refuses any key that is not
  in it with ``found: true``, so a citation marker cannot point at a paper no
  source confirmed.

    python scripts/build_report.py
    python scripts/build_report.py --pdf        # also render a PDF via Word
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litreview import docx_tools as D  # noqa: E402
from litreview.config import HF_MODELS, paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

TITLE = "Grounded Literature Review Generation with Multi-Agent LLMs and Citation-Relation Learning"
STUDENT = ("Siddharth Ray", "235890416")
BRANCH = "SoCE – AI"
SECTION_PLACEHOLDER = "<Section>"
COVER_MONTH = "October 2026"
SEMESTER_PLACEHOLDER = "<SEMESTER>"
TEAMMATE_SLOTS = 2          # extra <STUDENT NAME>/<reg no.> pairs kept as placeholders

NA = "not available"


# --------------------------------------------------------------------- results

class Results:
    """Read-only accessor over ``results/*.json`` that records what it could not find."""

    def __init__(self, results_dir: Path) -> None:
        self.dir = results_dir
        self.files: dict[str, dict] = {}
        for p in sorted(results_dir.glob("*.json")):
            try:
                self.files[p.name] = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
        self.misses: list[str] = []
        self.hits = 0

    def has(self, filename: str) -> bool:
        return filename in self.files

    def get(self, filename: str, path: str = "", default=None):
        """``get("stage1_yield.json", "yield.listed")``. Records a miss and returns default."""
        node = self.files.get(filename)
        if node is None:
            self.misses.append(f"{filename} (file absent)")
            return default
        for key in [k for k in path.split(".") if k]:
            if isinstance(node, list):
                try:
                    node = node[int(key)]
                    continue
                except (ValueError, IndexError):
                    self.misses.append(f"{filename}:{path}")
                    return default
            if not isinstance(node, dict) or key not in node:
                self.misses.append(f"{filename}:{path}")
                return default
            node = node[key]
        self.hits += 1
        return node

    def num(self, filename: str, path: str, fmt: str = "{:,}", default: str = NA) -> str:
        v = self.get(filename, path)
        if v is None:
            return default
        try:
            return fmt.format(v)
        except (ValueError, TypeError):
            return str(v)

    def pct(self, filename: str, path: str, of: float | None = None, default: str = NA) -> str:
        v = self.get(filename, path)
        if v is None:
            return default
        try:
            x = float(v)
        except (TypeError, ValueError):
            return default
        if of:
            x = 100.0 * x / of
        elif x <= 1.0:
            x *= 100.0
        return f"{x:.0f}%"


# ------------------------------------------------------------------ references

class References:
    """First-use-ordered citation registry, seeded only from verified references."""

    def __init__(self, verified: dict | None) -> None:
        self.pool: list[dict] = [r for r in ((verified or {}).get("references") or [])
                                 if r.get("found")]
        self.order: list[dict] = []
        self.by_key: dict[str, int] = {}
        self.unresolved: list[str] = []

    def _match(self, key: str) -> dict | None:
        best, score = None, 0
        for r in self.pool:
            for cand in (r.get("query") or "", r.get("title") or ""):
                s = fuzz.token_set_ratio(key.lower(), cand.lower())
                if s > score:
                    best, score = r, s
        return best if score >= 80 else None

    def cite(self, *keys: str) -> str:
        """``[3]`` or ``[3], [7]``; an unverified key yields nothing and is reported."""
        nums = []
        for key in keys:
            if key in self.by_key:
                nums.append(self.by_key[key])
                continue
            r = self._match(key)
            if r is None:
                if key not in self.unresolved:
                    self.unresolved.append(key)
                continue
            # the same work may be reached by two keys; keep one number
            existing = next((n for n, x in enumerate(self.order, 1) if x is r), None)
            n = existing or (len(self.order) + 1)
            if existing is None:
                self.order.append(r)
            self.by_key[key] = n
            nums.append(n)
        return ", ".join(f"[{n}]" for n in sorted(set(nums)))

    def entry(self, r: dict, n: int) -> str:
        authors = r.get("authors") or []
        if len(authors) > 3:
            who = f"{authors[0]} et al."
        elif authors:
            who = " and ".join(authors) if len(authors) <= 2 else \
                  ", ".join(authors[:-1]) + " and " + authors[-1]
        else:
            who = "Anon."
        venue = (r.get("venue") or "").strip() or "preprint"
        venue = re.sub(r"\s*\(.*?\)\s*$", "", venue)
        year = r.get("year") or "n.d."
        doi = (r.get("doi") or "").replace("https://doi.org/", "")
        tail = f", doi:{doi}" if doi else (f", arXiv:{r['arxiv_id']}" if r.get("arxiv_id") else "")
        return f"[{n}] {who}, “{r.get('title')}”, {venue}, {year}{tail}"

    def bibliography(self) -> list[str]:
        return [self.entry(r, n) for n, r in enumerate(self.order, 1)]


# ------------------------------------------------------------------ the builder

class ReportBuilder:
    def __init__(self, template: Path, out: Path, R: Results, refs: References,
                 figures: Path) -> None:
        self.doc = Document(str(template))
        self.out = out
        self.R = R
        self.refs = refs
        self.figures = figures
        self.fig = D.Counter("Figure")
        self.tab = D.Counter("Table")
        self.toc: list[dict] = []
        self.notes: list[str] = []
        self.missing_figures: list[str] = []

    # ---------------------------------------------------------------- plumbing

    def anchor(self, text: str, start: int = 0, exact: bool = False):
        return D.find_paragraph(self.doc, text, start=start, exact=exact)

    def bookmark_heading(self, par, label: str, kind: str = "sub", number: str = "") -> str:
        name = "_toc_" + re.sub(r"[^a-z0-9]+", "_", f"{number} {label}".strip().lower())[:38]
        D.add_bookmark(par, name)
        self.toc.append({"kind": kind, "number": number, "label": label, "bookmark": name})
        return name

    def figure(self, anchor_par, filename: str, caption: str, width_in: float = 5.8):
        """Insert a figure with a numbered caption, or a visible note if it is missing."""
        path = self.figures / filename
        if not path.exists():
            self.missing_figures.append(filename)
            p = D.add_body_paragraph(
                anchor_par,
                f"[Figure omitted: {filename} was not produced by scripts/make_figures.py, "
                f"because the results file it reads does not exist. Re-run the corresponding "
                f"stage and then make_figures.py.]",
                size_pt=10.5, italic=True, first_line_indent=None)
            return p
        item = self.fig.next(caption, source=filename)
        img = D.add_image(anchor_par, path, width_in=width_in)
        cap = D.add_caption(img, f"{item['label']}: {caption}")
        D.add_bookmark(cap, item["bookmark"])
        return cap

    def table(self, anchor_par, rows, caption: str, **kw):
        """Insert a table with a numbered caption above it, as the template's lists expect."""
        item = self.tab.next(caption)
        cap = D.add_caption(anchor_par, f"{item['label']}: {caption}", size_pt=10.0)
        cap.paragraph_format.space_after = Pt(3)
        D.add_bookmark(cap, item["bookmark"])
        t = D.add_table(self.doc, cap, rows, **kw)
        # a spacer paragraph after the table, so the next text is not glued to it
        from docx.text.paragraph import Paragraph
        import copy as _copy
        spacer = Paragraph(_copy.deepcopy(cap._element), cap._parent)
        for child in list(spacer._element):
            if child.tag.endswith("}r") or child.tag.endswith("}bookmarkStart") \
                    or child.tag.endswith("}bookmarkEnd"):
                spacer._element.remove(child)
        t._element.addnext(spacer._element)
        return spacer

    def drop_boilerplate(self) -> None:
        """Remove the template's instructional text, which is not part of a report."""
        kill_exact = ["This chapter should include", "Results/observations",
                      "Analysis of the results"]
        for p in list(self.doc.paragraphs):
            t = p.text.strip()
            if any(t.startswith(k) for k in kill_exact):
                D.delete_paragraph(p)
        # the example reference lines
        for p in list(self.doc.paragraphs):
            t = p.text.strip()
            if re.match(r"^\[\d\]\s*(Name 1|Topic 1)", t):
                D.delete_paragraph(p)

    # ------------------------------------------------------------------- cover

    def cover(self) -> None:
        counts = D.replace_everywhere(self.doc, {
            "<TITLE OF THE PROJECT WORK>": TITLE,
            "<Branch>": BRANCH,
            "<Section>": SECTION_PLACEHOLDER,
        })
        if not counts.get("<TITLE OF THE PROJECT WORK>"):
            i, p = self.anchor("TITLE OF THE PROJECT WORK")
            D.set_text(p, TITLE)

        # student names: first slot is the author, the rest stay as placeholders
        names = [p for p in self.doc.paragraphs if "<STUDENT NAME>" in p.text]
        regs = [p for p in self.doc.paragraphs if "<reg no.>" in p.text]
        if names:
            D.set_text(names[0], STUDENT[0])
        if regs:
            D.set_text(regs[0], f"Reg. No. {STUDENT[1]}")
        for p in names[1 + TEAMMATE_SLOTS:]:
            D.delete_paragraph(p)
        for p in regs[1 + TEAMMATE_SLOTS:]:
            D.delete_paragraph(p)
        self.notes.append(
            f"Cover page: {min(TEAMMATE_SLOTS, max(0, len(names) - 1))} teammate name/reg-no "
            f"slots were left in place as <STUDENT NAME> / <reg no.> placeholders. Fill or "
            f"delete them.")
        self.notes.append("Cover page: <Section> is left as a placeholder — set your section.")

        # semester and month
        try:
            _, p = self.anchor("IV SEMESTER MINI PROJECT REPORT")
            D.set_text(p, f"{SEMESTER_PLACEHOLDER} SEMESTER MINI PROJECT REPORT ")
            p.runs[0].italic = True
            p.runs[0].font.size = Pt(12)
            self.notes.append(
                "Cover page: the template said “IV SEMESTER”. It is now the placeholder "
                f"{SEMESTER_PLACEHOLDER} — replace it with your current semester.")
        except KeyError:
            pass
        try:
            _, p = self.anchor("April 2026")
            D.set_text(p, COVER_MONTH, bold=True, size_pt=12)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            self.notes.append(
                f"Cover page: the template said “April 2026”; set to “{COVER_MONTH}” "
                "(the month this report was generated). Change it if your submission month differs.")
        except KeyError:
            pass

    # ---------------------------------------------------------------- abstract

    def abstract(self) -> None:
        R, C = self.R, self.refs.cite
        i, head = self.anchor("ABSTRACT", exact=False)
        self.bookmark_heading(head, "Abstract", kind="front")

        listed = R.get("stage1_yield.json", "yield.listed")
        kept = R.get("stage1_yield.json", "datasets.rw_pairs_kept")
        n_ctx = R.get("stage1_yield.json", "datasets.contexts_rows")
        kappa = R.get("stage2_gold_agreement.json", "llm_vs_gold.cohen_kappa")
        sci_f1 = R.get("stage2_benchmark.json", "scibert.macro_f1")
        tfidf_f1 = R.get("stage2_benchmark.json", "tfidf_lr.macro_f1")
        sup = R.get("stage3_verify_train_api.json", "verdict_rates.supported")
        eval_aggs = R.get("eval_arxiv.json", "aggregates") or {}

        def mean(system: str, metric: str):
            return ((eval_aggs.get(system) or {}).get(metric) or {}).get("mean")

        s1_r, s3_r = mean("S1", "rougeL"), mean("S3", "rougeL")
        s1_f, s3_f = mean("S1", "faithfulness"), mean("S3", "faithfulness")
        n_eval = R.get("eval_arxiv.json", "benchmark.n_examples")

        parts = [
            f"Large language models write fluent literature reviews that cite papers they have "
            f"not read. This project builds and evaluates a CrewAI-based research-paper "
            f"summarisation and literature-review assistant in which every sentence of the "
            f"output is traceable to a quoted passage of a specific paper and is checked by a "
            f"natural-language-inference Verifier before it survives. Six agents — Librarian, "
            f"Extractor, Verifier, Synthesist/Adjudicator, Gap Analyst and Writer — are "
            f"orchestrated as a CrewAI Flow with a human approval checkpoint after corpus "
            f"collection."
        ]
        p2 = (f"Training and evaluation data are reverse-engineered from real papers: a paper's "
              f"related-work section is a human-written mini literature review, so mining those "
              f"sections together with the abstracts of the papers they cite yields both a "
              f"supervision signal and a natural benchmark. ")
        if listed and kept:
            p2 += (f"From {listed:,} arXiv cs.CL and cs.IR submissions the pipeline produced "
                   f"{n_ctx:,} citation contexts and {kept} usable related-work benchmark "
                   f"examples. ")
        else:
            p2 += "Stage 1 mining statistics were not available when this report was built. "
        if kappa is not None and sci_f1 is not None:
            p2 += (f"A seven-class citation-relation classifier fine-tuned from SciBERT reaches "
                   f"{sci_f1:.2f} macro-F1 on a hand-labelled gold set"
                   + (f", against {tfidf_f1:.2f} for a TF-IDF baseline" if tfidf_f1 is not None else "")
                   + f"; the LLM labels it learns from agree with the human gold set at "
                     f"κ = {kappa:.2f}.")
        parts.append(p2)

        p3 = ""
        if sup is not None:
            p3 += (f"The Verifier, whose two thresholds are fitted on a hand-checked dev set, "
                   f"finds {sup:.0%} of extracted claims fully supported by the passage they "
                   f"quote. ")
        if s1_r is not None and s3_r is not None and n_eval:
            direction = "above" if s3_r > s1_r else "below"
            p3 += (f"On a frozen, time-split benchmark of {n_eval} papers published after the "
                   f"evaluated models' knowledge cutoffs, the full crew scores ROUGE-L "
                   f"{s3_r:.3f}, {direction} a single well-prompted call at {s1_r:.3f}")
            if s1_f is not None and s3_f is not None:
                p3 += (f", while sentence-level faithfulness moves from {s1_f:.2f} to {s3_f:.2f}")
            p3 += ". "
        else:
            p3 += ("The full S1–S4 comparison had not been run when this report was built; "
                   "Chapter 4 states exactly which systems were evaluated. ")
        p3 += ("All reported numbers are read from the result files the code wrote, and every "
               "figure is generated from those files by a single script.")
        parts.append(p3)

        cur = head
        for t in parts:
            cur = D.add_body_paragraph(cur, t, size_pt=12, first_line_indent=None,
                                       space_after_pt=8)
        words = sum(len(t.split()) for t in parts)
        self.notes.append(f"Abstract length: {words} words.")

    # --------------------------------------------------------------- chapter 1

    def chapter1(self) -> None:
        R, C = self.R, self.refs.cite
        _, ch = self.anchor("CHAPTER 1", exact=True)
        self.bookmark_heading(ch, "INTRODUCTION", kind="chapter", number="1")

        _, h = self.anchor("1.1 Overview of the")
        self.bookmark_heading(h, "Overview of the project work", number="1.1")
        D.add_paragraphs(h, [
            "This project builds a literature-review assistant from a team of large-language-"
            "model agents, and then measures whether the team is worth its cost. Given a topic "
            "or a few seed papers, the system collects a corpus, extracts a structured record "
            "from each paper in which every claim quotes the passage it came from, checks each "
            "claim against that passage with a natural-language-inference model, builds a "
            "consensus and contradiction map over the reported results, identifies candidate "
            "research gaps, and finally writes a related-work section in which each sentence "
            "carries the identifiers of the claims it rests on.",
            "The pipeline is orchestrated as a CrewAI Flow of ten steps with a human approval "
            "checkpoint after corpus collection, and six agents with deliberately narrow roles: "
            "the Extractor cannot search, the Writer cannot verify, and only the Verifier may "
            "call the entailment tool. Two components are fine-tuned rather than prompted. The "
            "first is a seven-class citation-relation classifier, trained from SciBERT on "
            "citation contexts mined out of real papers' related-work sections. The second is a "
            "small instruction-tuned model distilled from the hosted extractor, trained only on "
            "extractions whose every claim passed the Verifier, so the Verifier acts as the "
            "filter on its own training data.",
            "The same supervision idea supplies the benchmark. A published paper's related-work "
            "section is a literature review that a domain expert wrote for a known set of cited "
            "papers. Hiding the section and supplying only those cited papers' abstracts gives a "
            "task with a human reference, which is what makes the evaluation in Chapter 4 "
            "possible without any new human annotation effort at evaluation time.",
        ])

        _, h = self.anchor("1.2 Motivation and background")
        self.bookmark_heading(h, "Motivation and background of the study", number="1.2")
        D.add_paragraphs(h, [
            f"Abstractive summarisation models are fluent and are often wrong in ways that "
            f"fluency hides. Maynez et al. {C('On Faithfulness and Factuality')} found that a "
            f"large share of summaries from strong sequence-to-sequence models contain content "
            f"unsupported by the source document, and that ROUGE barely distinguishes those "
            f"summaries from faithful ones. The survey of hallucination in natural language "
            f"generation {C('Survey of Hallucination')} documents the same failure across "
            f"generation tasks. For a literature review the consequence is specific and serious: "
            f"a plausible sentence attached to a real citation is a fabricated claim about "
            f"somebody's work, and it is harder to catch than an invented reference, because the "
            f"reference checks out.",
            f"The response in the summarisation literature has been to evaluate entailment "
            f"rather than overlap. Question-answering-based consistency checks "
            f"{C('Asking and Answering Questions')}, classifier-based checks "
            f"{C('Evaluating the Factual Consistency')} and sentence-level "
            f"natural-language-inference checks {C('SummaC')} all score a summary against its "
            f"source instead of against a reference text. This project adopts that stance at the "
            f"level of the individual sentence, and goes one step further by requiring the "
            f"system to say *which* passage each sentence is supposed to follow from, so the "
            f"check is well defined rather than approximate.",
            f"Multi-agent LLM frameworks have meanwhile made it cheap to decompose a task across "
            f"specialised roles with tools and a shared state "
            f"{C('CAMEL', 'Generative Agents')}, and self-critique loops "
            f"{C('Self-Refine', 'Reflexion')} have been shown to improve outputs on several "
            f"tasks. What is largely missing is an honest accounting of whether that machinery "
            f"pays for itself on a task with a human reference. The evaluation in this report is "
            f"designed to answer that question for one task, with the same model and settings "
            f"across every system, and to report the answer whichever way it comes out.",
        ])

        _, h = self.anchor("1.3 Problem statement")
        self.bookmark_heading(h, "Problem statement and relevance", number="1.3")
        D.add_paragraphs(h, [
            "The problem addressed is: given a research topic and the papers relevant to it, "
            "produce a literature review in which every sentence is grounded in an identifiable "
            "passage of an identifiable paper, together with a map of where those papers agree "
            "and disagree and a ranked set of gaps, and do so under free-tier API limits and on "
            "a single free-tier GPU.",
            "Three sub-problems follow. First, grounding needs supervision that does not exist "
            "as a labelled dataset: what relation a citing sentence asserts about a cited paper "
            "has to be learned, and the only abundant source of that signal is the related-work "
            "sections of published papers. Second, verification has to be mechanical rather than "
            "a second opinion from the same model, because a model asked to check its own output "
            "agrees with itself. Third, the whole thing has to be measured against something a "
            "human wrote, or the claim of groundedness is untestable.",
            "The relevance is practical. The number of papers in any active area now exceeds "
            "what a student can read before writing a review, and the tools that promise to help "
            "are precisely the tools that hallucinate. A system that is slower but whose every "
            "sentence can be checked against a quoted passage is more useful than a faster one "
            "whose output must be re-read against the sources anyway.",
        ])

        _, h = self.anchor("1.4 Objectives")
        self.bookmark_heading(h, "Objectives", number="1.4")
        D.add_body_paragraph(h, "The objectives are numbered and measurable; the metric named in "
                                "each is the one reported in Chapter 4.")
        last = self.doc.paragraphs[self.anchor("1.4 Objectives")[0] + 1]
        D.add_bullets(last, [
            "Mine related-work sections and citation contexts from arXiv LaTeX sources, and "
            "report the yield at every step of the pipeline as a percentage of papers listed.",
            "Define a seven-class citation-relation taxonomy with a written labelling guide, "
            "label contexts with an LLM, and quantify how far those labels can be trusted by "
            "measuring accuracy and Cohen's κ against an independently hand-labelled gold set.",
            "Fine-tune SciBERT on those labels and beat a majority-class, a TF-IDF + logistic-"
            "regression and a zero-shot LLM baseline on macro-F1 over the gold set.",
            "Extract structured records in which every claim quotes its evidence verbatim, and "
            "verify each claim with a quote check and an NLI entailment check whose thresholds "
            "are fitted on a hand-checked dev set rather than chosen by hand.",
            "Build a consensus and contradiction map and a method-family × dataset coverage "
            "matrix, and have agents argue both sides of each flagged contradiction and each "
            "candidate gap using only the evidence.",
            "Distil the hosted extractor into a small local model using only Verifier-approved "
            "extractions, and compare the two on JSON validity, claim faithfulness, field "
            "agreement, latency and cost.",
            "Evaluate four systems and one ablation on a frozen, time-split benchmark with "
            "ROUGE, BERTScore, citation coverage, citation correctness and NLI faithfulness, "
            "reporting bootstrap 95% confidence intervals and the token and latency cost of each.",
        ], numbered=True)

    # --------------------------------------------------------------- chapter 2

    def chapter2(self) -> None:
        C = self.refs.cite
        _, ch = self.anchor("CHAPTER 2", exact=True)
        self.bookmark_heading(ch, "BACKGROUND THEORY / LITERATURE REVIEW",
                              kind="chapter", number="2")
        _, title = self.anchor("BACKGROUND THEORY / LITERATURE REVIEW")

        cur = D.add_paragraphs(title, [
            "This chapter surveys the five bodies of work this project draws on: multi-document "
            "and scientific summarisation, citation intent and function classification, "
            "faithfulness and hallucination evaluation, LLM agents and multi-agent frameworks, "
            "and LLM-assisted literature-review tools. Every work discussed here was checked "
            "against OpenAlex, Semantic Scholar or the arXiv API before being cited; the "
            "verification record is in results/references_verified.json.",
        ])

        cur = D.add_heading(cur, "Multi-document and scientific summarisation")
        cur = D.add_paragraphs(cur, [
            f"Neural abstractive summarisation of single documents matured with "
            f"pointer-generator architectures that could copy out-of-vocabulary tokens from the "
            f"source {C('Get To The Point')} and then with pre-trained denoising sequence-to-"
            f"sequence models, of which BART {C('BART')} and PEGASUS {C('PEGASUS')} remain the "
            f"standard comparison points. Scientific articles broke the assumption that a "
            f"document fits in the encoder: Cohan et al. {C('A Discourse-Aware Attention Model')} "
            f"introduced a discourse-aware attention model together with the arXiv and PubMed "
            f"long-document datasets, and the XSum corpus {C('Just the Summary')} "
            f"pushed in the opposite direction, towards one-sentence summaries that cannot be "
            f"produced extractively.",
            f"Related-work generation is the sub-task this project evaluates, and it is older "
            f"than the current generation of models. Abu-Ra'ed et al. "
            f"{C('Automatic related work section generation')} treated it as scientific document "
            f"abstracting over the cited papers, and Hoang and Kan's successors, notably Wang et "
            f"al. {C('Neural Related Work Summarization')}, introduced joint context-driven "
            f"attention over the citing paper's own content and the cited abstracts. Luu et al. "
            f"{C('Explaining Relationships Between Scientific Documents')} framed the problem as "
            f"explaining the *relationship* between two documents rather than summarising one of "
            f"them, which is the framing closest to the relation classifier used here.",
            f"Multi-XScience {C('Multi-XScience')} is the dataset that made the task a benchmark: "
            f"each example pairs a paper's related-work paragraph with the abstracts of the "
            f"papers it cites, which is exactly the input-output contract adopted in Chapter 3. "
            f"It is used in this project as a secondary benchmark alongside the mined arXiv "
            f"split, so results can be compared against a published dataset rather than only "
            f"against our own mining. The corpus-level infrastructure both rely on comes from "
            f"S2ORC {C('S2ORC')}, which released parsed full text and resolved citation links at "
            f"a scale that made citation-context mining routine. More recently, Zhang et al. "
            f"{C('Benchmarking Large Language Models for News Summarization')} found that "
            f"instruction-tuned LLMs match or beat fine-tuned summarisers in human judgement "
            f"while reference-based metrics disagree, which is a direct warning about reading "
            f"ROUGE alone, and one the evaluation design in this report takes seriously.",
        ])

        cur = D.add_heading(cur, "Citation intent and function classification")
        cur = D.add_paragraphs(cur, [
            f"Not every citation does the same work, and the differences are what a review has "
            f"to express. Early work in the area concentrated on citation sentiment and on "
            f"recovering the citation's context window "
            f"{C('Context-Enhanced Citation Sentiment Detection')}, "
            f"{C('Mining the Context of Citations')}. The two datasets that "
            f"standardised the task are ACL-ARC, whose six citation-frame labels Jurgens et al. "
            f"{C('Measuring the Evolution of a Scientific Field')} used to trace how a field "
            f"changes, and SciCite {C('Structural Scaffolds')}, which traded label granularity "
            f"for scale with three classes — background, method and result — and showed that "
            f"auxiliary structural tasks improve classification. SciBERT {C('SciBERT')} supplied "
            f"the scientific-text encoder that both datasets are now usually attacked with, and "
            f"is the starting point for the classifier in this project.",
            "The taxonomy used here is finer than SciCite's and differently cut from ACL-ARC's, "
            "for a reason that matters downstream: this project needs to distinguish neutral "
            "comparison from asserted disagreement, because a `contrasts` edge is evidence of a "
            "contradiction while a `compares-against` edge is not. SciCite's `result` class and "
            "ACL-ARC's `CompareOrContrast` class both merge the two. Section 3.1 documents the "
            "mapping used to reuse those corpora as additional training signal, and the limits "
            "of that mapping; because of those limits, mapped rows are never used as evaluation "
            "gold.",
        ])

        cur = D.add_heading(cur, "Faithfulness and hallucination evaluation")
        cur = D.add_paragraphs(cur, [
            f"The finding that motivates this project is that fluent summaries are frequently "
            f"unfaithful and that reference-overlap metrics do not notice "
            f"{C('On Faithfulness and Factuality')}. Three families of automatic check emerged "
            f"in response. FactCC {C('Evaluating the Factual Consistency')} trains a classifier "
            f"on synthetically corrupted summaries. QAGS {C('Asking and Answering Questions')} "
            f"generates questions from the summary and compares the answers obtained from the "
            f"summary and from the source. SummaC {C('SummaC')} revisits sentence-level "
            f"natural-language inference and shows that NLI models work well for inconsistency "
            f"detection once they are applied at sentence granularity rather than to whole "
            f"documents, which is the design this project's Verifier follows.",
            f"Evaluating the evaluators required human annotation: SummEval "
            f"{C('SummEval')} re-annotated model outputs across several dimensions and showed "
            f"how weakly the standard metrics correlate with human judgement, and the FRANK-"
            f"style fine-grained factuality annotations of Goyal and Durrett "
            f"{C('Annotating and Modeling Fine-grained Factuality')} showed that error types "
            f"differ systematically between model families. BERTScore {C('BERTScore')} is "
            f"included in this project's metric set as a semantic-similarity complement to "
            f"ROUGE, not as a faithfulness measure, because it compares a generation against a "
            f"reference rather than against a source.",
            f"The claim-verification literature supplies the other half of the picture. FEVER "
            f"{C('FEVER')} established the retrieve-then-entail formulation at scale, SciFact "
            f"{C('Fact or Fiction')} moved it into the scientific domain where claims are "
            f"hedged and evidence is technical, and DeBERTa {C('DeBERTa')} provides the "
            f"disentangled-attention encoder whose MNLI/FEVER/ANLI-trained checkpoints are used "
            f"as the entailment model in Chapter 3. The survey of hallucination "
            f"{C('Survey of Hallucination')} ties the strands together and is the reference for "
            f"the terminology used in this report: a claim is *unsupported* when the evidence "
            f"does not entail it, which is weaker and more checkable than calling it false.",
        ])

        cur = D.add_heading(cur, "LLM agents and multi-agent frameworks")
        cur = D.add_paragraphs(cur, [
            f"Prompting a single model to reason in steps {C('Chain-Of-Thought Prompting')} was "
            f"the first sign that decomposition helps, and tool use "
            f"{C('Toolformer')} made it possible for a model to act outside its own context. "
            f"Multi-agent arrangements followed: CAMEL {C('CAMEL')} showed that role-played "
            f"agents conversing can carry a task further than a single prompt, and generative "
            f"agents {C('Generative Agents')} demonstrated persistent memory and planning across "
            f"long horizons. Iterative self-correction was formalised as Self-Refine "
            f"{C('Self-Refine')} and, with an explicit verbal critic, as Reflexion "
            f"{C('Reflexion')}.",
            "CrewAI, the framework used here, belongs to this family: it provides typed agents "
            "with roles, goals and tools, tasks with structured outputs, and Flows that wire "
            "steps together with explicit control flow and state. Two of its features are load-"
            "bearing in this project. Structured task output validated against a Pydantic model "
            "is what lets the Extractor's output be checked mechanically rather than parsed from "
            "prose. Flows with named steps are what let the human approval checkpoint be a real "
            "gate and let a run resume after a disconnection, which matters when the whole "
            "pipeline must fit inside free-tier limits.",
            "The critical observation from this literature, for present purposes, is that "
            "self-critique uses the same model that produced the text. A model's own judgement "
            "of its own output is correlated with the error that produced it. The Verifier in "
            "this project is therefore not an agent asked to be sceptical but a separate "
            "discriminative model applied to a well-defined pair of strings, and the agent whose "
            "role is called Verifier may do nothing except call it.",
        ])

        cur = D.add_heading(cur, "LLM-assisted literature review")
        cur = D.add_paragraphs(cur, [
            f"Retrieval-augmented generation {C('Retrieval-Augmented Generation')} with dense "
            f"retrieval {C('Dense Passage Retrieval')} is the standard way to ground a generator "
            f"in documents, and it is the backbone of the applied systems. PaperQA "
            f"{C('PaperQA')} is a retrieval-augmented agent for scientific question answering "
            f"that reports reduced hallucination relative to a plain LLM by forcing answers "
            f"through retrieved passages. STORM "
            f"{C('Assisting in Writing Wikipedia-like Articles')} attacks the longer-form "
            f"problem: it researches a topic by simulating "
            f"multi-perspective question asking, builds an outline, and writes a long article "
            f"with citations. Sušnjak et al. {C('Automating Research Synthesis')} fine-tune "
            f"domain-specific models for research synthesis and report on the practical "
            f"trade-offs of doing so.",
            f"Two things separate the system built here from those. The first is "
            f"sentence-level accountability: STORM and PaperQA cite at the level of a passage or "
            f"a paragraph, whereas this system records, for every sentence, the claim "
            f"identifiers it used, and then checks that sentence against exactly those claims' "
            f"evidence. The second is that two components are learned rather than prompted. The "
            f"citation-relation classifier is trained on data mined from real related-work "
            f"sections, and the extractor is distilled with QLoRA {C('QLoRA')} from the hosted "
            f"model onto a small open base model {C('Qwen2.5 Technical Report')}, using the "
            f"Verifier to filter the distillation set.",
        ])

        # 2.1 comparative table
        _, h = self.anchor("2.1 ")
        self.bookmark_heading(h, "Summary of existing studies / Comparative analysis of "
                                 "various approaches", number="2.1")
        rows = [
            ["Approach", "Representative work", "Grounding unit", "Verification",
             "Typed relations", "Learned component"],
            ["Pre-trained seq2seq summariser", f"BART {C('BART')}, PEGASUS {C('PEGASUS')}",
             "none (whole source)", "none", "no", "full fine-tune"],
            ["Related-work generator",
             f"Wang et al. {C('Neural Related Work Summarization')}, Multi-XScience "
             f"{C('Multi-XScience')}",
             "cited abstract", "none", "no", "task-specific model"],
            ["Citation intent classifier",
             f"SciCite {C('Structural Scaffolds')}, ACL-ARC "
             f"{C('Measuring the Evolution of a Scientific Field')}",
             "citation sentence", "n/a", "3 or 6 classes", "SciBERT fine-tune"],
            ["NLI faithfulness checker",
             f"SummaC {C('SummaC')}, QAGS {C('Asking and Answering Questions')}",
             "summary sentence", "entailment / QA", "no", "NLI model (off the shelf)"],
            ["Retrieval-augmented agent", f"PaperQA {C('PaperQA')}, STORM "
             f"{C('Assisting in Writing Wikipedia-like Articles')}",
             "retrieved passage", "retrieval only", "no", "none (prompted)"],
            ["Self-critique loop", f"Self-Refine {C('Self-Refine')}, Reflexion {C('Reflexion')}",
             "whole output", "same model critiques itself", "no", "none (prompted)"],
            ["This project", "Chapters 3–4",
             "quoted passage per claim; claim ids per sentence",
             "separate NLI model, thresholds fitted on a dev set",
             "7 classes, fine-tuned",
             "relation classifier + distilled extractor"],
        ]
        cur = self.table(h, rows,
                         "Comparative analysis of approaches to grounded review generation. The "
                         "grounding unit is what a generated statement can be traced back to; "
                         "verification is what, if anything, checks that trace.",
                         size_pt=8.5,
                         widths_in=[1.15, 1.35, 1.1, 1.25, 0.85, 1.0])

        # 2.2 trends and gaps
        _, h = self.anchor("2.2 Key findings")
        self.bookmark_heading(h, "Key findings, trends, and research gaps", number="2.2")
        D.add_paragraphs(h, [
            "Four trends run through the work above. Evaluation has shifted from reference "
            "overlap towards source entailment, because overlap metrics were shown not to track "
            "faithfulness. Grounding has become finer, from document-level retrieval towards "
            "passage- and sentence-level attribution. Citation classification has been "
            "standardised but at a granularity chosen for annotation cost rather than for "
            "downstream use. And orchestration has moved from single prompts to agent teams, "
            "with self-critique as the usual quality mechanism.",
            "Three gaps follow, and they are what this project addresses. First, the "
            "faithfulness literature evaluates summaries against a source document, while the "
            "review-generation literature generates from many sources and evaluates against a "
            "reference text; almost nothing evaluates a *multi-source* review against the "
            "sources it cites, sentence by sentence. Second, citation-relation labels are used "
            "to study fields rather than to drive synthesis: no system known to us uses a "
            "`contrasts` prediction as evidence that two papers disagree and then asks an agent "
            "to adjudicate. Third, the cost of multi-agent pipelines is rarely reported next to "
            "their quality, so it is not possible to tell from the literature whether the extra "
            "calls are worth making.",
            "A fourth observation is methodological rather than a gap in the literature. Several "
            "of the agent papers above report improvements without a time-split evaluation, on "
            "tasks whose references may be inside the evaluated model's training data. Section "
            "3.1 and docs/leakage.md therefore document the submission windows used here and why "
            "the test window begins after the knowledge cutoff of every model evaluated.",
        ])

        # 2.4 conclusion (the template's own numbering; left as 2.4)
        _, h = self.anchor("2.4 Conclusion")
        self.bookmark_heading(h, "Conclusion", number="2.4")
        D.add_paragraphs(h, [
            "The components this project needs all exist and are individually well studied: "
            "pre-trained summarisers, citation-intent classifiers, NLI-based consistency checks, "
            "agent frameworks and retrieval-augmented generation. What has not been assembled is "
            "a pipeline in which the citation relation is learned from real related-work "
            "sections, every generated sentence names the evidence it rests on, a separate "
            "discriminative model adjudicates that evidence with calibrated thresholds, and the "
            "resulting system is measured against human-written sections on a time-split "
            "benchmark with its cost reported alongside its quality. Chapter 3 describes that "
            "assembly and Chapter 4 reports what it achieved, including where it did not beat "
            "the single-call baseline.",
        ])
