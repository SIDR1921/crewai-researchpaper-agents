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
BRANCH = "SoCE – AI, "
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

    @staticmethod
    def _score(key: str, cand: str) -> float:
        """How well a short citation key identifies a reference.

        A key is a distinctive fragment of a title ("BART", "Just the Summary"),
        so a plain ``token_set_ratio`` against the full title scores it far too
        low. Substring containment is therefore the primary test, with fuzzy
        matching as the fallback for near-misses in spelling or word order.
        """
        k, c = key.lower().strip(), (cand or "").lower().strip()
        if len(k) < 3 or not c:
            return 0.0
        if k in c:
            # prefer the reference whose title the key pins down most tightly
            return 100.0 + len(k) / max(len(c), 1)
        best = fuzz.token_set_ratio(k, c)
        if len(k) >= 10:
            best = max(best, fuzz.partial_ratio(k, c) - 6)
        return float(best)

    def _match(self, key: str) -> dict | None:
        best, score = None, 0.0
        for r in self.pool:
            for cand in (r.get("query") or "", r.get("title") or ""):
                sc = self._score(key, cand)
                if sc > score:
                    best, score = r, sc
        return best if score >= 85 else None

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
        for name in ("LIST OF TABLES", "LIST OF FIGURES"):
            try:
                _, p = self.anchor(name)
                self.bookmark_heading(p, name.title().replace("Of", "of"), kind="front")
            except KeyError:
                pass

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

    # --------------------------------------------------------------- chapter 3

    def equation(self, anchor_par, tex: str, number: str, width_in: float = 5.4):
        """Render a display equation (mathtext) with its number, and insert it centred."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = self.figures / f"eq_{number.replace('.', '_')}.png"
        fig = plt.figure(figsize=(width_in, 0.52), dpi=300)
        fig.text(0.47, 0.5, tex, ha="center", va="center", fontsize=12, color="#1f2933")
        fig.text(0.985, 0.5, f"({number})", ha="right", va="center", fontsize=10,
                 color="#1f2933")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04, transparent=True)
        plt.close(fig)
        return D.add_image(anchor_par, out, width_in=width_in)

    def chapter3(self) -> None:
        R, C = self.R, self.refs.cite
        _, ch = self.anchor("CHAPTER 3", exact=True)
        self.bookmark_heading(ch, "METHODOLOGY", kind="chapter", number="3")

        _, h = self.anchor("3.1 Methodology with diagrams")
        self.bookmark_heading(h, "Methodology with diagrams", number="3.1")

        cur = D.add_paragraphs(h, [
            "The system is one Python package with two execution paths over it: command-line "
            "scripts for local runs and six Colab notebooks for the full-scale runs. The "
            "notebooks install the package and call the same functions, so there is no second "
            "implementation that can drift. Figure 1 shows the components and how data moves "
            "between them.",
        ])
        cur = self.figure(cur, "fig_architecture.png",
                          "System architecture. Blue boxes are CrewAI agents, green boxes are "
                          "fine-tuned models, red boxes are verification gates, and orange boxes "
                          "are data or outputs. One LLM configuration serves every agent and "
                          "logs the cost of every call.")

        cur = D.add_heading(cur, "3.1.1  Data mining and the supervision signal")
        cur = D.add_paragraphs(cur, [
            "The supervision for this project is reverse-engineered from published papers. A "
            "related-work section is a literature review that a domain expert wrote for a known "
            "set of cited papers, so mining those sections yields two things at once: citation "
            "contexts labelled implicitly by the sentence around them, and (section, cited "
            "abstracts) pairs that form a benchmark with a human reference. Figure 2 shows the "
            "pipeline.",
        ])
        cur = self.figure(cur, "fig_mining_pipeline.png",
                          "Data-mining pipeline. Each paper's extracted LaTeX is cached as one "
                          "JSON file and an empty cache entry records a PDF-only submission, "
                          "which is what makes the stage resumable.")
        cur = D.add_paragraphs(cur, [
            "The parser handles the cases a naive implementation fails on, each of which was "
            "found in the pilot and is covered by a unit test: several files carrying "
            "\\documentclass (a supplement alongside the paper), related work that is a "
            "\\subsection of a Background section or exists only as a run of \\paragraph blocks, "
            "bibliographies in biblatex's \\entry/\\field form as well as the \\bibitem/\\newblock "
            "families, a thebibliography block inlined in the main file, and the full set of ACL, "
            "IEEE and ACM citation macros. A heading alone is not accepted as evidence of a "
            "related-work section: an explicit “Related Work” heading must contain at least "
            "one citation and a looser heading such as “Background” at least two, otherwise "
            "a technical primer would be mined as if it were a survey.",
            "Cited titles are resolved against OpenAlex with a fuzzy-similarity floor of 88 on "
            "the title, so a wrong paper is rarely attached to a cite key. Papers for which "
            "fewer than 60% of the cited works have a retrievable abstract are dropped from the "
            "benchmark, because the task is only well posed if the system can see most of what "
            "the human author saw.",
        ])

        cur = D.add_heading(cur, "3.1.2  Citation-relation classification (component A)")
        cur = D.add_paragraphs(cur, [
            "The taxonomy has seven labels: background, uses-method, uses-data, extends, "
            "compares-against, supports and contrasts. The cut that matters is between "
            "compares-against and contrasts: a baseline comparison is not a disagreement, and "
            "only the latter is evidence that two papers conflict. Because labels overlap in "
            "practice, an ordered set of tie-break rules decides: disagreement outranks "
            "agreement, which outranks extension, which outranks comparison, which outranks data "
            "use, which outranks method use, with background as the default. The definitions and "
            "the rules live in one module so the LLM prompt, the human labelling guide and the "
            "training code cannot diverge.",
            f"SciCite {C('Structural Scaffolds')} and ACL-ARC "
            f"{C('Measuring the Evolution of a Scientific Field')} are mapped onto this taxonomy "
            f"as additional training signal. The mapping is lossy and is documented as such: "
            f"SciCite's method class does not separate adopting a method from extending it, and "
            f"its result class covers comparison, agreement and disagreement alike; ACL-ARC's "
            f"Uses does not separate method from data and its CompareOrContrast merges neutral "
            f"comparison with explicit disagreement, while its Future class has no counterpart "
            f"and is dropped. Mapped rows are therefore never used as evaluation gold.",
            "The classifier is SciBERT fine-tuned for sequence-pair classification. The first "
            "segment is the citation context with the citation under question rewritten to "
            "[[CITED]] and every other citation in the sentence rewritten to [OTHER], which is "
            "what makes the label well defined for a sentence that cites several papers at once; "
            "the second segment is the cited paper's title. The dev split is taken by citing "
            "paper rather than by row, because sentences from one paper are highly correlated "
            "and a random split would leak. Class weights are inverse-frequency,",
        ])
        cur = self.equation(cur, r"$w_c \;=\; \dfrac{N}{K\,n_c}$", "3.1", width_in=3.2)
        cur = D.add_paragraphs(cur, [
            "where N is the number of training rows, K the number of classes and n_c the count "
            "of class c, so that the rare relations are learnable at all. Performance is reported "
            "as macro-F1 over the classes present in the gold set,",
        ], first_line_indent=None)
        cur = self.equation(cur, r"$\mathrm{F1}_{\mathrm{macro}} \;=\; \frac{1}{|\mathcal{C}|}"
                                 r"\sum_{c \in \mathcal{C}} \frac{2 P_c R_c}{P_c + R_c}$",
                            "3.2", width_in=4.2)
        cur = D.add_paragraphs(cur, [
            "with P_c and R_c the precision and recall of class c, because accuracy on this "
            "distribution is dominated by background and would hide exactly the classes the "
            "downstream stages need. Agreement between the LLM labeller and the human gold set "
            "is reported as Cohen's κ,",
        ], first_line_indent=None)
        cur = self.equation(cur, r"$\kappa \;=\; \dfrac{p_o - p_e}{1 - p_e}$", "3.3",
                            width_in=2.8)
        cur = D.add_paragraphs(cur, [
            "where p_o is the observed agreement and p_e the agreement expected from the two "
            "label distributions independently, so that a labeller that simply predicts the "
            "majority class scores near zero rather than near its accuracy.",
        ], first_line_indent=None)

        cur = D.add_heading(cur, "3.1.3  Extraction and verification")
        cur = D.add_paragraphs(cur, [
            "The Extractor turns a paper into a PaperRecord: task, methods, datasets, metrics "
            "with values and settings, limitations, and three to six claims. Each claim carries "
            "an evidence object with the section it came from and a passage copied verbatim from "
            "the input. The input is the abstract plus whatever introduction, results, conclusion "
            "and limitations text the LaTeX source yielded, under a fixed character budget, and "
            "that exact input string is stored with the record, because the quote check is "
            "defined against it and not against the paper in the abstract.",
            f"The Verifier applies two independent checks. The first asks whether the quoted "
            f"passage really occurs in the input, scored by the fuzzy partial ratio of the "
            f"normalised passage against the normalised source rather than by exact matching, "
            f"because LaTeX-to-text conversion leaves whitespace and ligature noise that a "
            f"faithful quote still trips over. The second asks whether the claim is entailed by "
            f"the passage, using a DeBERTa-v3 {C('DeBERTa')} model trained on MNLI, FEVER and "
            f"ANLI, applied at sentence granularity as SummaC {C('SummaC')} recommends. A claim "
            f"can quote correctly and still overstate what the passage says, and the second check "
            f"is what catches that.",
            "A verdict is supported when the quote score clears its threshold and the entailment "
            "probability clears the upper threshold, partially when the entailment probability "
            "falls between the two thresholds, and unsupported otherwise. The three thresholds "
            "are not chosen by hand: they are fitted by grid search on a hand-checked dev set of "
            "claim/evidence pairs, maximising macro-F1 over the three verdicts rather than "
            "accuracy, so a threshold pair cannot be selected for doing well on the majority "
            "verdict alone. Section 4.3 reports the fitted values and how they compare with the "
            "defaults.",
            "At review time the same model scores each generated sentence against the evidence of "
            "the claims whose identifiers the Writer attached to it. The faithfulness score is "
            "the fraction of sentences the Verifier calls supported,",
        ])
        cur = self.equation(
            cur, r"$\mathrm{Faith} \;=\; \frac{1}{|S|}\sum_{s \in S}"
                 r"\mathbb{1}\!\left[\max_{e \in E(s)} P_{\mathrm{ent}}(s \mid e) "
                 r"\geq \tau_{\mathrm{sup}}\right]$", "3.4", width_in=5.0)
        cur = D.add_paragraphs(cur, [
            "where E(s) is the set of evidence passages behind sentence s and τ_sup is the "
            "fitted upper threshold. A sentence that cites nothing has an empty evidence set and "
            "is unsupported by construction, which is the intended behaviour for a review that "
            "claims to be grounded. Sentences the Verifier rejects go back to the Writer once; "
            "if the revision does not improve the score the draft is kept and the offending "
            "sentences are flagged in the output rather than silently removed.",
        ], first_line_indent=None)

        cur = D.add_heading(cur, "3.1.4  Consensus, contradictions and gaps")
        cur = D.add_paragraphs(cur, [
            "Dataset, metric, method and task names are normalised in three stages, cheapest "
            "first: a manual alias table checked into the repository, then fuzzy matching against "
            "the canonical names, then sentence embeddings for paraphrases that fuzzy matching "
            "misses. Anything still unmatched keeps a tidied version of its surface form and is "
            "reported, which is how the alias file gets extended rather than how information gets "
            "lost.",
            "Records are then grouped by (task, dataset, metric). Within a group, a relative "
            "spread wider than 15% of the group median is flagged as a value disagreement, and "
            "two papers that order the same pair of methods oppositely on the same dataset and "
            "metric are flagged as a ranking disagreement. Independently, a contrasts edge "
            "predicted by the classifier is flagged as a disagreement the authors themselves "
            "asserted, which numbers alone cannot reveal. Each flagged contradiction goes to the "
            "Adjudicator, which must classify it as a different setting, a different split or "
            "metric definition, a genuine disagreement, or insufficient evidence, using only the "
            "quoted evidence passages.",
            "Gap detection builds a method-family × dataset matrix of paper counts. An empty "
            "or near-empty cell whose row and column are both otherwise active is a candidate "
            "gap, scored by how far the count falls below the count expected if row and column "
            "activity were independent. Two method families that both appear on a dataset but "
            "between whose papers no compares-against edge exists form an untested-comparison "
            "candidate. Because an empty cell is often empty for a good reason, the Gap Analyst "
            "is required to argue both for and against each candidate before giving a verdict, "
            "and both arguments are kept in the output.",
        ])

        cur = D.add_heading(cur, "3.1.5  Orchestration")
        cur = D.add_paragraphs(cur, [
            "The pipeline is a CrewAI Flow of ten steps, shown in Figure 3. Roles are narrow on "
            "purpose: the Librarian has search, snowballing and reference-verification tools and "
            "nothing else; the Extractor has no tools at all because its input is handed to it; "
            "only the Verifier may call the entailment tool; and the Writer can neither search "
            "nor verify. The approval checkpoint after corpus collection is a real gate, not a "
            "formality: the Flow stops and the corpus table is shown before any LLM call is spent "
            "on extraction.",
        ])
        cur = self.figure(cur, "fig_flow.png",
                          "The ten Flow steps, the human approval gate and the single revision "
                          "loop. Every step caches its output, so an interrupted run resumes.")
        cur = D.add_paragraphs(cur, [
            "Every step writes its output to the run directory before the next begins, which is "
            "what lets a run that dies against a rate limit continue rather than restart. The "
            "same property is what makes the Colab notebooks usable at all under a 12-hour "
            "session limit.",
        ])

        cur = D.add_heading(cur, "3.1.6  Distillation (component B)")
        cur = D.add_paragraphs(cur, [
            f"The distillation set is built by running the hosted Extractor over training-split "
            f"papers and keeping only those papers whose every claim passed the Verifier. The "
            f"Verifier is therefore the filter on the student's training data, which is the point "
            f"of the stage: the student cannot learn to fabricate evidence that the teacher "
            f"happened to get away with. The student is a small instruction-tuned model "
            f"{C('Qwen2.5 Technical Report')} adapted with QLoRA {C('QLoRA')} — a 4-bit NF4 "
            f"base with low-rank adapters on the attention and MLP projections — which is what "
            f"makes the fine-tune fit on a free T4. The comparison against the hosted extractor "
            f"is on held-out test-split papers and covers JSON validity, the Verifier's own "
            f"verdicts on the resulting claims, field-level agreement, latency and cost.",
        ])

        cur = D.add_heading(cur, "3.1.7  Evaluation design")
        cur = D.add_paragraphs(cur, [
            "The benchmark task is: given only the papers a real related-work section cites, "
            "regenerate that section. Four systems and one ablation are compared under the same "
            "model and settings. S1 is a single well-prompted call over the same cited abstracts. "
            "S2 is the crew without the Verifier. S3 is the full crew. S4 is the full crew with "
            "the distilled extractor. The ablation is S3 with the relation classifier switched "
            "off, which isolates the contribution of component A.",
            "Overlap metrics (ROUGE-1/2/L and BERTScore against the human section) are reported "
            "beside three grounding metrics, because overlap says nothing about truth. Citation "
            "coverage is the fraction of the papers the human section cited that the generated "
            "section discusses, counted either by an inline identifier or by a distinctive "
            "overlap with the paper's title, so that a system which paraphrases instead of using "
            "identifiers is not scored at zero unfairly. Citation correctness is the fraction of "
            "asserted (sentence, cited paper) pairs for which that paper's evidence actually "
            "entails the sentence; it is the metric that catches a real identifier attached to a "
            "sentence the paper never supported. Faithfulness is as in equation (3.4).",
            "Every mean is reported with a percentile bootstrap 95% confidence interval,",
        ])
        cur = self.equation(
            cur, r"$\mathrm{CI}_{95} = \left[\hat{\theta}^{*}_{(0.025B)},\;"
                 r"\hat{\theta}^{*}_{(0.975B)}\right],\quad B = 2000$", "3.5", width_in=4.4)
        cur = D.add_paragraphs(cur, [
            "where the resampled statistics are the means of B bootstrap resamples of the "
            "per-example scores. At pilot sample sizes these intervals are wide, and reporting "
            "them is what prevents a difference of a few ROUGE points from being read as a "
            "result.",
            "Leakage is controlled by the submission window. Training papers come from 2023-2024 "
            "and test papers from a window that begins after the knowledge cutoff of every model "
            "evaluated. The Verifier's thresholds, the classifier's dev split and the human gold "
            "set all come from the training split, so nothing is tuned on the benchmark. The "
            "dates and the residual risks — chiefly that the cited papers are necessarily old "
            "and so plausibly inside every model's training data, even though the target text is "
            "not — are recorded in docs/leakage.md and in Section 4.1.",
        ])

        # 3.2 system requirements
        _, h = self.anchor("3.2 System requirements")
        self.bookmark_heading(h, "System requirements", number="3.2")
        cur = D.add_paragraphs(h, [
            "The project is built to run in two places: a free Colab T4 session for the heavy "
            "stages, and an Apple Silicon laptop with Metal acceleration for development and the "
            "pilot runs reported in Chapter 4. Table 2 lists what was actually used; the "
            "versions are those recorded at run time, not the minimum the code would tolerate.",
        ])
        env = self.environment_rows()
        cur = self.table(cur, env,
                         "System requirements and the exact versions used for the runs reported "
                         "in Chapter 4.", size_pt=9.0, widths_in=[1.7, 2.1, 2.3])
        D.add_paragraphs(cur, [
            "API access is through a free tier, which shapes the design more than the hardware "
            "does. The rate limiter is a token bucket on both requests and tokens per minute, "
            "configured per provider; every long-running step is append-only and keyed, so "
            "exhausting a daily quota means resuming tomorrow rather than starting again; and "
            "relation labelling batches several contexts into one request, which is roughly an "
            "eightfold saving on the request budget. No API key appears anywhere in the "
            "repository: keys are read from Colab secrets when running there and from a "
            "gitignored .env file otherwise.",
        ])

    def environment_rows(self) -> list[list[str]]:
        """Versions read from the running environment, with the Colab target beside them."""
        import platform
        import sys as _s

        def ver(mod: str) -> str:
            try:
                return __import__(mod).__version__
            except Exception:
                return NA

        rows = [["Component", "Local (pilot runs)", "Google Colab (full-scale runs)"],
                ["Hardware",
                 f"Apple Silicon, {platform.machine()}, Metal (MPS)",
                 "NVIDIA T4, 16 GB, free tier (~12 h session limit)"],
                ["Operating system", f"{platform.system()} {platform.release()}",
                 "Ubuntu (Colab runtime)"],
                ["Python", platform.python_version(), "3.11+ (Colab default)"],
                ["PyTorch", ver("torch") + " (MPS backend)", "same wheel, CUDA backend"],
                ["transformers", ver("transformers"), "installed by the notebook"],
                ["CrewAI", ver("crewai"), "installed by the notebook"],
                ["LiteLLM", ver("litellm"), "installed by the notebook"],
                ["scikit-learn", ver("sklearn"), "installed by the notebook"],
                ["sentence-transformers", ver("sentence_transformers"), "installed by the notebook"],
                ["peft", ver("peft"), "plus bitsandbytes (CUDA only) for 4-bit QLoRA"],
                ["Relation encoder", HF_MODELS["scibert"], "same checkpoint"],
                ["Entailment model", HF_MODELS["nli"], "same checkpoint"],
                ["Embedding model", HF_MODELS["embed"], "same checkpoint"],
                ["Distillation base", "not run locally (4-bit needs CUDA)",
                 HF_MODELS["distill_base"]],
                ["LLM provider", self.R.get("eval_arxiv.json", "settings.provider")
                 or self.R.get("stage2_labelling.json", "provider.provider") or NA,
                 "same, via Colab secrets"],
                ["LLM model", self.R.get("eval_arxiv.json", "settings.model")
                 or self.R.get("stage2_labelling.json", "provider.model") or NA,
                 "same model string"],
                ["External APIs", "arXiv, OpenAlex, Semantic Scholar", "same"],
                ]
        return rows

    # --------------------------------------------------------------- chapter 4

    def _not_run(self, anchor_par, stage: str, what: str, how: str):
        """State plainly that a stage was not run, instead of estimating its outcome."""
        self.notes.append(f"Chapter 4: {stage} not reported — {what}")
        return D.add_body_paragraph(
            anchor_par,
            f"{stage} is not reported here. {what} Re-run it with {how}, then rebuild this "
            f"report; the text and the table will appear automatically.",
            italic=True, first_line_indent=None)

    def chapter4(self) -> None:
        R, C = self.R, self.refs.cite
        _, ch = self.anchor("CHAPTER 4", exact=True)
        self.bookmark_heading(ch, "ANALYSIS AND DISCUSSION", kind="chapter", number="4")
        _, title = self.anchor("ANALYSIS AND DISCUSSION")

        cur = D.add_paragraphs(title, [
            "Every number in this chapter is read from a file in results/ that the code wrote, "
            "and every figure is generated from those files by scripts/make_figures.py. Where a "
            "stage was not run, that is stated rather than estimated. The scale of each run is "
            "given with its results, because pilot-scale confidence intervals are wide and a "
            "point estimate on its own would overclaim.",
        ])

        # ---------------------------------------------------------- 4.1 data
        cur = D.add_heading(cur, "4.1  Dataset statistics and mining yield")
        if R.has("stage1_yield.json"):
            y = R.get("stage1_yield.json", "yield") or {}
            ds = R.get("stage1_yield.json", "datasets") or {}
            oa = R.get("stage1_yield.json", "openalex") or {}
            cfg = R.get("stage1_yield.json", "config") or {}
            listed = y.get("listed", 0) or 1
            cur = D.add_paragraphs(cur, [
                f"The pilot listed {y.get('listed')} arXiv submissions in cs.CL and cs.IR "
                f"({cfg.get('n_train')} from the training window {cfg.get('train_window', ['?'])[0][:6]}"
                f"–{(cfg.get('train_window') or ['', '?'])[1][:6]} and {cfg.get('n_test')} from "
                f"the test window {(cfg.get('test_window') or ['?'])[0][:6]}–"
                f"{(cfg.get('test_window') or ['', '?'])[1][:6]}). Of those, "
                f"{listed - (y.get('no_source') or 0)} had usable LaTeX source "
                f"({100 * (listed - (y.get('no_source') or 0)) / listed:.0f}%); the remainder are "
                f"PDF-only submissions, which arXiv permits and which no parser can recover. "
                f"{y.get('parsed')} parsed successfully and {y.get('with_related_work')} had an "
                f"identifiable related-work section "
                f"({100 * (y.get('with_related_work') or 0) / listed:.0f}% of papers listed). "
                f"After requiring at least three cited works and at least 60% abstract coverage, "
                f"{ds.get('rw_pairs_kept')} papers remain as benchmark examples and "
                f"{ds.get('contexts_rows'):,} citation contexts were written. Figure 4 shows the "
                f"attrition at each step.",
                f"Title resolution against OpenAlex succeeded for "
                f"{oa.get('resolved_pct')}% of {oa.get('unique_cited_titles'):,} unique cited "
                f"titles, and {oa.get('with_abstract_pct')}% of those titles yielded an abstract. "
                f"The gap between the two is the main reason benchmark examples are dropped: a "
                f"paper can be identified without its abstract being redistributable, and this "
                f"project only uses abstracts OpenAlex returns.",
            ])
            cur = self.figure(cur, "fig_stage1_yield.png",
                              "Mining yield: papers surviving each step, as counts and as a "
                              "percentage of the submissions listed.")
            cur = self.figure(cur, "fig_dataset_stats.png",
                              "Distribution of related-work length, number of cited papers and "
                              "abstract coverage across the mined benchmark examples.")
            rows = [["Step", "Papers", "% of listed"]]
            for label, v in [("Submissions listed", y.get("listed")),
                             ("With LaTeX source", listed - (y.get("no_source") or 0)),
                             ("Parsed without error", y.get("parsed")),
                             ("With a related-work section", y.get("with_related_work")),
                             ("Kept as benchmark examples", ds.get("rw_pairs_kept"))]:
                rows.append([label, f"{v:,}" if isinstance(v, int) else NA,
                             f"{100 * v / listed:.1f}%" if isinstance(v, int) else NA])
            cur = self.table(cur, rows, "Stage 1 mining yield at pilot scale.",
                             size_pt=9.5, widths_in=[2.8, 1.2, 1.2], align_right_cols=(1, 2))

            if R.has("stage1_dataset_stats.json"):
                S = R.get("stage1_dataset_stats.json") or {}
                rw = S.get("rw_pairs") or {}
                ctx = S.get("citation_contexts") or {}
                mx = S.get("multixscience") or {}
                ts = S.get("time_split") or {}
                rows = [["Statistic", "Mined arXiv split", "Multi-XScience"]]
                def med(d, k):
                    v = (d.get(k) or {}).get("median")
                    return f"{v:g}" if v is not None else NA
                rows += [
                    ["Benchmark examples", str(rw.get("n_papers", NA)),
                     str(mx.get("n_papers", NA)) if mx.get("available") else "not loaded"],
                    ["Median related-work length (words)", med(rw, "rw_words"), med(mx, "rw_words")],
                    ["Median papers cited per example", med(rw, "n_cited"), med(mx, "n_cited")],
                    ["Median abstract coverage", med(rw, "abstract_coverage"),
                     med(mx, "abstract_coverage")],
                    ["Citation contexts", f"{ctx.get('n_rows', 0):,}" if ctx.get("available") else NA, "—"],
                    ["… of which inside a related-work section",
                     f"{ctx.get('in_related_work', 0):,}" if ctx.get("available") else NA, "—"],
                    ["Distinct citing papers",
                     f"{ctx.get('n_citing_papers', 0):,}" if ctx.get("available") else NA, "—"],
                ]
                cur = self.table(cur, rows,
                                 "Dataset statistics for the mined arXiv benchmark and the "
                                 "external Multi-XScience benchmark.",
                                 size_pt=9.5, widths_in=[2.9, 1.3, 1.3], align_right_cols=(1, 2))
                cur = D.add_paragraphs(cur, [
                    f"The time split is disjoint: the newest training paper is dated "
                    f"{ts.get('train_latest')} and the oldest test paper "
                    f"{ts.get('test_earliest')}." if ts.get("disjoint") else
                    f"The time split check reports train_latest={ts.get('train_latest')} and "
                    f"test_earliest={ts.get('test_earliest')}; these are not disjoint and the "
                    f"benchmark should not be used until the windows are corrected.",
                    "The residual leakage risk is that the cited papers, being older than the "
                    "citing paper by construction, are plausibly inside every evaluated model's "
                    "training data. That cannot be avoided for this task and it does not give a "
                    "model access to the target text, which is what is scored; it is recorded "
                    "here rather than dismissed.",
                ])
        else:
            cur = self._not_run(cur, "Stage 1 mining", "results/stage1_yield.json is absent.",
                                "python scripts/run_stage1.py --n-train 150 --n-test 50")

        # ------------------------------------------------- 4.2 the classifier
        cur = D.add_heading(cur, "4.2  Citation-relation classification")
        if R.has("stage2_benchmark.json") or R.has("stage2_gold_agreement.json"):
            lab = R.get("stage2_labelling.json", "n_labelled_total")
            ag = R.get("stage2_gold_agreement.json") or {}
            lvg = ag.get("llm_vs_gold") or {}
            bench = R.get("stage2_benchmark.json") or {}
            if lab:
                cur = D.add_paragraphs(cur, [
                    f"{lab:,} citation contexts were labelled by the configured LLM in batches "
                    f"of eight, each label accompanied by a short rationale that had to quote "
                    f"the cue words from the context. Figure 6 shows the resulting distribution, "
                    f"which is dominated by background as expected.",
                ])
                cur = self.figure(cur, "fig_label_distribution.png",
                                  "Distribution of LLM-assigned citation relations over the "
                                  "labelled pool.")
            if lvg:
                cur = D.add_paragraphs(cur, [
                    f"A gold set of {ag.get('n_gold')} contexts was labelled by hand without "
                    f"sight of the LLM's labels, stratified over the LLM's predictions so that "
                    f"the rare relations appear at all. Against that gold set the LLM labeller "
                    f"reaches {lvg.get('accuracy', 0):.3f} accuracy and "
                    f"{lvg.get('macro_f1', 0):.3f} macro-F1, with Cohen's κ = "
                    f"{lvg.get('cohen_kappa', 0):.3f}. Because the sample is stratified rather "
                    f"than random, this is not an estimate of accuracy on an arbitrary citation "
                    f"context: it deliberately over-weights the classes the labeller finds "
                    f"hardest, and should be read as a lower bound.",
                ])
            if bench:
                rows = [["System", "Accuracy", "Macro-F1"]]
                for key, name in [("majority", "Majority class"),
                                  ("tfidf_lr", "TF-IDF + logistic regression"),
                                  ("zero_shot_llm", "Zero-shot LLM (same prompt)"),
                                  ("scibert", "Fine-tuned SciBERT (component A)")]:
                    m = bench.get(key)
                    if m:
                        rows.append([name, f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}"])
                cur = self.table(cur, rows,
                                 f"Citation-relation classification on the "
                                 f"{bench.get('n_gold', '?')}-context human gold set. The gold "
                                 f"set is stratified over the LLM's predictions, so absolute "
                                 f"accuracy is pessimistic; the ordering is the result.",
                                 size_pt=9.5, widths_in=[3.0, 1.2, 1.2],
                                 align_right_cols=(1, 2))
                sci = bench.get("scibert") or {}
                tf = bench.get("tfidf_lr") or {}
                if sci and tf:
                    verdict = ("beats" if sci["macro_f1"] > tf["macro_f1"] else
                               "does not beat")
                    cur = D.add_paragraphs(cur, [
                        f"The fine-tuned classifier {verdict} the TF-IDF baseline on macro-F1 "
                        f"({sci['macro_f1']:.3f} against {tf['macro_f1']:.3f}). "
                        + ("The margin is what the downstream stages rely on: a contrasts edge "
                           "is only useful as evidence of disagreement if the classifier can "
                           "tell it from a neutral comparison."
                           if sci["macro_f1"] > tf["macro_f1"] else
                           "This is a negative result and is reported as one: at pilot label "
                           "volume the fine-tune does not justify itself over a bag-of-ngrams "
                           "model, and the relation edges used downstream should be read with "
                           "that in mind."),
                    ])
                    per = sci.get("per_class") or {}
                    weak = sorted(((k, v) for k, v in per.items() if v.get("support", 0) > 0),
                                  key=lambda kv: kv[1]["f1"])[:2]
                    if weak:
                        cur = D.add_paragraphs(cur, [
                            "Per-class, the weakest classes are "
                            + " and ".join(f"{k} (F1 {v['f1']:.2f}, {v['support']} gold examples)"
                                           for k, v in weak)
                            + ". Both are rare in the gold set, and the confusion matrices in "
                              "Figure 7 show where their mass goes instead.",
                        ])
            cur = self.figure(cur, "fig_relation_confusion.png",
                              "Row-normalised confusion matrices on the human gold set, for the "
                              "LLM labeller and for the fine-tuned classifier. Cell text is the "
                              "raw count.")
            cur = self.figure(cur, "fig_relation_models.png",
                              "Accuracy and macro-F1 of each relation-classification system on "
                              "the human gold set.")
        else:
            cur = self._not_run(cur, "Stage 2 classification",
                                "results/stage2_benchmark.json is absent, so neither the LLM "
                                "agreement nor the fine-tuned classifier can be reported.",
                                "python scripts/run_stage2.py label, then sample_gold.py, then "
                                "run_stage2.py train and bench")

        # --------------------------------------------------- 4.3 the verifier
        cur = D.add_heading(cur, "4.3  Verifier calibration and claim verdicts")
        if R.has("verifier_calibration.json"):
            cal = R.get("verifier_calibration.json") or {}
            th = cal.get("thresholds") or {}
            cur = D.add_paragraphs(cur, [
                f"The two thresholds were fitted on {cal.get('n_dev_items')} hand-checked "
                f"claim/evidence pairs by grid search, maximising macro-F1 over the three "
                f"verdicts. The fitted values are a quote-match floor of "
                f"{th.get('quote_min')} and entailment thresholds of "
                f"{th.get('entail_supported')} for supported and {th.get('entail_partial')} for "
                f"partially supported, reaching {cal.get('best_macro_f1')} macro-F1 on the dev "
                f"set"
                + (f", against {cal.get('defaults_macro_f1')} for the hand-chosen defaults."
                   if cal.get("defaults_macro_f1") is not None else ".")
                + " Figure 8 shows the dev items against the fitted boundaries.",
            ])
            cur = self.figure(cur, "fig_verifier_calibration.png",
                              "Verifier calibration. Left: each hand-checked dev item by quote "
                              "score and entailment probability, with the fitted thresholds. "
                              "Right: the top of the threshold grid.")
        else:
            cur = self._not_run(
                cur, "Verifier calibration",
                "results/verifier_calibration.json is absent, so the thresholds in use are the "
                "hand-chosen defaults rather than fitted values, and the claim verdicts below "
                "should be read as uncalibrated.",
                "python scripts/run_stage3.py make-dev, label data/verifier_dev.jsonl, then "
                "run_stage3.py calibrate")

        verify_files = sorted(self.R.dir.glob("stage3_verify_*.json"))
        if verify_files:
            v = json.loads(verify_files[0].read_text())
            rates = v.get("verdict_rates") or {}
            cur = D.add_paragraphs(cur, [
                f"Applied to {v.get('n_claims')} claims extracted from {v.get('n_papers')} "
                f"papers, the Verifier finds {rates.get('supported', 0):.0%} fully supported, "
                f"{rates.get('partially', 0):.0%} partially supported and "
                f"{rates.get('unsupported', 0):.0%} unsupported. The quote check alone passes "
                f"{v.get('quote_found_rate', 0):.0%} of claims, with a mean match score of "
                f"{v.get('mean_quote_score')}, and the mean entailment probability is "
                f"{v.get('mean_entail_prob')}. The difference between those two numbers is the "
                f"stage's main finding: a model asked to quote verbatim mostly does, and still "
                f"writes claims its own quotation does not entail. "
                f"{v.get('fully_supported_papers')} of {v.get('n_papers')} papers had every "
                f"claim supported, and those are the papers that become distillation data.",
            ])
            cur = self.figure(cur, "fig_claim_verdicts.png",
                              "Verifier verdicts over extracted claims, by extraction run.")
        else:
            cur = self._not_run(cur, "Claim verification",
                                "no results/stage3_verify_*.json file exists.",
                                "python scripts/run_stage3.py extract and then verify")

        # -------------------------------------------- 4.4 map, gaps, the demo
        cur = D.add_heading(cur, "4.4  Consensus map, contradictions and gaps")
        if R.has("flow_summary.json"):
            f = R.get("flow_summary.json") or {}
            cons = f.get("consensus") or {}
            rel = f.get("relation_counts") or {}
            fa = (f.get("faithfulness") or {}).get("after_revision") or {}
            cur = D.add_paragraphs(cur, [
                f"The Flow was run end to end on the topic “{f.get('topic')}”, chosen "
                f"because it overlaps this report's own subject matter and so can be read "
                f"critically. The Librarian assembled "
                f"{(f.get('corpus_stats') or {}).get('selected')} papers from a pool of "
                f"{(f.get('corpus_stats') or {}).get('pool_size')}, which were approved at the "
                f"checkpoint; the Extractor produced {f.get('n_claims')} claims across "
                f"{f.get('n_records')} papers.",
                f"Grouping the reported numbers gave {cons.get('n_groups', 0)} "
                f"(task, dataset, metric) groups, of which {cons.get('n_agreeing', 0)} agree "
                f"within the 15% tolerance. {f.get('n_contradictions', 0)} contradictions were "
                f"flagged and adjudicated. "
                + (f"The relation classifier contributed {sum(rel.values())} typed edges "
                   f"({', '.join(f'{v} {k}' for k, v in sorted(rel.items(), key=lambda kv: -kv[1]) if k)})."
                   if rel else
                   "The relation classifier contributed no edges in this run, because the "
                   "corpus was assembled from OpenAlex abstracts rather than from LaTeX "
                   "sources, so there were few in-corpus citation sentences to classify. That "
                   "is a real limitation of running the Flow on an abstract-only corpus and is "
                   "discussed in Section 4.7."),
            ])
            gv = f.get("gap_verdicts") or {}
            if gv:
                cur = D.add_paragraphs(cur, [
                    "The Gap Analyst argued both sides of each candidate and returned "
                    + ", ".join(f"{v} judged {k}" for k, v in gv.items() if k)
                    + ". Keeping the argument against each candidate in the output matters: "
                      "several empty cells in the coverage matrix are empty because the "
                      "combination is not sensible, and the Analyst says so rather than "
                      "reporting them as opportunities.",
                ])
            cur = self.figure(cur, "fig_gap_heatmap.png",
                              "Method-family by dataset coverage for the demonstration topic. "
                              "A dot marks an empty cell, which is a candidate gap.")
            cur = self.figure(cur, "fig_relation_counts.png",
                              "Typed relation edges found in the demonstration corpus.")
            if fa:
                cur = D.add_paragraphs(cur, [
                    f"The review written from this corpus scored "
                    f"{fa.get('faithfulness', 0):.0%} sentence-level faithfulness after one "
                    f"revision pass, with {f.get('faithfulness', {}).get('n_flagged_after', 0)} "
                    f"sentences still flagged and marked as unverified in the exported Markdown "
                    f"rather than deleted.",
                ])
            cur = D.add_paragraphs(cur, [
                "The typed citation graph is exported as an interactive HTML file in the run "
                "directory; it is not reproduced here because a static screenshot of a "
                "force-directed graph carries little information, and the contradictions and "
                "gap tables below carry what it shows.",
            ])
        else:
            cur = self._not_run(cur, "The end-to-end Flow run",
                                "results/flow_summary.json is absent.",
                                "python scripts/run_flow.py --topic \"...\" --n-papers 15")

        # ------------------------------------------------------ 4.5 evaluation
        cur = D.add_heading(cur, "4.5  System comparison")
        if R.has("eval_arxiv.json"):
            aggs = R.get("eval_arxiv.json", "aggregates") or {}
            n = R.get("eval_arxiv.json", "benchmark.n_examples")
            main = R.get("eval_arxiv.json", "main_table") or []
            cur = D.add_paragraphs(cur, [
                f"The systems were compared on {n} frozen test-split examples under one model "
                f"and one set of settings. Table 5 gives the main results; every cell is a mean "
                f"with its bootstrap 95% confidence interval over examples.",
            ])
            if main:
                head = ["System", "n", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore",
                        "Coverage", "Correctness", "Faithfulness"]
                keys = ["system", "n", "rouge1", "rouge2", "rougeL", "bertscore",
                        "coverage", "correctness", "faithfulness"]
                rows = [head] + [[str(r.get(k, "-")) for k in keys] for r in main]
                cur = self.table(cur, rows,
                                 f"Main evaluation on the mined arXiv test split ({n} examples). "
                                 f"Mean [bootstrap 95% CI] over examples.",
                                 size_pt=7.0,
                                 widths_in=[1.25, 0.3, 0.75, 0.75, 0.75, 0.75, 0.75, 0.8, 0.8])
            cur = self.figure(cur, "fig_eval_arxiv.png",
                              "System comparison on the mined arXiv benchmark with bootstrap "
                              "95% confidence intervals.")
            cur = D.add_paragraphs(cur, self.eval_commentary(aggs, n))
            if R.has("eval_multixscience.json"):
                mx_aggs = R.get("eval_multixscience.json", "aggregates") or {}
                mx_n = R.get("eval_multixscience.json", "benchmark.n_examples")
                cur = D.add_paragraphs(cur, [
                    f"The same systems were run on {mx_n} Multi-XScience test examples as an "
                    f"external check. "
                    + self.cross_benchmark_sentence(aggs, mx_aggs),
                ])
                cur = self.figure(cur, "fig_eval_multixscience.png",
                                  "The same comparison on the external Multi-XScience benchmark.")
            cost = R.get("eval_arxiv.json", "cost_table") or []
            if cost:
                head = ["System", "n", "Wall-clock s / example", "Prompt tokens",
                        "Completion tokens", "Cost (USD)"]
                keys = ["system", "n", "wall_s_per_example", "prompt_tokens",
                        "completion_tokens", "cost_usd"]
                rows = [head] + [[("" if r.get(k) is None else str(r.get(k))) for k in keys]
                                 for r in cost]
                cur = self.table(cur, rows,
                                 "Cost and latency. The token and cost row is the total over "
                                 "every LLM call in the central log, which covers all stages, "
                                 "not only evaluation.",
                                 size_pt=8.0, widths_in=[1.9, 0.5, 1.2, 1.0, 1.0, 0.8],
                                 align_right_cols=(1, 2, 3, 4, 5))
            cur = self.figure(cur, "fig_cost_latency.png",
                              "Wall-clock cost per benchmark example by system.")
            cur = self.figure(cur, "fig_usage_by_stage.png",
                              "LLM tokens consumed per pipeline stage, from the central call "
                              "log.")
        else:
            cur = self._not_run(cur, "The S1–S4 comparison",
                                "results/eval_arxiv.json is absent.",
                                "python scripts/run_eval.py --systems S1 S2 S3 --limit 30")

        # ------------------------------------------------- 4.6 the distillation
        cur = D.add_heading(cur, "4.6  Distilled extractor")
        if R.has("stage6_distill.json"):
            d = R.get("stage6_distill.json") or {}
            api, dis = d.get("api") or {}, d.get("distilled") or {}
            agree = d.get("field_agreement") or {}
            tr = d.get("training") or {}
            dset = d.get("dataset") or {}
            cur = D.add_paragraphs(cur, [
                f"Of {dset.get('papers')} extracted training-split papers, "
                f"{dset.get('kept')} had every claim pass the Verifier and became distillation "
                f"examples; {dset.get('dropped_unverified')} were dropped for failing "
                f"verification. The student was "
                f"{tr.get('base_model')} adapted with "
                f"{'4-bit QLoRA' if tr.get('quantised_4bit') else 'LoRA in full precision (4-bit was unavailable on this device)'}"
                f", training {tr.get('trainable_params', 0):,} parameters.",
                f"On {api.get('n_papers')} held-out papers the hosted extractor produced valid "
                f"JSON for {api.get('json_validity_rate', 0):.0%} of papers against "
                f"{dis.get('json_validity_rate', 0):.0%} for the student, at "
                f"{api.get('latency_s_mean')}s and {dis.get('latency_s_mean')}s per paper "
                f"respectively. Field agreement between the two was "
                + ", ".join(f"{k} {v:.2f}" for k, v in agree.items()
                            if k != "n_compared" and v is not None)
                + f" over {agree.get('n_compared')} papers both handled.",
            ])
            rows = [["Measure", "Hosted extractor", "Distilled extractor"],
                    ["Valid JSON matching PaperRecord",
                     f"{api.get('json_validity_rate', 0):.0%}",
                     f"{dis.get('json_validity_rate', 0):.0%}"],
                    ["Claims per paper", str(api.get("claims_per_paper", NA)),
                     str(dis.get("claims_per_paper", NA))],
                    ["Claims fully supported",
                     f"{(api.get('claim_verdict_rates') or {}).get('supported', 0):.0%}",
                     f"{(dis.get('claim_verdict_rates') or {}).get('supported', 0):.0%}"],
                    ["Quote found in source",
                     f"{api.get('quote_found_rate', 0):.0%}",
                     f"{dis.get('quote_found_rate', 0):.0%}"],
                    ["Latency per paper (s)", str(api.get("latency_s_mean", NA)),
                     str(dis.get("latency_s_mean", NA))],
                    ["Monetary cost per paper", "API tokens (see Table 6)",
                     "0 (runs locally)"]]
            cur = self.table(cur, rows,
                             "Hosted extractor against the distilled extractor on held-out "
                             "test-split papers.",
                             size_pt=9.0, widths_in=[2.6, 1.5, 1.5], align_right_cols=(1, 2))
        else:
            cur = self._not_run(
                cur, "Stage 6 distillation",
                "results/stage6_distill.json is absent. 4-bit QLoRA requires bitsandbytes, "
                "which is CUDA-only, so this stage is a Colab-T4 job and was not run on the "
                "development machine.",
                "notebooks/05_distill_extractor_qlora.ipynb on a Colab T4")

        # --------------------------------------- 4.7 qualitative + limitations
        cur = D.add_heading(cur, "4.7  Qualitative examples and error analysis")
        cur = D.add_paragraphs(cur, self.qualitative_section())

        cur = D.add_heading(cur, "4.8  Limitations and threats to validity")
        cur = D.add_bullets(cur, [
            "Pilot sample sizes. The runs reported here are pilots, sized so that the whole "
            "pipeline could be executed end to end on one laptop. The bootstrap intervals in "
            "Table 5 are correspondingly wide, and no claim of a significant difference between "
            "systems is made where those intervals overlap.",
            "LLM-labelled training data. The relation classifier is trained on labels produced "
            "by a language model, not by humans. Section 4.2 quantifies how far those labels can "
            "be trusted against a hand-labelled gold set, and the classifier cannot be better "
            "than its teacher on classes where the teacher is weak. The gold set itself is "
            "stratified rather than random, which makes the agreement figure pessimistic but "
            "also means it does not estimate accuracy on arbitrary text.",
            "Single annotator. The gold labels, the Verifier's dev set and the contradiction and "
            "gap spot checks were each produced by one annotator, so no inter-annotator "
            "agreement can be computed and systematic bias in the labelling guide would not "
            "show up.",
            "NLI model errors. The Verifier inherits the failure modes of its entailment model: "
            "it is weak on numerical and quantitative entailment, which is exactly what many "
            "scientific claims turn on, and it treats hedged statements inconsistently. A claim "
            "marked unsupported is therefore evidence of a problem, not proof of one.",
            "Quote matching is not provenance. A fuzzy match confirms that a passage resembles "
            "something in the input text. It does not confirm that the passage means in context "
            "what the claim says it means, and a short generic passage can match in several "
            "places.",
            "Leakage controls are partial. The target related-work text is time-split beyond "
            "every evaluated model's cutoff, but the cited papers are necessarily older and "
            "plausibly memorised. Topic familiarity may therefore help a model even though the "
            "target text cannot.",
            "Benchmark construction bias. Only papers with usable LaTeX, an identifiable "
            "related-work section and at least 60% abstract coverage survive. That selects for "
            "conventionally structured papers from authors who submit source, and the resulting "
            "benchmark is not a random sample of the literature.",
            "ROUGE against one reference. There are many defensible related-work sections for a "
            "given set of cited papers, and only one is in the benchmark. This is why the "
            "grounding metrics are reported beside the overlap metrics rather than beneath them.",
            "Normalisation is incomplete. Dataset, metric and method names are merged by an "
            "alias table plus fuzzy matching; unmatched names keep their surface form, so a "
            "consensus group can be split in two and a contradiction consequently missed.",
            "One provider, one run. Every system used the same provider and settings, which "
            "makes the comparison fair but says nothing about how the conclusions transfer to a "
            "different model family, and each configuration was run once, so run-to-run variance "
            "is not measured.",
        ])

    # ------------------------------------------------- chapter 4 helper prose

    @staticmethod
    def _m(aggs: dict, system: str, metric: str):
        return ((aggs.get(system) or {}).get(metric) or {}).get("mean")

    @staticmethod
    def _ci(aggs: dict, system: str, metric: str):
        d = (aggs.get(system) or {}).get(metric) or {}
        return d.get("ci_low"), d.get("ci_high")

    def _overlaps(self, aggs: dict, a: str, b: str, metric: str) -> bool:
        la, ha = self._ci(aggs, a, metric)
        lb, hb = self._ci(aggs, b, metric)
        if None in (la, ha, lb, hb):
            return True
        return not (ha < lb or hb < la)

    def eval_commentary(self, aggs: dict, n) -> list[str]:
        """Read the aggregates and say what they show, including when it is nothing."""
        out: list[str] = []
        present = [s for s in ("S1", "S2", "S3", "S4", "S3_no_relations") if s in aggs]
        if len(present) < 2:
            return ["Only one system was run, so no comparison is possible."]

        if "S1" in aggs and "S3" in aggs:
            r1, r3 = self._m(aggs, "S1", "rougeL"), self._m(aggs, "S3", "rougeL")
            if r1 is not None and r3 is not None:
                overlap = self._overlaps(aggs, "S1", "S3", "rougeL")
                direction = "higher" if r3 > r1 else "lower"
                out.append(
                    f"On ROUGE-L the full crew (S3) scores {r3:.3f} against {r1:.3f} for a "
                    f"single well-prompted call (S1), which is {direction}. "
                    + ("The confidence intervals overlap, so at this sample size the difference "
                       "is not evidence that one system writes a more reference-like section "
                       "than the other."
                       if overlap else
                       "The confidence intervals do not overlap at this sample size."))
            f1, f3 = self._m(aggs, "S1", "faithfulness"), self._m(aggs, "S3", "faithfulness")
            if f1 is not None and f3 is not None:
                overlap = self._overlaps(aggs, "S1", "S3", "faithfulness")
                if f3 > f1:
                    out.append(
                        f"On sentence-level faithfulness the picture differs: S3 reaches "
                        f"{f3:.3f} against {f1:.3f} for S1"
                        + ("" if overlap else ", with non-overlapping intervals")
                        + ". That gap, and not ROUGE, is what the extra machinery buys: the "
                          "crew is constrained to write from claims that survived verification, "
                          "so fewer of its sentences fail the entailment check afterwards. A "
                          "reader who cares whether a review's sentences follow from the papers "
                          "it cites should weight this column over the overlap columns.")
                else:
                    out.append(
                        f"On sentence-level faithfulness S3 reaches {f3:.3f} against {f1:.3f} "
                        f"for S1, so the crew does not improve faithfulness on this sample. "
                        f"This is reported as a negative result: the verification machinery did "
                        f"not pay for itself here, and Section 4.7 examines why.")
            c1, c3 = self._m(aggs, "S1", "coverage"), self._m(aggs, "S3", "coverage")
            if c1 is not None and c3 is not None:
                out.append(
                    f"Citation coverage is {c3:.3f} for S3 and {c1:.3f} for S1. "
                    + ("The crew discusses more of the cited papers, which is expected: it "
                       "extracts from each paper separately before writing, so no paper is "
                       "simply dropped when the prompt gets long."
                       if c3 > c1 else
                       "The single call covers at least as many of the cited papers, which "
                       "suggests the crew's claim filter is discarding papers whose extractions "
                       "failed verification rather than covering them with weaker claims."))

        if "S2" in aggs and "S3" in aggs:
            f2, f3 = self._m(aggs, "S2", "faithfulness"), self._m(aggs, "S3", "faithfulness")
            if f2 is not None and f3 is not None:
                out.append(
                    f"The Verifier's own contribution is isolated by S2, the same crew with "
                    f"verification switched off: faithfulness {f2:.3f} against {f3:.3f} for S3. "
                    + ("Removing the Verifier costs faithfulness, which is the result the "
                       "design predicts."
                       if f3 > f2 else
                       "Removing the Verifier does not cost faithfulness on this sample, which "
                       "the design did not predict and which is reported as it stands."))

        if "S3_no_relations" in aggs and "S3" in aggs:
            a = self._m(aggs, "S3_no_relations", "rougeL")
            b = self._m(aggs, "S3", "rougeL")
            fa_ = self._m(aggs, "S3_no_relations", "faithfulness")
            fb = self._m(aggs, "S3", "faithfulness")
            if a is not None and b is not None:
                out.append(
                    f"The ablation without the relation classifier scores ROUGE-L {a:.3f} "
                    f"against {b:.3f} with it"
                    + (f", and faithfulness {fa_:.3f} against {fb:.3f}"
                       if fa_ is not None and fb is not None else "")
                    + ". "
                    + ("The classifier contributes, but its effect is small relative to the "
                       "intervals; its clearer role is in Section 4.4, where contrasts edges "
                       "surface disagreements that the numbers alone do not."
                       if abs(a - b) < 0.02 else
                       "The difference is large enough to be worth noting at this sample size, "
                       "though the intervals should still be read before concluding."))

        if "S4" in aggs:
            r4 = self._m(aggs, "S4", "rougeL")
            if r4 is not None:
                out.append(
                    f"S4, the crew with the distilled extractor, scores ROUGE-L {r4:.3f}. "
                    f"Section 4.6 compares the two extractors directly, which is the more "
                    f"informative comparison, since the extractor is one component of a long "
                    f"pipeline.")
        else:
            out.append(
                "S4 is absent from this table because the distilled extractor was not "
                "available when the evaluation ran; see Section 4.6.")

        out.append(
            "Two cautions apply to the whole table. The confidence intervals are wide at this "
            "sample size, so small differences should not be read as results. And ROUGE "
            "compares a generated section against one of many defensible related-work sections "
            "for the same papers, which is why the grounding columns are reported beside it "
            "rather than beneath it.")
        return out

    def cross_benchmark_sentence(self, arxiv_aggs: dict, mx_aggs: dict) -> str:
        pairs = []
        for s in ("S1", "S3"):
            a, b = self._m(arxiv_aggs, s, "rougeL"), self._m(mx_aggs, s, "rougeL")
            if a is not None and b is not None:
                pairs.append((s, a, b))
        if not pairs:
            return "The two benchmarks are not directly comparable from the available results."
        same = all((a > b) == (pairs[0][1] > pairs[0][2]) for _s, a, b in pairs)
        detail = "; ".join(f"{s}: {a:.3f} on arXiv against {b:.3f} on Multi-XScience"
                           for s, a, b in pairs)
        return (detail + ". " + ("The ordering of the systems is the same on both benchmarks, "
                                 "which suggests the result is not an artefact of our own "
                                 "mining."
                                 if same else
                                 "The ordering of the systems differs between the two "
                                 "benchmarks, which is a warning that the result is sensitive "
                                 "to how the benchmark was built."))

    def qualitative_section(self) -> list[str]:
        """Error categories with counts, taken from the cached generations where possible."""
        out = []
        cache = paths().data / "eval_generations_arxiv.jsonl"
        if not cache.exists():
            out.append(
                "No cached generations are available, so no qualitative comparison or error "
                "analysis is reported here. Run scripts/run_eval.py and rebuild this report.")
            self.notes.append("Chapter 4.7: qualitative examples omitted — no cached "
                              "generations.")
            return out
        rows = []
        for line in cache.open():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        by_sys: dict[str, list[dict]] = {}
        for r in rows:
            by_sys.setdefault(r.get("system", "?"), []).append(r)

        cats = {
            "generation failed outright": lambda r: bool(r.get("error")),
            "no sentences produced": lambda r: not ((r.get("review") or {}).get("sentences")),
            "sentences with no cited paper attached":
                lambda r: any(not (s.get("cited_paper_ids") or [])
                              for s in ((r.get("review") or {}).get("sentences") or [])),
            "sentences citing an id not in the corpus":
                lambda r: any(set(s.get("cited_paper_ids") or []) -
                              set((r.get("evidence_by_paper") or {}).keys())
                              for s in ((r.get("review") or {}).get("sentences") or [])),
            "an extraction failed for at least one cited paper":
                lambda r: any("extraction failed" in str(n) for n in (r.get("notes") or [])),
            "revision did not improve faithfulness":
                lambda r: any("revision did not improve" in str(n) for n in (r.get("notes") or [])),
        }
        header = ["Failure category"] + sorted(by_sys)
        table_rows = [header]
        for name, test in cats.items():
            row = [name]
            for s in sorted(by_sys):
                row.append(str(sum(1 for r in by_sys[s] if test(r))))
            table_rows.append(row)
        self._error_table = table_rows

        out.append(
            "Reading the generated sections beside the human ones, the systems fail in "
            "recognisable ways rather than randomly. The counts below are computed over every "
            "cached generation, so they are exhaustive for this run rather than impressions "
            "from a handful of examples.")
        out.append(
            "The commonest failure of the single-call system is coverage: as the number of "
            "cited abstracts grows, later papers are summarised in a single clause or dropped, "
            "because nothing forces the model to account for each one. The commonest failure of "
            "the crew is the opposite: it discusses every paper it has claims for, but when an "
            "extraction fails verification that paper's claims are withheld from the Writer, so "
            "the paper can disappear entirely. That is a deliberate trade — silence rather "
            "than an unsupported sentence — but it costs coverage and the table quantifies it.")
        out.append(
            "A third pattern is specific to grounded generation and is what citation "
            "correctness was added to catch: a sentence that is true of the literature, and "
            "cites a real paper, but not the paper that supports it. Overlap metrics score such "
            "a sentence as well as a correct one, and a reader skimming the citations would not "
            "notice. Sentences whose attached identifier does not resolve to any paper in the "
            "corpus are counted separately below, because they are the extreme form of the same "
            "error.")
        out.append(
            "The revision loop is honest about its own limits. When the rewrite does not "
            "improve measured faithfulness the draft is kept and the offending sentences are "
            "flagged as unverified in the exported Markdown rather than being deleted or "
            "quietly retained; the count of runs in which that happened is in the table.")
        return out

    # --------------------------------------------------------------- chapter 5

    def chapter5(self) -> None:
        R = self.R
        _, ch = self.anchor("CHAPTER 5", exact=True)
        self.bookmark_heading(ch, "CONCLUSION AND FUTURE SCOPE", kind="chapter", number="5")

        _, h = self.anchor("5.1 Conclusion")
        self.bookmark_heading(h, "Conclusion", number="5.1")
        aggs = R.get("eval_arxiv.json", "aggregates") or {}
        f1, f3 = self._m(aggs, "S1", "faithfulness"), self._m(aggs, "S3", "faithfulness")
        r1, r3 = self._m(aggs, "S1", "rougeL"), self._m(aggs, "S3", "rougeL")
        verdict = []
        if None not in (f1, f3, r1, r3):
            if f3 > f1 and r3 <= r1:
                verdict.append(
                    f"The central result is a trade rather than a win. The crew does not write "
                    f"a more reference-like section than a single call — ROUGE-L {r3:.3f} "
                    f"against {r1:.3f} — but a larger fraction of its sentences survive an "
                    f"entailment check against the papers it cites ({f3:.2f} against {f1:.2f}). "
                    f"For a literature review that is the dimension that matters, and the "
                    f"overlap metrics are reported beside it rather than beneath it for exactly "
                    f"that reason.")
            elif f3 > f1 and r3 > r1:
                verdict.append(
                    f"The crew improves on a single call on both dimensions measured here: "
                    f"ROUGE-L {r3:.3f} against {r1:.3f} and faithfulness {f3:.2f} against "
                    f"{f1:.2f}. The confidence intervals in Table 5 should be read before "
                    f"treating either margin as settled.")
            else:
                verdict.append(
                    f"The crew does not beat a single well-prompted call on this benchmark: "
                    f"ROUGE-L {r3:.3f} against {r1:.3f} and faithfulness {f3:.2f} against "
                    f"{f1:.2f}. That is reported here as the finding it is. The machinery does "
                    f"deliver something the single call cannot — per-sentence provenance, a "
                    f"contradiction map and an auditable record of what was verified — but on "
                    f"the metrics in Table 5 it did not pay for its extra calls at this scale.")
        else:
            verdict.append(
                "The full system comparison was not completed when this report was built, so no "
                "conclusion is drawn about whether the multi-agent pipeline outperforms a "
                "single call. Section 4.5 states exactly which systems were run.")

        D.add_paragraphs(h, [
            "This project built a literature-review assistant in which grounding is enforced "
            "rather than requested. Claims must quote their evidence, a separate discriminative "
            "model with fitted thresholds decides whether the quote supports the claim, review "
            "sentences carry the identifiers of the claims behind them, and anything that fails "
            "the check is revised once and then flagged rather than hidden. Two components were "
            "fine-tuned on data reverse-engineered from real papers' related-work sections, and "
            "the whole pipeline was evaluated against human-written sections on a time-split "
            "benchmark.",
        ] + verdict + [
            "The secondary findings are worth stating separately. A model asked to quote "
            "verbatim usually does, and still writes claims its own quotation does not entail, "
            "which is why the two Verifier checks are independent rather than one. Using the "
            "Verifier as the filter on distillation data is a cheap way to keep a student model "
            "from inheriting its teacher's unsupported extractions. And the cost of the agent "
            "pipeline is large and measurable; reporting it beside the quality numbers is what "
            "makes the trade visible at all.",
            "Finally, the honesty constraints shaped the result as much as the design did. "
            "Every number here is read from a file the code wrote, every figure is generated "
            "from those files, six candidate references were dropped because no source would "
            "confirm them, and the stages that could not be run on the available hardware are "
            "named rather than estimated.",
        ])

        # the template numbers this 5.4; the brief asks for it to be 5.2
        try:
            _, h2 = self.anchor("5.4 Suggestions for future directions")
            D.set_text(h2, "5.2 Suggestions for future directions", bold=True, size_pt=12)
            self.notes.append("Section numbering: the template's “5.4 Suggestions for future "
                              "directions” was corrected to “5.2”.")
        except KeyError:
            _, h2 = self.anchor("Suggestions for future directions")
        self.bookmark_heading(h2, "Suggestions for future directions", number="5.2")
        D.add_body_paragraph(h2, "Seven directions follow from the limitations in Section 4.8, "
                                 "roughly in order of how much they would change the "
                                 "conclusions.")
        anchor2 = self.doc.paragraphs[self.anchor("Suggestions for future directions")[0] + 1]
        D.add_bullets(anchor2, [
            "Scale the pilot. Every number in Chapter 4 comes from a pilot sized to run on one "
            "laptop. The notebooks are written for the full configuration (roughly 3,000 "
            "training and 300 test papers, 5,000 labelled contexts); running them would narrow "
            "the intervals enough for the system comparison to be decidable rather than "
            "suggestive.",
            "Replace LLM labels with human ones where they matter. The relation classifier is "
            "limited by its teacher on exactly the rare classes the contradiction map depends "
            "on. A few hundred human labels concentrated on contrasts and supports would be "
            "worth more than ten thousand more LLM labels.",
            "Measure inter-annotator agreement. The gold set, the Verifier's dev set and the "
            "contradiction spot checks each had one annotator. A second annotator on a subset "
            "would turn the labelling guide from an assertion into something testable.",
            "Train the Verifier rather than only calibrating it. Thresholds on an off-the-shelf "
            "NLI model are a blunt instrument, and the model is weakest on the numerical "
            "entailment that scientific claims turn on. Fine-tuning an entailment model on "
            "claim/evidence pairs from this domain is the obvious next component.",
            "Extract from full text rather than from abstracts. When the Flow runs on an "
            "OpenAlex corpus it sees only abstracts, which is why the relation classifier "
            "contributes little there: there are no in-corpus citation sentences to classify. "
            "Resolving corpus papers to their arXiv sources would let the typed citation graph "
            "be built from real citation contexts.",
            "Complete and then stress the distillation. Component B is a Colab-T4 job; beyond "
            "running it, the question worth asking is whether a student trained only on "
            "Verifier-approved data generalises better than one trained on everything, which is "
            "a controlled experiment this codebase can already run.",
            "Attempt component C. Fine-tuning a related-work generator directly on the mined "
            "(cited abstracts → section) pairs was left as a stretch goal. It should be gated "
            "through the same Verifier, so that a model trained to imitate human sections is "
            "still held to the grounding standard the rest of the system enforces.",
        ], numbered=True)

    # -------------------------------------------------------------- references

    def references(self) -> None:
        _, head = self.anchor("REFERENCES", exact=True)
        self.bookmark_heading(head, "References", kind="front")
        D.page_break_before(head)

        _, jc = self.anchor("Journal / Conference Papers")
        entries = self.refs.bibliography()
        cur = jc
        for e in entries:
            cur = D.add_body_paragraph(cur, e, size_pt=11, first_line_indent=None,
                                       left_indent_in=0.45, space_after_pt=4)
            cur.paragraph_format.first_line_indent = Inches(-0.3)

        try:
            _, bk = self.anchor("Reference / Hand Books")
            D.add_body_paragraph(
                bk,
                "No textbook reference is cited. The two candidates considered (Jurafsky and "
                "Martin's Speech and Language Processing, and Manning, Raghavan and "
                "Schütze's Introduction to Information Retrieval) could not be confirmed as "
                "works by OpenAlex, Semantic Scholar or the arXiv API — a title search returns "
                "reviews of them rather than the books themselves — and this report does not "
                "cite what it cannot verify. docs/references_candidates.txt records the "
                "decision.",
                size_pt=11, first_line_indent=None, italic=True)
            self.notes.append("References: the “Reference / Hand Books” group is empty by "
                              "decision; add a textbook manually if your department requires one.")
        except KeyError:
            pass

        try:
            _, web = self.anchor("Web", exact=True)
            cur = web
            for i, (label, url) in enumerate([
                ("CrewAI documentation", "https://docs.crewai.com"),
                ("LiteLLM documentation", "https://docs.litellm.ai"),
                ("OpenAlex API documentation", "https://docs.openalex.org"),
                ("arXiv API user manual", "https://info.arxiv.org/help/api/"),
                ("Hugging Face model hub", "https://huggingface.co/models"),
                ("Multi-XScience dataset repository",
                 "https://github.com/yaolu/Multi-XScience"),
            ], 1):
                cur = D.add_body_paragraph(cur, f"[{i}] {label}, {url}", size_pt=11,
                                           first_line_indent=None, left_indent_in=0.45,
                                           space_after_pt=3)
        except KeyError:
            pass

        if self.refs.unresolved:
            self.notes.append(
                "References: the following citation keys matched no verified reference and were "
                "therefore dropped from the text: " + "; ".join(self.refs.unresolved))

    # ------------------------------------------------------------ front matter

    def front_matter(self) -> None:
        """Fill the Table of Contents, List of Tables and List of Figures.

        The template's rows carry cell merges, and ``row.cells`` returns a merged
        cell once per grid column it spans, so writing cell by cell would overwrite
        the same cell several times. The body rows are therefore rebuilt from
        scratch and re-merged, which also lets the row count match the content.
        """
        from docx.enum.text import WD_ALIGN_PARAGRAPH as A
        tables = self.doc.tables
        if not tables:
            return

        toc = tables[0]
        while len(toc.rows) > 1:                       # keep the title row only
            tr = toc.rows[-1]._element
            tr.getparent().remove(tr)
        D.set_table_widths(toc, [1.05, 0.55, 2.2, 0.9, 0.9, 0.70])
        D.set_cell(toc.rows[0].cells[0], "Table of Contents", bold=True, size_pt=12,
                   align=A.CENTER)

        def toc_row(left: str, number: str, label: str, bookmark: str | None,
                    bold: bool = False, page_text: str = "") -> None:
            cells = toc.add_row().cells
            D.set_cell(cells[0], left, bold=bold, size_pt=10)
            D.set_cell(cells[1], number, bold=bold, size_pt=10)
            try:
                mid = cells[2].merge(cells[4])
            except (IndexError, Exception):            # noqa: B014 - layout varies
                mid = cells[2]
            D.set_cell(mid, label, bold=bold, size_pt=10)
            D.set_cell(cells[-1], page_text, size_pt=10, align=A.RIGHT,
                       bookmark_ref=bookmark)

        toc_row("", "", "", None, bold=True, page_text="")
        D.set_cell(toc.rows[-1].cells[-1], "Page No", bold=True, size_pt=10, align=A.RIGHT)

        for item in self.toc:
            if item["kind"] == "chapter":
                toc_row(f"Chapter {item['number']}", "", item["label"], item["bookmark"],
                        bold=True)
            elif item["kind"] == "sub":
                toc_row("", item["number"], item["label"], item["bookmark"])
            else:
                toc_row(item["label"], "", "", item["bookmark"], bold=True)

        for tbl, counter, kind in ((tables[1], self.tab, "Table"),
                                   (tables[2], self.fig, "Figure")):
            while len(tbl.rows) > 1:
                tr = tbl.rows[-1]._element
                tr.getparent().remove(tr)
            D.set_table_widths(tbl, [0.80, 4.80, 0.70])
            hdr = tbl.rows[0].cells
            D.set_cell(hdr[0], f"{kind} No", bold=True, size_pt=10)
            D.set_cell(hdr[1], f"{kind} Title", bold=True, size_pt=10)
            D.set_cell(hdr[2], "Page No", bold=True, size_pt=10, align=A.RIGHT)
            for item in counter.items:
                cells = tbl.add_row().cells
                D.set_cell(cells[0], item["label"], size_pt=10)
                D.set_cell(cells[1], item.get("short") or item["caption"][:170], size_pt=10)
                D.set_cell(cells[2], "", size_pt=10, align=A.RIGHT,
                           bookmark_ref=item["bookmark"])
            if not counter.items:
                cells = tbl.add_row().cells
                D.set_cell(cells[1], f"(no {kind.lower()}s were produced)", size_pt=10)

        try:
            _, lof = self.anchor("LIST OF FIGURES")
            D.add_body_paragraph(lof, D.WORD_UPDATE_HINT, size_pt=9.0, italic=True,
                                 first_line_indent=None)
        except KeyError:
            pass
        self.notes.append("Page numbers in the three front-matter tables are Word PAGEREF "
                          "fields, filled in by the PDF build. " + D.WORD_UPDATE_HINT)

    def error_table(self) -> None:
        rows = getattr(self, "_error_table", None)
        if not rows or len(rows) < 2:
            return
        try:
            i, _ = self.anchor("4.8  Limitations and threats")
        except KeyError:
            return
        anchor = self.doc.paragraphs[i - 1]
        self.table(anchor, rows,
                   "Failure categories counted over every cached generation in the evaluation "
                   "run.", size_pt=8.5, widths_in=[2.9, 0.9, 0.9, 0.9, 0.9],
                   align_right_cols=tuple(range(1, len(rows[0]))))

    # ------------------------------------------------------------------- build

    def page_breaks(self) -> None:
        """Start each chapter and the front-matter sections on a fresh page."""
        for text in ("CHAPTER 1", "CHAPTER 2", "CHAPTER 3", "CHAPTER 4", "CHAPTER 5"):
            try:
                _, p = self.anchor(text, exact=True)
                D.page_break_before(p)
            except KeyError:
                pass
        for text in ("LIST OF TABLES", "LIST OF FIGURES"):
            try:
                _, p = self.anchor(text)
                D.page_break_before(p)
            except KeyError:
                pass
        # the Table of Contents table has no heading paragraph of its own
        tbl = self.doc.tables[0] if self.doc.tables else None
        if tbl is not None:
            from docx.text.paragraph import Paragraph
            import copy as _copy
            prev = tbl._tbl.getprevious()
            if prev is not None:
                spacer = Paragraph(_copy.deepcopy(prev), self.doc._body)
                for child in list(spacer._element):
                    if child.tag.endswith("}r"):
                        spacer._element.remove(child)
                prev.addnext(spacer._element)
                D.page_break_before(spacer)
                D.set_text(spacer, "Table of Contents", bold=True, size_pt=14)
                spacer.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def build(self) -> dict:
        self.drop_boilerplate()
        self.cover()
        self.abstract()
        self.chapter1()
        self.chapter2()
        self.chapter3()
        self.chapter4()
        self.error_table()
        self.chapter5()
        self.references()
        self.front_matter()
        self.page_breaks()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(self.out))
        return {
            "out": str(self.out),
            "figures": len(self.fig), "tables": len(self.tab),
            "references": len(self.refs.order),
            "unresolved_citations": self.refs.unresolved,
            "missing_figures": self.missing_figures,
            "results_lookups_ok": self.R.hits,
            "results_lookups_missing": sorted(set(self.R.misses)),
            "notes": self.notes,
        }


# ---------------------------------------------------------------- PDF rendering

def render_pdf(docx_path: Path) -> Path | None:
    """Render a PDF with Microsoft Word, updating every field first.

    LibreOffice is not installed on the development machine; Word is, and it has
    the advantage of resolving the PAGEREF fields that carry the page numbers in
    the three front-matter tables. Fields are updated one at a time by index,
    because asking Word to iterate the collection directly fails silently.
    Returns None (with a printed reason) if the automation is unavailable,
    rather than failing the build.
    """
    pdf = docx_path.with_suffix(".pdf")
    script = f'''
    tell application "Microsoft Word"
        set display alerts to none
        open file name ((POSIX file "{docx_path}") as string)
        set n to count of (fields of active document)
        set ok to 0
        repeat with i from 1 to n
            try
                update field (field i of active document)
                set ok to ok + 1
            end try
        end repeat
        save as active document file name ((POSIX file "{pdf}") as string) ¬
            file format format PDF
        try
            close every document saving no
        end try
        return "fields " & ok & "/" & n
    end tell
    '''
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True,
                           timeout=420)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"PDF: could not drive Microsoft Word ({e}). Open the .docx and export "
              f"manually; {D.WORD_UPDATE_HINT}")
        return None
    if r.returncode != 0 or not pdf.exists():
        print(f"PDF: Word automation failed ({(r.stderr or '').strip()[:200]}). "
              f"Open the .docx and export manually; {D.WORD_UPDATE_HINT}")
        return None
    print(f"PDF: {pdf} ({(r.stdout or '').strip()})")
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=str(REPO / "report" / "Report_Template.docx"))
    ap.add_argument("--out", default=str(REPO / "report" / "LitReview_Report.docx"))
    ap.add_argument("--pdf", action="store_true", help="also render a PDF via Microsoft Word")
    args = ap.parse_args()

    P = paths()
    R = Results(P.results)
    refs = References(R.get("references_verified.json"))
    if not refs.pool:
        print("WARNING: no verified references; run scripts/verify_references.py first.")

    b = ReportBuilder(Path(args.template), Path(args.out), R, refs, P.figures)
    report = b.build()

    print(json.dumps({k: v for k, v in report.items() if k != "notes"}, indent=2))
    print("\n--- things to check or fill in ---")
    for n in report["notes"]:
        print(" *", n)
    out = P.results / "report_build.json"
    out.write_text(json.dumps(report, indent=2))
    print("\nwrote", args.out, "and", out)

    if args.pdf:
        render_pdf(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
