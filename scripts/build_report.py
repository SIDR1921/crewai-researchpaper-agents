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


def _clip(text: str | None, limit: int) -> str:
    """Trim a quote at a word boundary so it never ends mid-word."""
    t = " ".join(str(text or "").split())
    if not t:
        return "none"
    if len(t) <= limit:
        return t
    return t[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "\u2026"


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
        self.unresolved_refs: list[str] = []

    # ---------------------------------------------------------------- plumbing

    def anchor(self, text: str, start: int = 0, exact: bool = False):
        return D.find_paragraph(self.doc, text, start=start, exact=exact)

    def bookmark_heading(self, par, label: str, kind: str = "sub", number: str = "") -> str:
        name = "_toc_" + re.sub(r"[^a-z0-9]+", "_", f"{number} {label}".strip().lower())[:38]
        D.add_bookmark(par, name)
        self.toc.append({"kind": kind, "number": number, "label": label, "bookmark": name})
        return name

    def figure(self, anchor_par, filename: str, caption: str, width_in: float = 5.1):
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

    def table(self, anchor_par, rows, caption: str, key: str | None = None, **kw):
        """Insert a table with a numbered caption above it, as the template's lists expect."""
        item = self.tab.next(caption, source=key or caption[:40])
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
        n_lab = R.get("stage2_labelling.json", "n_labelled_total")
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
            "A large language model can write a literature review that reads well and still "
            "describe papers it has not understood. The sentences look right, the citations "
            "are real, and only someone who reads the cited papers can tell that a claim does "
            "not belong to the paper it is attached to. This project builds a system that "
            "makes that failure visible. It is a team of six language-model agents, built with "
            "the CrewAI framework, that reads a set of papers and writes a related-work "
            "section in which every sentence can be traced back to a quoted line from a "
            "specific paper. A separate checking model decides whether each sentence really "
            "follows from the lines behind it, and anything that fails is rewritten once and "
            "then marked rather than hidden."
        ]

        p2 = ("The training data comes from the papers themselves. The related-work section of "
              "any published paper is a short literature review that an expert has already "
              "written, and the papers it cites are known. Hiding that section and keeping only "
              "the cited papers' abstracts gives a task with a correct answer, at no annotation "
              "cost. ")
        if listed and kept:
            p2 += (f"From {listed:,} arXiv papers in computer-science language and information "
                   f"retrieval, the pipeline produced {n_ctx:,} citation sentences and {kept} "
                   f"usable test examples. ")
        else:
            p2 += "The mining statistics were not available when this report was built. "
        if n_lab:
            p2 += (f"{n_lab:,} of those citation sentences were labelled with one of seven "
                   f"relation types, such as whether a paper is being used, compared against "
                   f"or disagreed with. ")
        if kappa is not None and sci_f1 is not None:
            p2 += (f"A classifier fine-tuned from SciBERT scores {sci_f1:.2f} macro-F1 on a "
                   f"hand-checked set"
                   + (f", against {tfidf_f1:.2f} for a simple word-counting baseline"
                      if tfidf_f1 is not None else "")
                   + f", and the machine labels it learns from agree with the hand-checked "
                     f"ones at a kappa of {kappa:.2f}.")
        parts.append(p2)

        p3 = ""
        if sup is not None:
            p3 += (f"The checker finds {sup:.0%} of extracted claims fully supported by the "
                   f"line they quote. The gap between quoting correctly and being supported is "
                   f"the problem this project is about: a model usually copies the quote "
                   f"accurately and still writes a claim the quote does not prove. ")
        if s1_r is not None and s3_r is not None and n_eval:
            direction = "better than" if s3_r > s1_r else "below"
            p3 += (f"On {n_eval} test papers published after every tested model's training "
                   f"cut-off, the full team scores ROUGE-L {s3_r:.3f}, {direction} a single "
                   f"well-written prompt at {s1_r:.3f}")
            if s1_f is not None and s3_f is not None:
                p3 += (f", while the share of sentences that survive checking moves from "
                       f"{s1_f:.2f} to {s3_f:.2f}")
            p3 += ". "
        else:
            p3 += ("The full comparison between systems had not been run when this report was "
                   "built, and Chapter 4 says exactly which parts were and were not measured. ")
        p3 += ("Every number in this report is read from a file the code wrote, and every "
               "figure is drawn from those files by a single script.")
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
            "This project builds a tool that writes the \u201crelated work\u201d part of a research "
            "report, and checks its own writing before handing it over. You give it a topic "
            "or a few starting papers. It finds related papers, reads each one, pulls out the "
            "main claims, and keeps a short quoted line from the paper next to every claim as "
            "proof. It then compares the papers against each other to see where they agree, "
            "where they disagree, and which combinations nobody has tried. Finally it writes "
            "the section, and every sentence it writes carries a list of the claims it was "
            "built from.",
            "The work is divided between six agents, each with a narrow job: a Librarian that "
            "finds papers, an Extractor that pulls out claims, a Verifier that checks them, a "
            "Synthesist that compares papers, a Gap Analyst that looks for missing work, and a "
            "Writer. The limits matter as much as the jobs. The Extractor cannot search for "
            "anything, so it can only use the text it is given. The Writer cannot check its "
            "own output, because a model asked to grade its own work tends to agree with "
            "itself. Only the Verifier can run the checking model.",
            "Two parts of the system are trained rather than prompted. The first is a "
            "classifier that reads a sentence containing a citation and decides what the "
            "sentence is doing: using the cited work, comparing against it, or disagreeing "
            "with it. The second is a small model trained to copy the extraction step, so the "
            "same work can be done on a laptop instead of through a paid interface.",
            "The same idea supplies the test set. Every published paper already contains a "
            "short literature review written by an expert \u2014 its related-work section \u2014 "
            "and the papers it cites are listed. Hiding that section and giving a system only "
            "the cited papers' abstracts creates a test with a correct answer that nobody had "
            "to write by hand.",
        ])

        _, h = self.anchor("1.2 Motivation and background")
        self.bookmark_heading(h, "Motivation and background of the study", number="1.2")
        D.add_paragraphs(h, [
            f"Language models write fluent summaries that are often wrong in ways fluency "
            f"hides. Maynez and colleagues {C('On Faithfulness and Factuality')} found that a "
            f"large share of summaries from strong models contain statements the source "
            f"document never made, and that the usual scoring method, ROUGE, barely "
            f"distinguishes those summaries from correct ones. A later survey "
            f"{C('Survey of Hallucination')} found the same pattern across many text-generation "
            f"tasks.",
            "For a literature review this is a serious problem rather than an annoying one. A "
            "made-up reference is easy to catch, because the paper does not exist. A real "
            "reference attached to a claim that paper never made is much harder to catch, "
            "because every part of the sentence checks out except the link between them. A "
            "reader would have to open the cited paper to notice, which is exactly the work "
            "the tool was supposed to save.",
            f"Researchers in summarisation responded by checking whether a summary follows "
            f"from its source, instead of comparing it against a reference answer. One line of "
            f"work generates questions from the summary and checks that the source gives the "
            f"same answers {C('Asking and Answering Questions')}. Another trains a classifier "
            f"on deliberately corrupted summaries {C('Evaluating the Factual Consistency')}. A "
            f"third uses natural language inference, the task of deciding whether one sentence "
            f"follows from another, applied one sentence at a time {C('SummaC')}. This project "
            f"follows the third approach and adds one requirement: the system must say which "
            f"line each sentence came from, so there is something definite to check against.",
            f"Meanwhile it has become easy to split a task across several model-driven agents "
            f"with their own roles and tools {C('CAMEL', 'Generative Agents')}, and to have a "
            f"model criticise and revise its own output {C('Self-Refine', 'Reflexion')}. What "
            f"is usually missing is an honest answer to whether all that extra machinery is "
            f"worth its cost. This project measures that directly, with the same model and "
            f"settings for every system compared, and reports the answer whichever way it "
            f"comes out.",
        ])

        _, h = self.anchor("1.3 Problem statement")
        self.bookmark_heading(h, "Problem statement and relevance", number="1.3")
        D.add_paragraphs(h, [
            "The problem is this: given a research topic and the papers that matter to it, "
            "produce a literature review in which every sentence can be traced to a specific "
            "line in a specific paper, together with a map of where those papers agree and "
            "disagree and a list of gaps \u2014 and do it within the limits of a free API account "
            "and one free cloud GPU.",
            "Three smaller problems follow. First, there is no ready-made dataset that says "
            "what a citing sentence is claiming about the paper it cites, so that signal has "
            "to be mined from published papers. Second, the checking cannot be done by asking "
            "the same model again, because it will usually agree with itself; it needs a "
            "separate model doing a different task. Third, the result has to be compared "
            "against something a human wrote, or the claim that the output is well-grounded "
            "cannot be tested at all.",
            "The practical case is simple. Any active research area now produces more papers "
            "than a student can read before writing about them, and the tools that offer to "
            "help are the same tools that invent things. A slower system whose every sentence "
            "can be checked against a quoted line is more useful than a fast one whose output "
            "has to be re-read against the sources anyway.",
        ])

        _, h = self.anchor("1.4 Objectives")
        self.bookmark_heading(h, "Objectives", number="1.4")
        D.add_body_paragraph(h, "Each objective below names the measurement that decides "
                                "whether it was met. Chapter 4 reports those measurements.")
        last = self.doc.paragraphs[self.anchor("1.4 Objectives")[0] + 1]
        D.add_bullets(last, [
            "Mine related-work sections and citation sentences from arXiv source files, and "
            "report how many papers survive each step as a percentage of those listed.",
            "Define seven citation relation types with a written labelling guide, label "
            "sentences with a language model, and measure how far those labels can be trusted "
            "by comparing them against a set labelled independently by hand.",
            "Fine-tune SciBERT on those labels and beat three baselines \u2014 always guessing "
            "the most common class, a word-counting classifier, and asking a language model "
            "with no training \u2014 on macro-F1.",
            "Extract claims that quote their evidence word for word, and check each claim "
            "twice: that the quote really appears in the source, and that the claim follows "
            "from the quote. Set the thresholds for both checks by testing them on a "
            "hand-checked sample rather than guessing.",
            "Build a map of agreement and disagreement between papers and a table of which "
            "method families have been tried on which datasets, and have agents argue both "
            "for and against every disagreement and every gap they report.",
            "Train a small local model to copy the extraction step, using only extractions "
            "that passed the checker, and compare the two on output validity, claim support, "
            "speed and cost.",
            "Compare four systems and one cut-down version on papers published after every "
            "tested model's training cut-off, reporting overlap with the human text, how many "
            "cited papers are discussed, how often a sentence is attached to the right paper, "
            "how many sentences survive checking, and what each system cost to run.",
        ], numbered=True)

    # --------------------------------------------------------------- chapter 2

    def chapter2(self) -> None:
        C = self.refs.cite
        _, ch = self.anchor("CHAPTER 2", exact=True)
        self.bookmark_heading(ch, "BACKGROUND THEORY / LITERATURE REVIEW",
                              kind="chapter", number="2")
        _, title = self.anchor("BACKGROUND THEORY / LITERATURE REVIEW")

        cur = D.add_paragraphs(title, [
            "{{TAB:approach-comparison}} at the end of this chapter summarises how the "
            "approaches below compare. This chapter covers the five areas this project draws on: summarising several "
            "documents at once, working out what a citation is doing, checking whether "
            "generated text is actually supported, teams of language-model agents, and tools "
            "that help write literature reviews. Every paper named here was checked against "
            "OpenAlex, Semantic Scholar or the arXiv listing before it was cited; the record "
            "of those checks is in results/references_verified.json.",
        ])

        cur = D.add_heading(cur, "Summarising scientific papers")
        cur = D.add_paragraphs(cur, [
            f"Summarising a single document improved in two steps. First came models that "
            f"could copy rare words straight out of the source instead of trying to spell them "
            f"{C('Get To The Point')}. Then came large models pre-trained on a general "
            f"text-repair task and adapted to summarising, of which BART {C('BART')} and "
            f"PEGASUS {C('PEGASUS')} are still the usual points of comparison.",
            f"Scientific papers broke an assumption these models relied on, which is that a "
            f"document fits inside the model at once. Cohan and colleagues "
            f"{C('A Discourse-Aware Attention Model')} handled longer documents by paying "
            f"attention to section structure, and released the arXiv and PubMed datasets that "
            f"made long-document summarising a standard task. The XSum dataset "
            f"{C('Just the Summary')} pushed the other way, asking for a single sentence that "
            f"cannot be produced by copying.",
            f"Writing a related-work section is the specific task this project is tested on, "
            f"and it is older than today's models. Abu-Ra'ed and colleagues "
            f"{C('Automatic related work section generation')} treated it as summarising the "
            f"cited papers. Wang and colleagues {C('Neural Related Work Summarization')} let "
            f"the model attend to both the citing paper's own content and the cited abstracts. "
            f"Luu and colleagues {C('Explaining Relationships Between Scientific Documents')} "
            f"reframed it as explaining the relationship between two papers rather than "
            f"summarising one, which is the closest existing idea to the relation classifier "
            f"used here.",
            f"Multi-XScience {C('Multi-XScience')} turned the task into a standard benchmark: "
            f"each example pairs a paper's related-work paragraph with the abstracts it cites, "
            f"which is exactly the setup described in Chapter 3. This project uses it as a "
            f"second test set so results can be compared against a published dataset and not "
            f"only against our own mining. The large-scale parsed full text both rely on comes "
            f"from S2ORC {C('S2ORC')}. More recently Zhang and colleagues "
            f"{C('Benchmarking Large Language Models for News Summarization')} found that "
            f"instruction-tuned models match or beat specially trained summarisers in human "
            f"judgement while the automatic scores disagree \u2014 a direct warning against "
            f"reading ROUGE on its own, and one the evaluation here takes seriously.",
        ])

        cur = D.add_heading(cur, "Working out what a citation is doing")
        cur = D.add_paragraphs(cur, [
            f"Citations are not all the same, and the differences are what a review has to "
            f"express. Early work looked at whether a citation was positive or negative and at "
            f"how much surrounding text was needed to tell "
            f"{C('Context-Enhanced Citation Sentiment Detection')}, "
            f"{C('Mining the Context of Citations')}. Two datasets then made the task "
            f"standard. ACL-ARC, used by Jurgens and colleagues "
            f"{C('Measuring the Evolution of a Scientific Field')} to trace how a research "
            f"field changes over time, has six categories. SciCite "
            f"{C('Structural Scaffolds')} traded detail for size with three \u2014 background, "
            f"method and result \u2014 and showed that training on related side-tasks helps. "
            f"SciBERT {C('SciBERT')}, a version of BERT trained on scientific text, is the "
            f"usual starting point for both, and is the starting point here.",
            "The seven categories used in this project are finer than SciCite's and cut "
            "differently from ACL-ARC's, for one reason that matters later. The system needs "
            "to tell a neutral comparison apart from an actual disagreement. Running someone "
            "else's method as a baseline and reporting that you scored higher is a "
            "measurement. Saying their conclusion does not hold is a disagreement, and only "
            "the second is evidence that two papers conflict. SciCite's \u201cresult\u201d class "
            "and ACL-ARC's \u201ccompare or contrast\u201d class both merge the two. Section 3.1 "
            "sets out how those datasets were mapped onto our categories and what that mapping "
            "loses; because it loses something, mapped rows are used only for extra training "
            "and never for scoring.",
        ])

        cur = D.add_heading(cur, "Checking whether generated text is supported")
        cur = D.add_paragraphs(cur, [
            f"The finding behind this project is that fluent summaries are often unsupported "
            f"and that overlap scores do not notice "
            f"{C('On Faithfulness and Factuality')}. Three families of automatic check "
            f"followed. FactCC {C('Evaluating the Factual Consistency')} trains a classifier "
            f"on summaries that were deliberately broken in known ways. QAGS "
            f"{C('Asking and Answering Questions')} asks questions about the summary and "
            f"checks whether the source gives the same answers. SummaC {C('SummaC')} revisits "
            f"natural language inference \u2014 deciding whether one sentence follows from "
            f"another \u2014 and shows it works well once applied sentence by sentence rather "
            f"than to whole documents. That last design is the one the checker here copies.",
            f"Working out whether these checks are any good needed human judgement. SummEval "
            f"{C('SummEval')} re-annotated model outputs and showed how weakly the standard "
            f"scores track what people actually think. Goyal and Durrett "
            f"{C('Annotating and Modeling Fine-grained Factuality')} annotated errors in "
            f"detail and found that different kinds of model fail in different ways. "
            f"BERTScore {C('BERTScore')} is included here as a meaning-based alternative to "
            f"ROUGE, not as a support check, because it compares generated text against a "
            f"reference answer rather than against the source.",
            f"The fact-checking field supplies the other half. FEVER {C('FEVER')} established "
            f"the pattern of retrieving evidence and then testing whether it supports a claim. "
            f"SciFact {C('Fact or Fiction')} moved that into science, where claims are hedged "
            f"and evidence is technical. DeBERTa {C('DeBERTa')} is the model design whose "
            f"trained versions are used here as the checker. The hallucination survey "
            f"{C('Survey of Hallucination')} supplies the wording used throughout this report: "
            f"a claim is called unsupported when the evidence does not establish it, which is "
            f"a weaker and more checkable statement than calling it false.",
        ])

        cur = D.add_heading(cur, "Teams of language-model agents")
        cur = D.add_paragraphs(cur, [
            f"Asking a model to work through a problem in steps was the first sign that "
            f"breaking a task up helps {C('Chain-Of-Thought Prompting')}, and giving a model "
            f"tools let it act outside its own text {C('Toolformer')}. Several agents working "
            f"together followed: CAMEL {C('CAMEL')} showed that agents given roles and allowed "
            f"to talk can carry a task further than one prompt, and generative agents "
            f"{C('Generative Agents')} showed memory and planning over long runs. Having a "
            f"model criticise and fix its own work was formalised as Self-Refine "
            f"{C('Self-Refine')} and, with a separate critic, as Reflexion {C('Reflexion')}.",
            "CrewAI, the framework used here, belongs to this family. It provides agents with "
            "a role, a goal and a list of tools; tasks whose output can be forced into a fixed "
            "shape; and flows that connect steps with explicit order and shared state. Two of "
            "its features carry real weight in this project. Forcing a task's output into a "
            "fixed data shape is what lets the Extractor be checked by code rather than read "
            "by eye. Named flow steps are what let the human approval point be a real stop, "
            "and let a run that is interrupted carry on where it left off \u2014 which matters "
            "when everything has to fit inside a free account's limits.",
            "The important lesson from this literature, for this project, is that a model "
            "checking its own work is using the same model that produced it, and its judgement "
            "is tied to the error that caused the problem. The checker here is therefore not "
            "an agent asked to be sceptical. It is a separate, smaller model doing a different "
            "job on a well-defined pair of texts, and the agent whose role is called Verifier "
            "is allowed to do nothing except call it.",
        ])

        cur = D.add_heading(cur, "Tools that help write literature reviews")
        cur = D.add_paragraphs(cur, [
            f"Retrieval-augmented generation {C('Retrieval-Augmented Generation')}, with "
            f"meaning-based search {C('Dense Passage Retrieval')}, is the standard way to tie "
            f"a model's output to documents, and it is what the practical tools are built on. "
            f"PaperQA {C('PaperQA')} answers scientific questions by forcing the answer "
            f"through retrieved passages, and reports less invention as a result. STORM "
            f"{C('Assisting in Writing Wikipedia-like Articles')} tackles longer writing: it "
            f"researches a topic by simulating questions from several points of view, builds "
            f"an outline, and writes a long article with citations. Su\u0161njak and colleagues "
            f"{C('Automating Research Synthesis')} fine-tune subject-specific models for "
            f"research synthesis and report the practical trade-offs.",
            f"Two things separate the system built here. The first is how finely it is "
            f"accountable. STORM and PaperQA attach citations at the level of a passage or a "
            f"paragraph; this system records, for every sentence, the exact claim identifiers "
            f"it used, then checks that sentence against those claims' quoted lines and no "
            f"others. The second is that two parts are trained rather than prompted: the "
            f"citation-relation classifier, trained on sentences mined from real papers, and "
            f"the extractor, compressed into a small open model {C('Qwen2.5 Technical Report')} "
            f"using QLoRA {C('QLoRA')} with the checker deciding which examples it may learn "
            f"from.",
        ])

        # 2.1 comparative table
        _, h = self.anchor("2.1 ")
        self.bookmark_heading(h, "Summary of existing studies / Comparative analysis of "
                                 "various approaches", number="2.1")
        rows = [
            ["Approach", "Example work", "What a statement traces back to",
             "What checks it", "Citation types", "Trained part"],
            ["Pre-trained summariser", f"BART {C('BART')}, PEGASUS {C('PEGASUS')}",
             "nothing specific", "nothing", "no", "whole model"],
            ["Related-work generator",
             f"Wang et al. {C('Neural Related Work Summarization')}, Multi-XScience "
             f"{C('Multi-XScience')}",
             "a cited abstract", "nothing", "no", "a task-specific model"],
            ["Citation-type classifier",
             f"SciCite {C('Structural Scaffolds')}, ACL-ARC "
             f"{C('Measuring the Evolution of a Scientific Field')}",
             "the citing sentence", "not applicable", "3 or 6", "SciBERT"],
            ["Support checker",
             f"SummaC {C('SummaC')}, QAGS {C('Asking and Answering Questions')}",
             "a summary sentence", "inference or questions", "no", "an off-the-shelf model"],
            ["Retrieval-based agent", f"PaperQA {C('PaperQA')}, STORM "
             f"{C('Assisting in Writing Wikipedia-like Articles')}",
             "a retrieved passage", "retrieval only", "no", "none, prompted"],
            ["Self-criticising loop",
             f"Self-Refine {C('Self-Refine')}, Reflexion {C('Reflexion')}",
             "the whole output", "the same model", "no", "none, prompted"],
            ["This project", "Chapters 3\u20134",
             "a quoted line per claim; claim ids per sentence",
             "a separate model, thresholds fitted on a sample",
             "7, trained", "relation classifier and extractor"],
        ]
        cur = self.table(h, rows,
                         "How different approaches compare. The third column is what a "
                         "generated statement can be traced back to, and the fourth is what, "
                         "if anything, checks that the trace is real.",
                         size_pt=8.5,
                         widths_in=[1.15, 1.35, 1.15, 1.2, 0.8, 1.05],
                             key="approach-comparison")

        # 2.2 trends and gaps
        _, h = self.anchor("2.2 Key findings")
        self.bookmark_heading(h, "Key findings, trends, and research gaps", number="2.2")
        D.add_paragraphs(h, [
            "Four trends run through the work above. Checking has moved from comparing "
            "against a reference answer towards testing whether the output follows from the "
            "source, because reference comparison was shown not to track correctness. What a "
            "statement is tied to has got finer, from whole documents towards single "
            "passages. Citation types have been standardised, but at a level of detail chosen "
            "to make annotation cheap rather than to make the labels useful afterwards. And "
            "orchestration has moved from single prompts to teams of agents, usually with "
            "self-criticism as the quality mechanism.",
            "Three gaps follow, and they are what this project addresses. First, the "
            "support-checking work tests a summary against one source document, while the "
            "review-writing work generates from many sources and is scored against a reference "
            "text; almost nothing checks a many-source review against the sources it cites, "
            "sentence by sentence. Second, citation labels are used to study how fields evolve "
            "rather than to drive the writing: we know of no system that treats a predicted "
            "disagreement as evidence that two papers conflict and then asks an agent to "
            "explain it. Third, the cost of running a team of agents is rarely reported next "
            "to its quality, so the literature does not say whether the extra calls are worth "
            "making.",
            "A fourth point is about method rather than subject. Several of the agent papers "
            "report improvements without separating training and test material by date, on "
            "tasks whose reference answers may already be inside the model being tested. "
            "Section 3.1 and docs/leakage.md therefore record exactly which publication dates "
            "were used here and why the test window starts after every tested model's training "
            "cut-off.",
        ])

        # 2.4 conclusion (the template's own numbering)
        _, h = self.anchor("2.4 Conclusion")
        self.bookmark_heading(h, "Conclusion", number="2.4")
        D.add_paragraphs(h, [
            "All the pieces this project needs already exist and are well studied on their "
            "own: pre-trained summarisers, citation classifiers, inference-based support "
            "checks, agent frameworks and retrieval. What has not been put together is a "
            "pipeline where the citation relation is learned from real related-work sections, "
            "every written sentence names the evidence behind it, a separate model judges that "
            "evidence using thresholds fitted on real examples, and the whole thing is then "
            "measured against human-written sections on papers the models cannot have seen, "
            "with its running cost reported next to its quality. Chapter 3 describes how that "
            "was built and Chapter 4 reports what it achieved, including the places where it "
            "did not beat a single well-written prompt.",
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
            "The system is one Python package used in two ways: command-line scripts for "
            "running locally, and six Colab notebooks for the larger runs. The notebooks "
            "install the package and call the same functions, so there is only one copy of "
            "the logic and the two cannot drift apart. {{FIG:fig_architecture.png}} shows the parts and how "
            "information moves between them.",
        ])
        cur = self.figure(cur, "fig_architecture.png",
                          "How the system fits together. Blue boxes are agents, green boxes "
                          "are trained models, red boxes are checks, and orange boxes are "
                          "data or output. One model setting serves every agent and records "
                          "what every call cost.")

        cur = D.add_heading(cur, "3.1.1  Where the training data comes from")
        cur = D.add_paragraphs(cur, [
            "The system needs to learn what a citing sentence is claiming about the paper it "
            "cites, and no ready-made dataset provides that at the level of detail needed. The "
            "solution is to take it from published papers. A related-work section is a short "
            "literature review an expert already wrote, and it names the papers it cites, so "
            "mining those sections gives two things at once: citing sentences whose meaning is "
            "visible from the text around them, and pairs of (cited abstracts, written "
            "section) that form a test set with a correct answer. {{FIG:fig_mining_pipeline.png}} shows the steps.",
        ])
        cur = self.figure(cur, "fig_mining_pipeline.png",
                          "Mining papers from arXiv. Each paper's source files are stored "
                          "once, and a paper with no usable source is recorded as such so it "
                          "is never downloaded twice. That is what lets the step be stopped "
                          "and restarted.")
        cur = D.add_paragraphs(cur, [
            "arXiv distributes most papers as LaTeX source, which is readable text rather "
            "than a PDF. The parser handles the awkward cases, each of which showed up in a "
            "real paper during testing and each of which now has a test: papers with more "
            "than one main file, related work written as a sub-section or as a run of "
            "unnamed paragraphs rather than its own section, two different bibliography "
            "formats, a bibliography written directly into the main file, and the various "
            "citation commands used by different publishers.",
            "A heading on its own is not treated as proof that a section is a literature "
            "review. A section actually titled \u201cRelated Work\u201d must contain at least "
            "one citation, and a vaguer heading such as \u201cBackground\u201d must contain at "
            "least two. Without that rule a section explaining standard theory would be mined "
            "as though it were a survey.",
            "Cited papers then have to be identified. Where the bibliography gives a DOI or an "
            "arXiv number, that number identifies the paper exactly, and those are looked up "
            "in batches. Only where no identifier exists does the system fall back to "
            "searching by title, which is both slower and less certain; a title match is "
            "accepted only above a similarity threshold, so the wrong paper is rarely "
            "attached. Papers for which fewer than 60% of the cited works could be found with "
            "an abstract are dropped from the test set, because the task is only fair if the "
            "system can see most of what the original author saw.",
        ])

        cur = D.add_heading(cur, "3.1.2  Telling citation types apart")
        cur = D.add_paragraphs(cur, [
            "Seven labels are used: background, uses-method, uses-data, extends, "
            "compares-against, supports and contrasts. The split that matters most is between "
            "the last three. Running another paper's method as a baseline and reporting a "
            "better score is compares-against, which is a measurement. Saying your results "
            "agree with theirs is supports. Saying their conclusion does not hold is "
            "contrasts, and only that last one is evidence that two papers actually conflict.",
            "Because real sentences often fit more than one label, the rules are applied in a "
            "fixed order: disagreement beats agreement, which beats extending, which beats "
            "comparing, which beats using data, which beats using a method, with background as "
            "the fallback. The definitions and this order live in one file, so the instructions "
            "given to the labelling model, the guide a human labeller follows and the training "
            "code cannot disagree about what a label means.",
            f"Two published datasets, SciCite {C('Structural Scaffolds')} and ACL-ARC "
            f"{C('Measuring the Evolution of a Scientific Field')}, are mapped onto these "
            f"seven labels to provide extra training examples. The mapping loses information "
            f"and the report says where: SciCite's \u201cmethod\u201d class does not separate "
            f"using a method from improving it, and its \u201cresult\u201d class covers "
            f"comparing, agreeing and disagreeing alike; ACL-ARC's \u201cuses\u201d does not "
            f"separate method from data, and its \u201ccompare or contrast\u201d merges neutral "
            f"comparison with real disagreement. Because of that, mapped examples are used "
            f"only to help training and never to score anything.",
            "The classifier is SciBERT, fine-tuned to read a pair of texts. The first is the "
            "citing sentence with the citation in question replaced by a marker and any other "
            "citations in the same sentence replaced by a different marker \u2014 without that, "
            "a sentence citing five papers at once would have no single correct answer. The "
            "second is the cited paper's title. The data is split by paper rather than by "
            "sentence, because sentences from one paper resemble each other and a random split "
            "would let the model see the answer. Rare labels are weighted more heavily so they "
            "can be learned at all:",
        ])
        cur = self.equation(cur, r"$w_c \;=\; \dfrac{N}{K\,n_c}$", "3.1", width_in=3.2)
        cur = D.add_paragraphs(cur, [
            "where N is the number of training sentences, K the number of labels and n\u27e8c\u27e9 "
            "how many times label c appears. Results are reported as macro-F1, which averages "
            "the score for each label separately:",
        ], first_line_indent=None)
        cur = self.equation(cur, r"$\mathrm{F1}_{\mathrm{macro}} \;=\; \frac{1}{|\mathcal{C}|}"
                                 r"\sum_{c \in \mathcal{C}} \frac{2 P_c R_c}{P_c + R_c}$",
                            "3.2", width_in=4.2)
        cur = D.add_paragraphs(cur, [
            "with P\u27e8c\u27e9 and R\u27e8c\u27e9 the precision and recall for label c. Plain "
            "accuracy would be misleading here, because most citations are background and a "
            "model that only ever predicted background would score well while being useless "
            "for everything that follows. Agreement between the machine labels and the "
            "hand-checked ones is reported as Cohen's kappa:",
        ], first_line_indent=None)
        cur = self.equation(cur, r"$\kappa \;=\; \dfrac{p_o - p_e}{1 - p_e}$", "3.3",
                            width_in=2.8)
        cur = D.add_paragraphs(cur, [
            "where p\u27e8o\u27e9 is how often the two agree and p\u27e8e\u27e9 is how often they "
            "would agree by chance given how often each uses each label. Subtracting the "
            "chance agreement means a labeller that simply guesses the most common answer "
            "scores near zero rather than near its raw accuracy.",
        ], first_line_indent=None)

        cur = D.add_heading(cur, "3.1.3  Pulling out claims and checking them")
        cur = D.add_paragraphs(cur, [
            "The Extractor turns a paper into a structured record: what problem it tackles, "
            "what methods and datasets it uses, what numbers it reports, what it admits it "
            "cannot do, and three to five claims. Each claim carries a short line copied word "
            "for word out of the paper as its evidence. The input is the abstract plus "
            "whatever introduction, results, conclusion and limitations text the source file "
            "provided, up to a fixed length, and that exact input is stored with the record, "
            "because the next check is defined against it and not against the paper in "
            "general.",
            f"The checker then runs two tests that can fail independently. The first asks "
            f"whether the quoted line really appears in the text the Extractor was given. It "
            f"uses approximate rather than exact matching, because converting LaTeX to plain "
            f"text leaves spacing and punctuation differences that an honest quote still trips "
            f"over. The second asks whether the claim actually follows from that line, using a "
            f"DeBERTa model {C('DeBERTa')} trained on inference datasets, applied one sentence "
            f"at a time as SummaC {C('SummaC')} recommends.",
            "Both tests are needed, and the reason is the heart of this project. During "
            "testing the Extractor produced a claim whose quote matched the source perfectly "
            "and which the same quote did not support at all: it had copied the line correctly "
            "and then written something stronger than the line said. A single check would have "
            "passed it.",
            "A claim is called supported when the quote matches and the inference score is "
            "high, partially supported when the inference score is in the middle, and "
            "unsupported otherwise. The thresholds are not picked by hand. They are chosen by "
            "trying every combination on a sample labelled by hand and keeping the one with "
            "the best macro-F1 across the three outcomes, so a setting cannot be chosen just "
            "because it does well on the most common outcome. Section 4.3 reports the values "
            "that were chosen.",
            "When the review is written, the same model scores each sentence against the "
            "evidence behind the claims the Writer attached to it. The faithfulness score is "
            "the share of sentences that come out supported:",
        ])
        cur = self.equation(
            cur, r"$\mathrm{Faith} \;=\; \frac{1}{|S|}\sum_{s \in S}"
                 r"\mathbb{1}\!\left[\max_{e \in E(s)} P_{\mathrm{ent}}(s \mid e) "
                 r"\geq \tau_{\mathrm{sup}}\right]$", "3.4", width_in=5.0)
        cur = D.add_paragraphs(cur, [
            "where E(s) is the set of quoted lines behind sentence s and \u03c4\u27e8sup\u27e9 is "
            "the chosen threshold. A sentence that cites nothing has no evidence and so counts "
            "as unsupported, which is the behaviour wanted from a system that claims every "
            "sentence is grounded. Sentences that fail go back to the Writer once; if the "
            "rewrite does not score better, the first version is kept and the failing "
            "sentences are marked in the output rather than quietly deleted.",
        ], first_line_indent=None)

        cur = D.add_heading(cur, "3.1.4  Comparing papers and finding gaps")
        cur = D.add_paragraphs(cur, [
            "Papers name the same thing in different ways, so dataset, metric, method and task "
            "names are tidied up in three stages, cheapest first: a hand-written list of known "
            "alternative names kept in the repository, then approximate string matching, then "
            "meaning-based matching for cases the first two miss. Anything still unmatched "
            "keeps its original form and is reported, so names are never silently thrown away "
            "and the list of known alternatives can be extended.",
            "Records are then grouped by task, dataset and metric. Inside a group, numbers "
            "that differ by more than 15% of the middle value are flagged as a disagreement "
            "about results, and two papers that rank the same pair of methods in opposite "
            "orders on the same dataset are flagged as a disagreement about ranking. "
            "Separately, a contrasts link predicted by the classifier is flagged as a "
            "disagreement the authors stated themselves, which no comparison of numbers could "
            "reveal. Every flagged case goes to an agent that must decide whether it is "
            "explained by a different experimental setup, a different split or metric "
            "definition, a real disagreement, or not enough information \u2014 using only the "
            "quoted evidence.",
            "Gap detection builds a table of how many papers apply each family of method to "
            "each dataset. A cell that is empty or nearly empty, where both its row and its "
            "column are otherwise busy, becomes a candidate gap. Two method families that both "
            "appear on a dataset but between whose papers no comparison link exists become a "
            "candidate of a different kind. Because an empty cell is often empty for a good "
            "reason, the agent is required to argue both for and against each candidate before "
            "giving a verdict, and both arguments are kept in the output.",
        ])

        cur = D.add_heading(cur, "3.1.5  Putting the steps in order")
        cur = D.add_paragraphs(cur, [
            "The pipeline is a CrewAI flow of ten steps, shown in {{FIG:fig_flow.png}}. Each agent is kept "
            "deliberately narrow. The Librarian has search tools and nothing else. The "
            "Extractor has no tools at all, because its text is handed to it. Only the "
            "Verifier may run the checking model. The Writer can neither search nor check its "
            "own output. The approval point after the corpus is collected is a real stop, not "
            "a formality: the flow pauses and shows the list of papers before any money or "
            "quota is spent on reading them.",
        ])
        cur = self.figure(cur, "fig_flow.png",
                          "The ten steps, the point where a human approves the paper list, "
                          "and the single rewrite loop. Every step saves its output before "
                          "the next begins, so an interrupted run carries on where it stopped.")
        cur = D.add_paragraphs(cur, [
            "Saving after every step is what makes an interrupted run recoverable rather than "
            "wasted, and that matters more than it sounds. Free accounts impose limits that "
            "stop work part-way through, and a free cloud session ends after a fixed time. "
            "Both happened repeatedly while this project was built.",
        ])

        cur = D.add_heading(cur, "3.1.6  Compressing the extractor into a small model")
        cur = D.add_paragraphs(cur, [
            f"The training set for the small model is built by running the hosted Extractor "
            f"over training papers and keeping only those papers where every claim passed the "
            f"checker. That is the point of the step: the checker decides what the small model "
            f"is allowed to learn from, so it cannot copy mistakes the larger model happened "
            f"to get away with. The small model {C('Qwen2.5 Technical Report')} is adapted "
            f"using QLoRA {C('QLoRA')}, a method that stores the original model in a "
            f"compressed form and trains only a small number of added parameters, which is "
            f"what makes the training fit on a free cloud GPU. It is then compared against the "
            f"hosted model on papers neither has seen.",
        ])

        cur = D.add_heading(cur, "3.1.7  How the systems are compared")
        cur = D.add_paragraphs(cur, [
            "The test is: given only the papers a real related-work section cites, write that "
            "section. Four systems and one cut-down version are compared using the same model "
            "and the same settings. S1 is a single well-written prompt over the same "
            "abstracts. S2 is the agent team with the checker switched off. S3 is the full "
            "team. S4 is the full team using the small extractor. The cut-down version is S3 "
            "with the citation classifier switched off, which isolates what that classifier "
            "contributes.",
            "Overlap with the human text is measured by ROUGE and BERTScore, but reported "
            "beside three other measures, because overlap says nothing about whether the text "
            "is true. Citation coverage is the share of the papers the human section cited "
            "that the generated section discusses at all. Citation correctness is the share of "
            "claimed links between a sentence and a paper where that paper's evidence actually "
            "supports the sentence; this is the measure that catches a real citation attached "
            "to a sentence the paper never supported. Faithfulness is as defined in equation "
            "(3.4).",
            "Every average is reported with a confidence interval computed by resampling:",
        ])
        cur = self.equation(
            cur, r"$\mathrm{CI}_{95} = \left[\hat{\theta}^{*}_{(0.025B)},\;"
                 r"\hat{\theta}^{*}_{(0.975B)}\right],\quad B = 2000$", "3.5", width_in=4.4)
        cur = D.add_paragraphs(cur, [
            "which means the test set is resampled two thousand times and the range covering "
            "the middle 95% of the resulting averages is reported. At the sample sizes used "
            "here those ranges are wide, and showing them is what stops a difference of a few "
            "points being read as a result.",
            "Finally, the test papers are chosen by publication date. Training papers come "
            "from 2023 and 2024 and test papers from a window that begins after the training "
            "cut-off of every model being tested, so no system can have read the answer. The "
            "thresholds, the classifier's development split and the hand-labelled set all come "
            "from the training side, so nothing is tuned on the test set. The dates and the "
            "risks that remain \u2014 chiefly that the cited papers are necessarily older and so "
            "probably were in the models' training data, even though the text being scored was "
            "not \u2014 are recorded in docs/leakage.md and in Section 4.1.",
        ], first_line_indent=None)

        cur = D.add_heading(cur, "3.1.8  Looking at what the system produced")
        cur = D.add_paragraphs(cur, [
            "Everything the pipeline writes is a file, which is reliable but hard to read. A "
            "small web application was therefore built on top of the same files, so the state "
            "of a run and the evidence behind a sentence can be inspected directly. It adds no "
            "logic of its own: it reads the same results files this report quotes, through the "
            "same module, so the two cannot disagree about a number. {{FIG:fig_console_overview.png}} shows the "
            "overview, which marks a step as pending when its results file does not exist "
            "rather than showing an estimate.",
        ])
        cur = self.figure(cur, "fig_console_overview.png",
                          "The console's overview. Each step shows real figures from the file "
                          "it wrote, and a step that has not run is marked pending instead of "
                          "being filled in with a guess.", width_in=4.9)
        cur = D.add_paragraphs(cur, [
            "The most useful view is the test set itself, in {{FIG:fig_console_benchmark.png}}. Opening a paper shows "
            "the related-work section its author actually wrote, which is hidden from every "
            "system and used only for scoring, beside the abstracts that are the only input a "
            "system receives. References whose abstract could not be found are greyed out. "
            "Seeing the two together makes the difficulty of the task concrete in a way a "
            "table of scores does not.",
        ])
        cur = self.figure(cur, "fig_console_benchmark.png",
                          "A test example in the console: the section the author wrote, above "
                          "the cited papers' abstracts that are the only input a system is "
                          "given. Greyed-out entries are references whose abstract could not "
                          "be retrieved.", width_in=4.9)
        cur = D.add_paragraphs(cur, [
            "A third view shows a generated review one sentence at a time. Clicking a sentence "
            "shows the quoted line it was checked against and the score it received, and "
            "sentences are underlined green, orange or red according to the outcome. The same "
            "information is also available to a conversational assistant through a small "
            "read-only interface, so the state of a run can be asked about in words rather "
            "than read off a screen.",
        ])

        # 3.2 system requirements
        _, h = self.anchor("3.2 System requirements")
        self.bookmark_heading(h, "System requirements", number="3.2")
        cur = D.add_paragraphs(h, [
            "The project is built to run in two places: a free Google Colab session with a T4 "
            "GPU for the heavier steps, and an Apple Silicon laptop for development and for "
            "the smaller runs reported in Chapter 4. {{TAB:system-requirements}} lists what was actually used. The "
            "version numbers are the ones recorded while the code ran, not the oldest versions "
            "it would tolerate.",
        ])
        env = self.environment_rows()
        cur = self.table(cur, env,
                         "What the system was run on, and the exact versions used for the "
                         "results in Chapter 4.", size_pt=9.0, widths_in=[1.7, 2.1, 2.3],
                             key="system-requirements")
        D.add_paragraphs(cur, [
            "The free API account shapes the design more than the hardware does. It imposes "
            "four separate limits \u2014 on requests per day, on tokens per minute, on generated "
            "tokens per minute, and on tokens per day \u2014 and each one was hit while building "
            "this project. The code now keeps track of all four before sending a request, asks "
            "for only as many output tokens as a reply actually needs, and treats a daily limit "
            "as a reason to stop and report rather than to keep retrying. Every long step is "
            "written to disk as it goes, so running out of quota means continuing tomorrow "
            "rather than starting again. No key is stored in the repository: keys are read "
            "from the environment, or from Colab's secret store when running there.",
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

    def resolution_paragraph(self) -> str:
        """How cited references were resolved, in the order the pipeline tries."""
        R = self.R
        ax = R.get("stage1_yield.json", "arxiv_metadata") or {}
        doi = R.get("stage1_yield.json", "doi_batch") or {}
        oa = R.get("stage1_yield.json", "openalex") or {}
        exhausted = R.get("stage1_yield.json", "budget.openalex_daily_budget_exhausted")
        parts = [
            "References are resolved by identifier first and by title only as a fallback, "
            "because an identifier resolves a reference exactly where a title match resolves "
            "it merely probably, and because the fallback is what costs money."
        ]
        if ax.get("ids_cited_in_related_work"):
            parts.append(
                f"Of the references cited from related-work sections, "
                f"{ax['ids_cited_in_related_work']:,} carried an arXiv identifier and all "
                f"{ax.get('resolved', 0):,} of them resolved through the arXiv API, every one "
                f"with an abstract; for a corpus mined from arXiv this is also the more "
                f"authoritative source, since it returns the abstract of the version actually "
                f"cited.")
        if doi.get("dois_cited_in_related_work"):
            parts.append(
                f"A further {doi['dois_cited_in_related_work']:,} carried a DOI, resolved "
                f"fifty at a time through OpenAlex, of which {doi.get('resolved', 0):,} came "
                f"back ({doi.get('with_abstract', 0):,} with an abstract).")
        if oa.get("titles_needing_search") is not None:
            parts.append(
                f"That left {oa['titles_needing_search']:,} references with no identifier at "
                f"all, which had to be resolved by title search; {oa.get('resolved', 0):,} "
                f"matched above the similarity floor of 88 and "
                f"{oa.get('with_abstract', 0):,} of those yielded an abstract.")
        if exhausted:
            parts.append(
                "The title-search pass did not finish: OpenAlex's free tier is a small daily "
                "budget per key, and it ran out partway through. Every reference resolved so "
                "far is cached, so re-running after the reset continues rather than restarts, "
                "and the benchmark reported below is therefore a lower bound on what this "
                "corpus yields.")
        parts.append(
            "The gap between finding a reference and being able to read its abstract "
            "is the main reason test examples are dropped: a paper can be identified "
            "without its abstract being available to redistribute.")
        return " ".join(parts)

    REF_TOKEN = re.compile(r"\{\{(FIG|TAB):([^}]+)\}\}")

    def resolve_refs(self) -> None:
        """Replace reference tokens with the numbers the captions actually got.

        Prose usually mentions a figure before the figure is inserted, so writing
        the number by hand means it silently goes stale the moment a figure is
        added anywhere earlier. Prose writes a token instead and this pass fills
        in the real number, which is also what keeps every figure referenced.
        """
        fig_by = {it["source"]: it["n"] for it in self.fig.items}
        tab_by = {it["source"]: it["n"] for it in self.tab.items}

        def sub(m):
            kind, name = m.group(1), m.group(2)
            table = fig_by if kind == "FIG" else tab_by
            n = table.get(name)
            if n is None:
                self.unresolved_refs.append(f"{kind}:{name}")
                return "the figure below" if kind == "FIG" else "the table below"
            return f"{'Figure' if kind == 'FIG' else 'Table'} {n}"

        def walk(par):
            if "{{" not in par.text:
                return
            for r in par.runs:
                if "{{" in (r.text or ""):
                    r.text = self.REF_TOKEN.sub(sub, r.text)
            if "{{" in par.text:                   # token split across runs
                D.set_text(par, self.REF_TOKEN.sub(sub, par.text))

        for par in self.doc.paragraphs:
            walk(par)
        for t in self.doc.tables:
            for row in t.rows:
                for cell in row.cells:
                    for par in cell.paragraphs:
                        walk(par)

    def demo_run(self) -> dict | None:
        """The saved demonstration run, if one exists.

        A demo regenerates one benchmark paper's related-work section from only
        the abstracts it cites, which is the whole task in miniature and the one
        place in this report where a reader can see a generated sentence, the
        line it was checked against, and the score side by side.
        """
        root = paths().results / "flow_runs"
        if not root.exists():
            return None
        dirs = sorted(d for d in root.glob("demo_*") if (d / "summary.json").exists())
        if not dirs:
            return None
        d = dirs[-1]
        out = {"dir": d}
        for name in ("summary", "review_verdicts", "reference"):
            f = d / f"{name}.json"
            out[name] = json.loads(f.read_text()) if f.exists() else {}
        return out

    def chapter4(self) -> None:
        R, C = self.R, self.refs.cite
        _, ch = self.anchor("CHAPTER 4", exact=True)
        self.bookmark_heading(ch, "ANALYSIS AND DISCUSSION", kind="chapter", number="4")
        _, title = self.anchor("ANALYSIS AND DISCUSSION")

        cur = D.add_paragraphs(title, [
            "Every number in this chapter comes from a file in results/ that the code "
            "wrote, and every figure is drawn from those files by one script. Where a step "
            "was not run, the chapter says so instead of estimating. The size of each run is "
            "given with its results, because these are small runs and a single number without "
            "its uncertainty would claim more than the evidence supports.",
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
                f"papers submitted only as a PDF, which arXiv allows and which no parser "
                f"can read. "
                f"{y.get('parsed')} parsed successfully and {y.get('with_related_work')} had an "
                f"identifiable related-work section "
                f"({100 * (y.get('with_related_work') or 0) / listed:.0f}% of papers listed). "
                f"After requiring at least three cited works and at least 60% abstract coverage, "
                f"{ds.get('rw_pairs_kept')} papers remain as benchmark examples and "
                f"{ds.get('contexts_rows'):,} citation contexts were written. {{FIG:fig_stage1_yield.png}} shows how many survive each step and {{TAB:stage1-yield}} gives the same figures as a table; {{FIG:fig_dataset_stats.png}} shows the "
                f"attrition at each step.",
                self.resolution_paragraph(),
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
                             size_pt=9.5, widths_in=[2.8, 1.2, 1.2], align_right_cols=(1, 2),
                                 key="stage1-yield")

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
                                 size_pt=9.5, widths_in=[2.9, 1.3, 1.3], align_right_cols=(1, 2),
                                     key="dataset-stats")
                cur = D.add_paragraphs(cur, [
                    f"The time split is disjoint: the newest training paper is dated "
                    f"{ts.get('train_latest')} and the oldest test paper "
                    f"{ts.get('test_earliest')}." if ts.get("disjoint") else
                    f"The time split check reports train_latest={ts.get('train_latest')} and "
                    f"test_earliest={ts.get('test_earliest')}; these are not disjoint and the "
                    f"benchmark should not be used until the windows are corrected.",
                    "One risk remains and is worth stating. The cited papers are "
                    "necessarily older than the paper citing them, so they were probably in "
                    "the training data of every model tested. That cannot be avoided for this "
                    "task. It does not give a model the text being scored, which is the "
                    "section itself, but it does mean the subject matter may be familiar.",
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
                    f"of eight, each label accompanied by a short reason that had to quote "
                    f"the words in the sentence that justified it. The distribution is "
                    f"dominated by background, as expected: most citations simply point at "
                    f"earlier work without using or disputing it.",
                ])
                cur = self.figure(cur, "fig_label_distribution.png",
                                  "Distribution of LLM-assigned citation relations over the "
                                  "labelled pool.")
            if lvg:
                cur = D.add_paragraphs(cur, [
                    f"A gold set of {ag.get('n_gold')} citation sentences was labelled "
                    f"without seeing the machine's answers, and deliberately sampled "
                    f"so the rare labels appear at all rather than being swamped by "
                    f"background. Who produced these labels matters, so it is stated "
                    f"plainly: they were assigned by the AI assistant that built this "
                    f"pipeline, following the written guide and blind to the "
                    f"labeller's output, then reviewed by the author. That is an "
                    f"independent second opinion, not expert human annotation, and "
                    f"the agreement figures should be read that way. Against this set "
                    f"the machine labeller "
                    f"reaches {lvg.get('accuracy', 0):.3f} accuracy and "
                    f"{lvg.get('macro_f1', 0):.3f} macro-F1, with a kappa of "
                    f"{lvg.get('cohen_kappa', 0):.3f}. Because the sample is stratified rather "
                    f"than random, this is not how accurate the labeller would be on a "
                    f"sentence picked at random: the sample is weighted towards the cases it "
                    f"finds hardest, so the figure is a lower bound.",
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
                                 f"{bench.get('n_gold', '?')}-context adjudicated gold set. The gold "
                                 f"set deliberately over-samples the hard cases, so the "
                                 f"absolute scores are pessimistic; the ordering is the "
                                 f"result.",
                                 size_pt=9.5, widths_in=[3.0, 1.2, 1.2],
                                 align_right_cols=(1, 2),
                                     key="relation-results")
                sci = bench.get("scibert") or {}
                tf = bench.get("tfidf_lr") or {}
                if sci and tf:
                    verdict = ("beats" if sci["macro_f1"] > tf["macro_f1"] else
                               "does not beat")
                    cur = D.add_paragraphs(cur, [
                        f"The fine-tuned classifier {verdict} the word-counting baseline "
                        f"on macro-F1 "
                        f"({sci['macro_f1']:.3f} against {tf['macro_f1']:.3f}). "
                        + ("The margin is what the downstream stages rely on: a contrasts edge "
                           "is only useful as evidence of disagreement if the classifier "
                           "can tell it apart from a neutral comparison."
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
                              "{{FIG:fig_relation_confusion.png}} shows where their mass goes instead.",
                        ])
            cur = self.figure(cur, "fig_relation_confusion.png",
                              "Confusion matrices on the adjudicated gold set, for the "
                              "LLM labeller and for the fine-tuned classifier. Cell text is the "
                              "raw count.")
            cur = self.figure(cur, "fig_relation_models.png",
                              "Accuracy and macro-F1 of each relation-classification system on "
                              "the adjudicated gold set.")
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
                f"The two thresholds were chosen by trying every combination on "
                f"{cal.get('n_dev_items')} hand-checked claim-and-quote pairs and keeping the "
                f"one with the best macro-F1 across the three outcomes. The fitted values are a quote-match floor of "
                f"{th.get('quote_min')} and entailment thresholds of "
                f"{th.get('entail_supported')} for supported and {th.get('entail_partial')} for "
                f"partially supported, reaching {cal.get('best_macro_f1')} macro-F1 on the dev "
                f"set"
                + (f", against {cal.get('defaults_macro_f1')} for the hand-chosen defaults."
                   if cal.get("defaults_macro_f1") is not None else ".")
                + " {{FIG:fig_verifier_calibration.png}} shows the dev items against the fitted boundaries.",
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
                f"{v.get('mean_entail_prob')}. The difference between those two figures is "
                f"the main finding of this step: a model asked to quote exactly usually does, "
                f"and still writes claims its own quote does not prove. "
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
                f"and one set of settings. {{TAB:main-eval}} gives the main results; every cell is a mean "
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
                                 widths_in=[1.25, 0.3, 0.75, 0.75, 0.75, 0.75, 0.75, 0.8, 0.8],
                                     key="main-eval")
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
                                 align_right_cols=(1, 2, 3, 4, 5),
                                     key="cost")
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
                    ["Monetary cost per paper", "API tokens (see {{TAB:cost}})",
                     "0 (runs locally)"]]
            cur = self.table(cur, rows,
                             "Hosted extractor against the distilled extractor on held-out "
                             "test-split papers.",
                             size_pt=9.0, widths_in=[2.6, 1.5, 1.5], align_right_cols=(1, 2),
                                 key="distillation")
        else:
            cur = self._not_run(
                cur, "Stage 6 distillation",
                "results/stage6_distill.json is absent. 4-bit QLoRA requires bitsandbytes, "
                "which is CUDA-only, so this stage is a Colab-T4 job and was not run on the "
                "development machine.",
                "notebooks/05_distill_extractor_qlora.ipynb on a Colab T4")

        # --------------------------------------- 4.7 qualitative + limitations
        cur = D.add_heading(cur, "4.7  A worked example, end to end")
        demo = self.demo_run()
        if demo:
            sm, vd = demo["summary"], demo["review_verdicts"]
            ref = demo.get("reference") or {}
            faith = (vd.get("after_revision") or {})
            sents = vd.get("sentence_verdicts") or []
            n = len(sents)
            sup = faith.get("supported", 0)
            st = sm.get("settings") or {}
            model_name = st.get("model") or "the configured model"
            local = (st.get("provider") or "").lower() == "ollama"
            cur = D.add_paragraphs(cur, [
                f"To show the whole pipeline on one case, the system was given a single test "
                f"paper \u2014 {ref.get('arxiv_id', '')}, "
                f"\u201c{(ref.get('title') or '')[:90]}\u201d \u2014 with its related-work section "
                f"hidden and only the abstracts of the {len(ref.get('cited') or [])} papers it "
                f"cites as input. The author's own section is "
                f"{len((ref.get('rw_text') or '').split())} words long.",
                f"The crew extracted {sm.get('n_claims', 0)} claims from "
                f"{sm.get('n_records', 0)} papers and wrote {n} sentences, taking "
                f"{sm.get('wall_s', 0):.0f} seconds. The checker then found "
                f"{sup} of those {n} sentences supported by the evidence behind them.",
            ])
            # Illustrate with a sentence that *has* evidence: one citing nothing is
            # unsupported by construction and shows nothing about the checker.
            withev = [v for v in sents if v.get("best_evidence")]
            if n and sup == 0 and withev:
                worst = min(withev, key=lambda v: v.get("entail_prob") or 0)
                cur = D.add_paragraphs(cur, [
                    "That result \u2014 none of the sentences supported \u2014 is worth looking at "
                    "closely, because it is the kind of failure this project was built to "
                    "make visible rather than the kind it was built to avoid. The sentences "
                    "read well and cite real papers from the correct set. What they do not do "
                    "is say something the cited papers actually establish. Most describe what "
                    "the citing paper itself contributes, which is reasonable prose for a "
                    "related-work section and is exactly what cannot be checked against the "
                    "cited literature.",
                    f"One sentence shows the pattern. It reads: "
                    f"\u201c{_clip(worst.get('text'), 210)}\u201d The closest supporting line the "
                    f"checker could find in the paper that sentence cites is: "
                    f"\u201c{_clip(worst.get('best_evidence'), 210)}\u201d The two are about the "
                    f"same subject, which is why the sentence looks convincing, but the second "
                    f"does not establish the first, and the score is "
                    f"{worst.get('entail_prob', 0):.3f} out of 1.",
                    "A reader skimming the citations would not catch this. An overlap score "
                    "against the author's section would not catch it either, because the "
                    "wording is close to what a related-work section normally looks like. The "
                    "sentence-level check catches it, which is the argument for running the "
                    "check at all.",
                ])
            elif n:
                cur = D.add_paragraphs(cur, [
                    f"{sup} of {n} sentences came through supported, and the ones that did not "
                    f"are marked in the output rather than removed, so a reader can see which "
                    f"parts of the text to treat with caution.",
                ])
            cur = D.add_body_paragraph(
                cur, "{{FIG:fig_console_review.png}} shows the same run in the console, "
                     "where selecting a sentence brings up the line it was checked against.")
            cur = self.figure(cur, "fig_console_review.png",
                              "The worked example in the console. Sentences are underlined by "
                              "outcome, and selecting one shows the paper it cites, the "
                              "closest supporting line, and the score. Here every sentence is "
                              "unsupported.", width_in=5.2)
            cur = D.add_paragraphs(cur, [
                f"Two things about this run should be kept in mind when reading the number. "
                f"It used {model_name}"
                + (", a small model running on this laptop" if local else "")
                + ", because the hosted "
                f"account's daily allowance had already been spent on labelling; a larger "
                f"model would be expected to do better. And it is one paper, so it "
                f"demonstrates the mechanism rather than measuring it. Section 4.5 is where "
                f"measurement across many papers belongs.",
            ])
        else:
            cur = self._not_run(
                cur, "The worked example",
                "no demonstration run exists under results/flow_runs/.",
                "python scripts/run_demo.py")

        cur = D.add_heading(cur, "4.7.1  Error analysis")
        cur = D.add_paragraphs(cur, self.qualitative_section())

        cur = D.add_heading(cur, "4.8  Limitations and threats to validity")
        cur = D.add_bullets(cur, [
            "Small runs. Everything here was run at a size that fits on one laptop and inside "
            "a free API account. The confidence ranges are correspondingly wide, and no claim "
            "is made that one system beats another where those ranges overlap.",
            "The classifier learns from machine labels. The citation classifier is trained on "
            "labels a language model produced, not a person. Section 4.2 measures how far "
            "those labels can be trusted, and the classifier cannot be better than its teacher "
            "on the labels the teacher gets wrong.",
            "Who produced the gold labels. The hand-checked sets were labelled by the AI "
            "assistant that built the pipeline, following the written guide and blind to the "
            "labeller's output, then reviewed by the author. That is an independent second "
            "opinion rather than expert human annotation. If the guide itself is misleading, "
            "it would mislead the labeller and the gold set in the same direction, and with "
            "one annotator there is no way to measure that. Replacing these with "
            "independently sourced human labels is the first item in Section 5.2.",
            "The checking model has its own weaknesses. It is unreliable on statements "
            "involving numbers and quantities, which is exactly what many scientific claims "
            "turn on, and it treats cautious wording inconsistently. A claim marked "
            "unsupported is a sign that something needs looking at, not proof that it is "
            "wrong.",
            "Matching a quote is not the same as proving provenance. The check confirms that "
            "a line resembling the quote appears in the text given to the extractor. It does "
            "not confirm that the line means in context what the claim says it means, and a "
            "short generic line can match in several places.",
            "The date separation is partial. The section being scored was published after "
            "every tested model's cut-off, but the papers it cites are necessarily older and "
            "were probably in those models' training data. Familiarity with the subject may "
            "therefore help a model even though it has not seen the answer.",
            "The test set is not a random sample of research. Only papers with readable "
            "source, an identifiable related-work section and most of their references "
            "findable survive. That favours conventionally structured papers from authors who "
            "publish their source files.",
            "There is only one correct answer in the test set. Many different related-work "
            "sections would be reasonable for the same set of cited papers, and only the one "
            "the original author wrote is available. This is why the grounding measures are "
            "reported next to the overlap scores rather than underneath them.",
            "Name tidying is incomplete. Dataset, metric and method names are merged using a "
            "known-alternatives list plus approximate matching; anything unmatched keeps its "
            "original form, so one group of results can end up split in two and a "
            "disagreement between papers missed as a result.",
            "One provider, one run. Every system used the same model and settings, which makes "
            "the comparison fair but says nothing about whether the conclusions hold for a "
            "different model. Each configuration was run once, so run-to-run variation is not "
            "measured.",
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
                    f"{f1:.2f}. The confidence ranges in {{TAB:main-eval}} should be read before "
                    f"treating either margin as settled.")
            else:
                verdict.append(
                    f"The crew does not beat a single well-prompted call on this benchmark: "
                    f"ROUGE-L {r3:.3f} against {r1:.3f} and faithfulness {f3:.2f} against "
                    f"{f1:.2f}. That is reported here as the finding it is. The machinery does "
                    f"deliver something the single call cannot — per-sentence provenance, a "
                    f"contradiction map and an auditable record of what was verified — but on "
                    f"the measures in {{TAB:main-eval}} it did not pay for its extra calls at this scale.")
        else:
            verdict.append(
                "The full system comparison was not completed when this report was built, so no "
                "conclusion is drawn about whether the multi-agent pipeline outperforms a "
                "single call. Section 4.5 states exactly which systems were run.")

        D.add_paragraphs(h, [
            "This project built a tool that writes a literature review and is forced to show "
            "its working. Claims must quote the line they came from. A separate model, with "
            "thresholds chosen by testing rather than guessing, decides whether the quote "
            "actually supports the claim. Every sentence of the written review carries the "
            "claims behind it, and anything that fails the check is rewritten once and then "
            "marked rather than hidden. Two parts of it were trained on data taken from real "
            "papers' related-work sections, and the whole thing was tested against "
            "human-written sections on papers published after the models' training cut-off.",
        ] + verdict + [
            "Three smaller findings are worth stating separately. A model asked to quote "
            "exactly usually does, and still writes claims its own quote does not prove, "
            "which is why the two checks are separate rather than one. Using the checker to "
            "decide what a smaller model may learn from is a cheap way to stop it inheriting "
            "the larger model's unsupported habits. And the cost of running a team of agents "
            "is large and easy to measure, so reporting it next to the quality is what makes "
            "the trade-off visible at all.",
            "Finally, the rules this project set itself shaped the result as much as the "
            "design did. Every number here is read from a file the code wrote. Every figure "
            "is drawn from those files. Six candidate references were dropped because no "
            "source would confirm they existed. The steps that could not be run on the "
            "available hardware are named rather than guessed at.",
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
            D.add_body_paragraph(lof, "Page numbers are Word fields; press Ctrl+A then F9 "
                                      "to refresh them after editing.",
                                 size_pt=9.0, italic=True, first_line_indent=None,
                                 space_after_pt=2)
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
                   align_right_cols=tuple(range(1, len(rows[0]))),
                       key="error-categories")

    # ------------------------------------------------------------------- build

    def page_breaks(self) -> None:
        """Start each chapter and the front-matter sections on a fresh page."""
        for text in ("CHAPTER 1", "CHAPTER 2", "CHAPTER 3", "CHAPTER 4", "CHAPTER 5"):
            try:
                _, p = self.anchor(text, exact=True)
                D.page_break_before(p)
            except KeyError:
                pass
        try:                                   # the two lists share a page
            _, p = self.anchor("LIST OF TABLES")
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
        self.resolve_refs()
        self.front_matter()
        self.page_breaks()
        self.removed_blanks = D.tighten(self.doc)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(self.out))
        return {
            "out": str(self.out),
            "figures": len(self.fig), "tables": len(self.tab),
            "references": len(self.refs.order),
            "unresolved_citations": self.refs.unresolved,
            "missing_figures": self.missing_figures,
            "unresolved_references": self.unresolved_refs,
            "blank_space_removed": getattr(self, "removed_blanks", {}),
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
        try
            close every document saving no
        end try
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
                           timeout=600)
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
