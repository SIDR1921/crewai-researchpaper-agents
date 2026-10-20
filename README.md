# Grounded literature review generation with multi-agent LLMs and citation-relation learning

A CrewAI-based research-paper summarisation and literature-review assistant in which
**every sentence of the output is traceable to a quoted passage of a specific paper**
and is checked by an NLI Verifier before it survives.

Given a topic or a few seed papers the system collects a corpus, extracts a structured
record from each paper in which every claim quotes its evidence verbatim, verifies each
claim, builds a consensus/contradiction map, detects research gaps, and writes a
related-work section whose every sentence carries the identifiers of the claims behind
it. Two components are fine-tuned rather than prompted:

* **A — citation-relation classifier.** SciBERT fine-tuned on a seven-class taxonomy,
  trained on contexts mined out of real papers' related-work sections.
* **B — distilled extractor.** A small instruct model QLoRA-tuned on the hosted
  extractor's output, filtered so that only extractions whose every claim passed the
  Verifier become training data.

The benchmark comes from the same idea: a published paper's related-work section is a
literature review a human wrote for a known set of cited papers, so hiding the section
and supplying only those papers' abstracts gives a task with a human reference.

## Repository layout

```
src/litreview/
  config.py            paths, provider presets, pinned HF checkpoints
  llm.py               CrewAI LLM factory, usage logging, rate limiting
  schemas.py           PaperRecord, Claim, ReviewSentence, verdicts, gaps
  mining/              arXiv listing + source download, LaTeX parsing, OpenAlex,
                       the stage-1 pipeline, Multi-XScience loader
  corpus.py            topic/seeds -> ranked corpus (OpenAlex + S2 snowballing)
  relations/           taxonomy, LLM labelling, SciBERT training, baselines,
                       SciCite / ACL-ARC loaders
  extract.py           Extractor with three interchangeable backends
  verify.py            quote check + NLI entailment, threshold calibration
  normalise.py         alias table + fuzzy + embedding name normalisation
  synthesis.py         consensus map, contradictions, Adjudicator, typed graph
  gaps.py              coverage matrix, candidate gaps, Gap Analyst
  review.py            grounded Writer, revise loop, S1 single-call baseline
  crew.py / tools.py   the six agents and the tools each one is allowed
  flow.py              the ten-step CrewAI Flow
  distill.py           Verifier-filtered distillation set + QLoRA
  eval/                metrics, systems S1-S4, the harness
  docx_tools.py        python-docx helpers for filling the report template
notebooks/             the six Colab notebooks (generated; see below)
scripts/               one CLI per stage, plus figures, diagrams and the report
app/streamlit_app.py   demo UI that drives the Flow one step at a time
results/               every number the report quotes
figures/               every figure the report embeds
report/                the generated .docx and .pdf
tests/                 60 unit tests
```

## Setup

Python 3.10-3.13 (3.12 recommended; PyTorch has no 3.14 wheels yet).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[ml,eval,dev]"
cp .env.example .env     # then add GROQ_API_KEY (or GEMINI_API_KEY)
.venv/bin/pytest -q      # 60 tests
```

Without any API key the project runs end to end against a local Ollama model
(`LITREVIEW_PROVIDER=ollama`), which is slower but has no rate limits. Keys are read
from Colab secrets on Colab and from the gitignored `.env` otherwise; nothing is ever
hard-coded.

## Running it locally

Each stage caches everything it does, so every command below is safe to re-run and
resumes where it stopped.

```bash
# Stage 1 - mine arXiv related-work sections and citation contexts (~45 min at pilot scale)
.venv/bin/python scripts/run_stage1.py --n-train 150 --n-test 50
.venv/bin/python scripts/dataset_stats.py

# Stage 2 - label citation relations, hand-check 200, fine-tune SciBERT
.venv/bin/python scripts/run_stage2.py label --limit 1200
.venv/bin/python scripts/sample_gold.py --n 200        # writes data/gold_blind.jsonl
#   ... label it by hand, save as data/gold_labelled.jsonl ...
.venv/bin/python scripts/score_gold.py
.venv/bin/python scripts/run_stage2.py train
.venv/bin/python scripts/run_stage2.py bench

# Stage 3 - extract records, calibrate the Verifier, verify every claim
.venv/bin/python scripts/run_stage3.py extract --n 20
.venv/bin/python scripts/run_stage3.py make-dev --n 60 # writes data/verifier_dev.jsonl
#   ... label it by hand ...
.venv/bin/python scripts/run_stage3.py calibrate
.venv/bin/python scripts/run_stage3.py verify

# Stages 4-5 - the whole Flow on one topic, with the approval checkpoint
.venv/bin/python scripts/run_flow.py \
  --topic "faithfulness and hallucination evaluation in abstractive summarisation" \
  --n-papers 15

# Stage 7 - the evaluation harness
.venv/bin/python scripts/run_eval.py --systems S1 S2 S3 S3_no_relations --limit 30

# Figures, diagrams, usage and the report
.venv/bin/python scripts/usage_report.py
.venv/bin/python scripts/make_diagrams.py
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/build_report.py --pdf
```

Stage 6 (QLoRA distillation) needs CUDA for 4-bit quantisation and is a Colab-T4 job;
see `notebooks/05_distill_extractor_qlora.ipynb`.

The demo UI:

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

## Running it on Colab

See [RUN_ON_COLAB.md](RUN_ON_COLAB.md) for the notebook order, expected T4 runtimes, the
secrets each notebook needs, the Drive folder layout and the free-tier budget. The
notebooks `pip install -e` this repository and call the same package functions the CLIs
call, so there is no second implementation to keep in sync. They are generated from one
spec:

```bash
.venv/bin/python scripts/build_notebooks.py
```

## Honesty rules this repository enforces

These are mechanical, not aspirational:

* **Every number in the report is read from a file in `results/`.** `build_report.py`
  reads through an accessor that records every lookup it could not satisfy, and a stage
  with no results file produces an explicit "not reported, run X" paragraph instead of an
  estimate. The gaps are printed at the end of every build and saved to
  `results/report_build.json`.
* **Every figure is generated from those files** by `scripts/make_figures.py`, which
  skips (and records) any figure whose results file is missing rather than drawing
  something plausible.
* **Every reference is verified** against OpenAlex, then Semantic Scholar, then the arXiv
  API before it can be cited (`scripts/verify_references.py` →
  `results/references_verified.json`). The report's citation registry refuses any key that
  does not resolve to a verified entry. Six candidates were dropped for this reason,
  including two textbooks whose titles fuzzy-matched to *reviews* of themselves.
* **Negative results are reported as findings.** The evaluation commentary in Chapter 4 is
  generated from the aggregates and says plainly when the multi-agent crew does not beat
  the single-call baseline, and when confidence intervals overlap.
* **Leakage is documented, not assumed away.** See [docs/leakage.md](docs/leakage.md) for
  the submission windows, why the test window starts after the evaluated models' knowledge
  cutoffs, and the residual risks that remain.

## Results summary

See `results/` for the machine-readable source of every figure below, and Chapter 4 of
the report for the discussion. This section is regenerated by hand from
`results/*.json` after a run; the authoritative numbers are in those files and in
`report/LitReview_Report.pdf`.

## Licence and provenance

Student mini-project for *Introduction to Artificial Intelligence (CSS_2203)*, School of
Computer Engineering, MIT Bengaluru. Third-party datasets (SciCite, ACL-ARC,
Multi-XScience) are downloaded from their original sources at run time and are not
redistributed here.
