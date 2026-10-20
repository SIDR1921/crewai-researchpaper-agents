# Project brief: CrewAI literature-review assistant with grounded claims, gap detection, fine-tuned components, and an evaluation harness

You are building, running, and writing up a complete mini-project. Read this whole brief before starting. Work in `~/Claude/litreview` (this folder). Already present:

- `01_mine_arxiv.ipynb`: working Colab notebook that mines arXiv LaTeX sources (stage 1, described below). Reuse it and don't rewrite it from scratch.
- `latex_mining.py`: the tested parsing library used by that notebook.
- `Report_Template.docx`: the college report template the final report must follow.
- This file.

## 0. Ground rules

1. **Assigned scope:** "a CrewAI-based research paper summarization and literature review assistant". Everything below stays inside that scope. The aim is depth, not a feature list.
2. **Compute:** the final heavy runs (labelling at scale, fine-tuning, full evaluation) happen on **Google Colab only** (free T4 GPU, about 12 h session limit, disconnects possible). Every notebook must:
   - mount Google Drive and cache all intermediate outputs under `/content/drive/MyDrive/litreview/`
   - be **resumable**: re-running all cells continues from where it stopped
   - read API keys from Colab secrets (`from google.colab import userdata`), never hard-coded
   - run top to bottom with no manual edits except a clearly marked config cell
3. **You also run it yourself.** On this Mac (Apple Silicon, `torch` with MPS, Ollama installed), run every stage end to end at **pilot scale** so the code is proven and real numbers exist for the report. Package the same code as Colab notebooks for full scale. Keep one shared Python package so notebook and local code never diverge. Notebooks do `pip install` of the repo (or `%%writefile` the modules) instead of copy-pasting different logic.
4. **LLM access:** all LLM calls go through CrewAI's `LLM` class (LiteLLM model strings) and are set from one config. Support:
   - a free hosted model (Gemini free tier or Groq; **check which models are currently available and free before choosing**, since names change)
   - a local Ollama model for development
   
   Log tokens, latency and cost for every call.
5. **Honesty (non-negotiable):**
   - Every number in the report comes from a results file the code wrote (`results/*.json`).
   - Every figure is generated from those files by a script.
   - If something underperforms (for example, the multi-agent crew doesn't beat a single LLM call), report it plainly as a finding.
   - Never invent references. Verify each reference exists (title, authors, venue, year) via OpenAlex / Semantic Scholar / arXiv before citing it.
   - If a stage could not be run, say so in the report rather than estimating.
6. **Time split / leakage:** test papers must be published after the training window and after the training cutoff of any LLM being evaluated. Document the dates used and why.
7. Use a task list. Commit to git with sensible messages as you go.

## 1. What the system does

Given a topic (or a few seed papers), the assistant:
1. collects a relevant corpus
2. extracts structured claims from each paper
3. builds a **consensus / contradiction map**
4. detects **research gaps**
5. writes a literature review in which **every sentence is grounded** in a specific passage of a specific paper and checked by a Verifier

A typed citation-relation classifier, fine-tuned on data reverse-engineered from real papers' related-work sections, drives the contradiction and gap analysis.

The "reverse-engineering" idea: every paper's related-work section is a human-written mini literature review. Mining those sections together with the cited papers shows how experts relate and summarise papers. That gives:
- training data for the relation classifier (A)
- distillation data for a small extractor model (B)
- a natural benchmark: given only the papers a real paper cites, regenerate its related-work section and compare against the real one

## 2. Repository layout (target)

```
litreview/
  pyproject.toml
  src/litreview/
    mining/latex_mining.py       # existing file, moved here
    mining/openalex.py           # title->metadata resolution with cache
    corpus.py                    # topic/seed -> corpus (OpenAlex + Semantic Scholar snowballing)
    schemas.py                   # Pydantic models
    relations/                   # taxonomy, LLM labelling, SciBERT train + inference
    extract.py                   # Extractor agent/task
    verify.py                    # NLI-based Verifier
    synthesis.py                 # consensus/contradiction map
    gaps.py                      # gap detection
    crew.py                      # agents + tasks
    flow.py                      # CrewAI Flow orchestration
    eval/                        # metrics, baselines, runners
    llm.py                       # LLM config, logging of tokens/latency/cost
  notebooks/
    01_mine_arxiv.ipynb
    02_relation_labels_and_scibert.ipynb
    03_crew_extract_verify.ipynb
    04_synthesis_and_gaps.ipynb
    05_distill_extractor_qlora.ipynb
    06_evaluation.ipynb
  app/streamlit_app.py           # optional demo UI, only after everything else works
  results/                       # json/csv written by code; source of truth for the report
  figures/                       # generated by scripts/make_figures.py
  report/                        # final .docx
  tests/
```

## 3. Stages in detail

### Stage 1: data mining (exists: `01_mine_arxiv.ipynb`)

**What it does:**
- lists cs.CL + cs.IR papers from the arXiv API for a train window (2023–2024) and a test window (Mar–Sep 2026)
- downloads LaTeX sources, inlines `\input`, strips comments
- finds the related-work section and converts citations to `[CITE:key]` markers
- maps keys to titles via `.bbl` / `.bib`
- resolves titles to OpenAlex (fuzzy match ≥ 88) for abstracts, year and citation count

**Outputs:**
- `citation_contexts.jsonl`: one row per citing sentence × cited key
- `rw_pairs.jsonl`: related-work text + cited papers' abstracts; papers with < 60% abstract coverage are dropped

**Your tasks:**
- Run it at pilot scale and check yield. Fix parser failures you find (for example `\paragraph`-only structure, `biblatex` `.bbl` format, related work as a subsection, multiple `.tex` mains).
- Add unit tests for any new case.
- Report yield statistics (papers listed → with source → with related work → kept).
- Also load **Multi-XScience** (github.com/yaolu/Multi-XScience) as an external benchmark with the same schema.

### Stage 2: citation-relation classifier (component A, fine-tuned)

- **Taxonomy:** `background`, `uses-method`, `uses-data`, `extends`, `compares-against`, `supports`, `contrasts`. Write a 1-page labelling guide with definitions and two real examples each.
- **Seed data:** SciCite and ACL-ARC (map their labels onto the taxonomy where possible; document the mapping and its limits).
- **LLM labelling:** label ~3–5k mined contexts with the configured LLM, using structured output plus a short rationale.
- **Gold set:** sample 200 contexts and label them yourself, independently of the LLM. Show these to the user for a final check before treating them as gold. Report LLM-vs-gold agreement (accuracy, Cohen's κ, per-class confusion).
- **Fine-tuning:** fine-tune `allenai/scibert_scivocab_uncased`.
  - Input: the citation context, with the citing sentence marked.
  - Use class weights for imbalance.
  - Split by citing paper, so no paper appears in both train and test.
  - Report macro-F1 and per-class F1 on the human gold set.
- **Baselines:** majority class, TF-IDF + logistic regression, zero-shot LLM.
- Save the model to Drive. Expose `classify(context) -> (label, prob)`.

### Stage 3: extraction + Verifier (#2 and #1)

**Pydantic schemas:**
- `PaperRecord {paper_id, title, year, task, methods[], datasets[], metrics[{name, value, setting, higher_is_better}], claims[{claim_id, text, type: finding|method|limitation|comparison, evidence: {section, passage}}], limitations[]}`
- `ReviewSentence {text, supporting_claim_ids[], cited_paper_ids[]}`

**Extractor agent:**
- Uses `output_pydantic`.
- Input is the paper's abstract plus, where available, the introduction/conclusion/results text from the LaTeX source.
- Every claim must quote its evidence passage verbatim from the input.

**Verifier:**
- Two checks per claim:
  - (a) string/fuzzy check that the evidence passage really exists in the source
  - (b) NLI entailment of claim ← evidence, using a DeBERTa-v3 MNLI/FEVER/ANLI model from Hugging Face (pick a currently available one)
- Thresholds are set on a small hand-checked dev set, not guessed.
- Outputs per-claim `supported | partially | unsupported`.
- At review time, the same model checks each review sentence against its cited claims' evidence. Unsupported sentences go back to the Writer once for revision, then are dropped or flagged.
- **Faithfulness score** = fraction of review sentences supported.

### Stage 4: consensus / contradiction map and gap detection (#2 and #3)

**Normalisation:** merge synonymous dataset, metric and method names using fuzzy matching plus sentence embeddings, with a manual alias file that is checked into the repo.

**Consensus / contradiction map:**
- Group records by (task, dataset, metric).
- Flag contradictions where reported results disagree on direction or ranking for comparable settings.
- Also add classifier `contrasts` / `supports` edges between papers.
- An Adjudicator agent explains each flagged contradiction using only the evidence (different setting, different split, genuine disagreement).

**Gap detection:**
- Build a method-family × dataset/task matrix, plus trends by year.
- Sparse or empty cells, and methods never `compares-against` each other, become candidate gaps.
- A Gap Analyst agent ranks the candidates and argues for or against each with evidence. Keep the reasoning visible in the output.

**Outputs:**
- a typed citation graph (networkx → interactive HTML via pyvis)
- the matrix heatmap
- a contradictions table

### Stage 5: orchestration (CrewAI Flow)

Build the pipeline as a `Flow` with these steps:
1. `collect_corpus` (OpenAlex + Semantic Scholar snowballing from seeds; rank by relevance and citation velocity)
2. **human approval checkpoint** of the corpus
3. `extract` (parallel per paper, cached)
4. `verify_claims`
5. `classify_relations`
6. `build_map`
7. `find_gaps`
8. `write_review`
9. `verify_review` → revise loop
10. `export`: Markdown review with inline citations, JSON of all records, graph HTML, gap and contradiction tables

Agents: Librarian, Extractor, Verifier, Synthesist/Adjudicator, Gap Analyst, Writer. Keep roles distinct and give each agent only the tools it needs.

### Stage 6: distilled extractor (component B, fine-tuned)

- Run the API-model Extractor over the training-split papers.
- Keep only extractions where every claim passed the Verifier. **The Verifier filters the training data.**
- QLoRA fine-tune a small instruct model (e.g. Qwen2.5-1.5B/3B-Instruct or Llama-3.2-3B-Instruct; check current availability) with Unsloth or PEFT on a Colab T4.
- Output must be valid JSON matching `PaperRecord`.
- Compare against the API extractor on held-out papers: JSON validity rate, claim faithfulness, field-level agreement, latency and cost per paper.
- Make the extractor swappable in the Flow.
- Component C (fine-tuning a related-work generator) is a stretch goal. Only attempt it if everything else is finished, and gate its output through the Verifier.

### Stage 7: evaluation harness (#6)

**Benchmark:**
- `rw_pairs` test split, which is time-split and frozen; tune nothing on it
- Multi-XScience test, as a secondary benchmark
- Task: given only the cited papers, generate the related-work section

**Systems:**
- S1: single well-prompted LLM call
- S2: crew without Verifier
- S3: full crew
- S4: full crew with the distilled extractor
- Ablation: S3 without the relation classifier

**Metrics:**
- ROUGE-1/2/L and BERTScore against the human section
- citation coverage (fraction of cited papers discussed) and citation correctness (cited claim actually comes from that paper)
- faithfulness (NLI, as in Stage 3)
- contradiction/gap precision on a small hand-checked sample
- tokens, latency, cost

**Reporting:**
- means with bootstrap 95% CIs
- same LLM and settings across systems
- a qualitative side-by-side for 2–3 examples
- a short error analysis with failure categories and counts

Write everything to `results/eval_*.json` and render the tables from those files.

## 4. Order of work

1. Restructure into the package and move `latex_mining.py`. Add tests and confirm notebook 01 still runs.
2. Run stage 1 at pilot scale locally (e.g. 150 train / 50 test papers). Fix the parser and record yield.
3. Stage 2 (pilot labelling of 1k + the 200 gold contexts → SciBERT on MPS) **in parallel with** Stage 3 on ~20 papers.
4. Stage 4 → Stage 5 Flow end to end on one real topic. Suggestion: "faithfulness/hallucination evaluation in abstractive summarisation", which suits a self-referential demo.
5. Stage 7 on a pilot test set (~30 papers) for S1–S3.
6. Stage 6 (needs GPU: give the user the Colab notebook; locally, either try a tiny model on MPS or mark it as Colab-only and use the user's Colab results if they provide them).
7. Finalise all six notebooks for full-scale Colab runs. Each notebook starts with a markdown cell giving purpose, inputs, outputs, expected runtime on a T4, and required secrets.
8. Figures, then the report.

Before each long or expensive run, estimate its duration and API usage and tell the user. Stay within free-tier rate limits: add rate limiting and retry with backoff.

## 5. Report (`report/LitReview_Report.docx`)

Fill `Report_Template.docx` with python-docx, preserving its cover page, fonts, heading styles, table of contents, and list-of-tables / list-of-figures tables (update their entries and page numbers, or insert a Word TOC field and tell the user to update fields).

**Cover details:**
- Title: choose a crisp one, e.g. "Grounded Literature Review Generation with Multi-Agent LLMs and Citation-Relation Learning"
- Student: **Siddharth Ray, Reg. No. 235890416**
- Branch: SoCE – AI, MIT Bengaluru
- Leave `<Section>` and the other student-name/reg-no slots as clearly highlighted placeholders, and ask the user whether there are teammates
- Keep the course line, guide name and other template text unchanged unless the user says otherwise. **Point out to the user** that the template says "IV Semester" and "April 2026" and ask whether these should change.

**Structure (follow the template's chapters):**
- **Abstract:** ~250 words, with the key numbers.
- **Ch 1 Introduction:** 1.1 overview, 1.2 motivation, 1.3 problem statement and relevance, 1.4 objectives (numbered, measurable).
- **Ch 2 Literature review**, written as a genuine survey of real, verified work:
  - multi-document and scientific summarisation (e.g. Multi-XScience, related-work generation)
  - citation intent / function classification (SciCite, ACL-ARC, SciBERT)
  - faithfulness and hallucination evaluation (NLI-based checking, QAGS/SummaC-style work)
  - LLM agents and multi-agent frameworks (CrewAI, related frameworks)
  - LLM-assisted literature-review tools
  - 2.1 comparative table of approaches, 2.2 trends and gaps that motivate this project, 2.4 conclusion
  - ~20–30 references, all verified
- **Ch 3 Methodology:**
  - 3.1 architecture diagram, Flow diagram, data-mining pipeline diagram (render as PNG via graphviz/mermaid and embed), each component with equations where relevant (faithfulness score, κ, macro-F1, bootstrap CI)
  - 3.2 system requirements (Colab T4, libraries with versions, APIs, local dev setup)
- **Ch 4 Analysis and discussion:**
  - dataset statistics
  - classifier results with confusion matrix
  - Verifier calibration
  - main eval table S1–S4 and ablation, cost/latency table
  - example contradiction map and gap heatmap, citation graph screenshot
  - qualitative examples, error analysis
  - limitations and threats to validity (LLM-labelled data, pilot sample sizes, leakage controls, NLI model errors)
- **Ch 5:** 5.1 conclusion and 5.2 future directions. The template numbers this as 5.4; fix that to 5.2.
- **References:** in the template's format, grouped as journal/conference, books and web.

**Writing style:** neutral, precise, academic. No marketing language, no filler, no "AI-written" tics. Short paragraphs, concrete numbers.

**Checks before finishing:**
- Open the generated .docx (render to PDF via LibreOffice and look at the pages).
- Check figure/table numbering and that every figure is referenced in the text.
- Confirm every number matches `results/`.
- Confirm every reference resolves.
- List anything the user still needs to fill in or decide.

## 6. Deliverables

1. The repo with tests passing and a README (setup, how to run locally, how to run on Colab, results summary).
2. Six Colab notebooks, each run at pilot scale with outputs saved, plus a `RUN_ON_COLAB.md` giving the order, runtimes, secrets, and the Drive folder layout.
3. `results/` and `figures/` from the actual runs.
4. `report/LitReview_Report.docx` (and a PDF render).
5. A short final message saying what was run at what scale, which numbers are pilot-scale, what remains for the full Colab run, and the placeholders left in the report.
