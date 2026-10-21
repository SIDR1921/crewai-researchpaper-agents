#!/usr/bin/env python
"""Generate the six Colab notebooks from one spec.

The notebooks are thin: they mount Drive, install this repository, and call the
same package functions the local CLIs call. Keeping them generated from a single
spec is what stops the notebook and local code paths from drifting apart -- the
only thing a notebook contains that the package does not is its config cell and
its narrative.

    python scripts/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

REPO_URL = "https://github.com/REPLACE-ME/litreview.git"   # see the setup cell's note

SETUP = '''# --- Colab setup: Drive, project code, secrets. Safe to re-run. ---
import json, os, subprocess, sys, zipfile
from pathlib import Path
ON_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

if ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

ROOT = "/content/drive/MyDrive/litreview" if ON_COLAB else os.path.abspath("..")
os.makedirs(ROOT, exist_ok=True)
os.environ["LITREVIEW_ROOT"] = ROOT
REPO = os.path.join(ROOT, "repo")

# Optional: a git URL to clone the project from. Leave blank to use the folder or
# the zip on Drive instead.
REPO_URL = ""

def _find_project() -> str | None:
    """Locate the project code: an existing folder, a zip on Drive, or a clone."""
    if os.path.exists(os.path.join(REPO, "pyproject.toml")):
        return REPO
    # a litreview_*.zip uploaded anywhere under MyDrive/litreview
    zips = sorted(Path(ROOT).glob("litreview*.zip")) + sorted(Path(ROOT).glob("**/litreview*.zip"))
    for z in zips:
        print(f"extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ROOT)
        # the zip may contain a top-level folder; find the pyproject inside
        for cand in [REPO, ROOT] + [str(p.parent) for p in Path(ROOT).glob("*/pyproject.toml")]:
            if os.path.exists(os.path.join(cand, "pyproject.toml")):
                return cand
    if REPO_URL:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=False)
        if os.path.exists(os.path.join(REPO, "pyproject.toml")):
            return REPO
    return None

if ON_COLAB:
    found = _find_project()
    if not found:
        raise SystemExit(
            "Could not find the project code. Do one of these, then re-run this cell:\n"
            "  1. upload litreview_colab.zip into MyDrive/litreview/ (easiest), or\n"
            "  2. copy the project folder to MyDrive/litreview/repo, or\n"
            "  3. set REPO_URL above to a git URL.\n"
            "The notebooks deliberately contain no second copy of the code.")
    REPO = found
    print("project code:", REPO)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-e", f"{REPO}[ml,eval]"],
                   check=True)

import litreview
from litreview.config import SETTINGS, paths, secret
P = paths().mkdirs()
print("data root:", P.root)
print("already present:", {f.name: f.stat().st_size // 1024 for f in sorted(P.data.glob("*.jsonl"))})
'''

CONFIG_HEADER = "# ======================= CONFIG (the only cell to edit) ======================="

PROVIDER_CELL = '''# Provider comes from one config. Keys are read from Colab secrets, never hard-coded.
import os
os.environ["LITREVIEW_PROVIDER"] = PROVIDER
if MODEL:
    os.environ["LITREVIEW_MODEL"] = MODEL

from litreview.config import Settings
from litreview.llm import install_call_logger, probe, usage_summary
import litreview.config as _cfg
_cfg.SETTINGS = Settings()                      # re-read after setting the env vars
import litreview.llm as _llm
_llm.SETTINGS = _cfg.SETTINGS

install_call_logger(P.results / "llm_calls.jsonl")
print(json.dumps(_cfg.SETTINGS.describe(), indent=1))
print(json.dumps(probe(_cfg.SETTINGS), indent=1))
'''


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {"colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }


def header(title: str, purpose: str, inputs: str, outputs: str, runtime: str,
           secrets: str, gpu: str) -> dict:
    return md(f"""# {title}

**Purpose.** {purpose}

| | |
|---|---|
| **Inputs** | {inputs} |
| **Outputs** | {outputs} |
| **Expected runtime (Colab T4)** | {runtime} |
| **Required secrets** | {secrets} |
| **GPU** | {gpu} |

Everything is cached under `/content/drive/MyDrive/litreview/` and every stage is
**resumable**: if Colab disconnects, re-run all cells and it continues where it
stopped. Only the cell marked CONFIG needs editing.
""")


# --------------------------------------------------------------------- 01

def nb01() -> dict:
    orig = json.loads(Path(__file__).with_name("nb01_markdown.json").read_text()) \
        if Path(__file__).with_name("nb01_markdown.json").exists() else None
    cells = [header(
        "01 · Mine related-work sections and citation contexts from arXiv",
        "Build the datasets every later stage depends on: one row per citing sentence x cited "
        "paper, and one row per paper pairing its human-written related-work section with the "
        "abstracts of the papers it cites. Also loads Multi-XScience as an external benchmark "
        "with the same schema.",
        "the arXiv API and OpenAlex (no local inputs)",
        "`data/citation_contexts.jsonl`, `data/rw_pairs.jsonl`, `data/openalex_cache.json`, "
        "`data/multixscience_test.jsonl`, `results/stage1_yield.json`, "
        "`results/stage1_dataset_stats.json`",
        "~45 min at pilot scale (150/50 papers); 2-6 h at full scale (3000/300)",
        "`OPENALEX_MAILTO`",
        "not needed")]
    cells.append(md("""### What the two datasets are for

| File | What it holds | Used by |
|---|---|---|
| `citation_contexts.jsonl` | one row per (citing paper, cited paper, sentence that cites it) | the relation classifier (component A, notebook 02) |
| `rw_pairs.jsonl` | a paper's related-work section + the titles/abstracts of the papers it cites | the evaluation benchmark (notebook 06) and the distillation data (05) |
| `openalex_cache.json` | cited title -> OpenAlex metadata | everything |

**Time split.** Train papers come from an older window, test papers from a recent
one. Test papers must be newer than the knowledge cutoff of every model being
evaluated, otherwise a model may have memorised the real related-work section.
The dates used and why are in `docs/leakage.md`.
"""))
    cells.append(code(SETUP))
    cells.append(md("""### Config

Start with a small pilot (a few hundred papers) to check yield, then scale up.
Roughly half of arXiv submissions have usable LaTeX with an identifiable
related-work section; the exact figures for this run are printed below."""))
    cells.append(code(f'''{CONFIG_HEADER}
CATEGORIES   = ["cs.CL", "cs.IR"]                 # NLP + IR: closest to the project's domain
TRAIN_WINDOW = ("202301010000", "202412312359")
TEST_WINDOW  = ("202606010000", "202609302359")   # after the evaluated models' cutoffs
N_TRAIN, N_TEST = 150, 50                         # pilot; scale to ~3000 / 300
N_MULTIXSCIENCE = 300                             # external benchmark examples (0 to skip)
SEED = 0
# ==============================================================================
'''))
    cells.append(md("""### The parsing library

The LaTeX miner lives in `litreview.mining.latex_mining` and is unit-tested
offline (`tests/test_latex_mining.py`). It finds the main `.tex` among several
candidates, inlines `\\input{}`, strips comments, locates the related-work section
whether it is a `\\section`, a `\\subsection` or only a run of `\\paragraph` blocks,
converts LaTeX to text while keeping citations as `[CITE:key]` markers, and maps
cite keys to titles from `.bbl` (both `\\bibitem` and `biblatex` formats), `.bib`,
or a `thebibliography` block inlined in the main file."""))
    cells.append(code('''from litreview.mining import latex_mining as lm

# the same self-test the unit tests run, as a smoke check that the install worked
_t = lm.mine_paper({
    "main.tex": r"\\documentclass{article}\\begin{document}\\section{Related Work}"
                r"Pointer networks were used by \\citet{see2017}. In contrast, \\citep{liu2019} "
                r"fine-tune BERT encoders for summarization.\\section{Method}x\\end{document}",
    "refs.bib": "@article{see2017,\\n title = {Get to the point},\\n}\\n",
})
assert _t["related_work"]["cited_keys"] == ["liu2019", "see2017"], _t
print("parser OK:", _t["related_work"]["text"])
'''))
    cells.append(md("""### Mine

`run_stage1` lists both splits from the arXiv API, downloads and caches every
LaTeX source, parses each paper, resolves cited titles against OpenAlex and writes
the two datasets. Each of those steps is cached, so this cell is safe to
re-run.

arXiv asks for at least three seconds between requests, which is enforced inside
the package. At pilot scale expect roughly 20 minutes of downloading and 20
minutes of OpenAlex resolution."""))
    cells.append(code('''from tqdm.auto import tqdm
from litreview.mining.pipeline import run_stage1

report = run_stage1(
    N_TRAIN, N_TEST, CATEGORIES, TRAIN_WINDOW, TEST_WINDOW, seed=SEED,
    fetch_progress=lambda ps, s: tqdm(ps, desc=f"fetch {s}"),
    progress=lambda ps, s: tqdm(ps, desc=f"parse {s}"),
    oa_progress=lambda ts: tqdm(ts, desc="openalex"),
)
print(json.dumps(report, indent=2))
'''))
    cells.append(md("""### Multi-XScience

The same task from a published dataset (Lu, Dong and Charlin, EMNLP 2020),
converted into the `rw_pairs` schema with the same filters applied. It is a check
that the evaluation numbers are not an artefact of our own mining."""))
    cells.append(code('''from litreview.mining import multixscience

if N_MULTIXSCIENCE:
    mx = multixscience.build(P.data, split="test", limit=N_MULTIXSCIENCE)
    print(json.dumps(mx, indent=2))
    report["multixscience"] = mx
    (P.results / "stage1_yield.json").write_text(json.dumps(report, indent=2))
'''))
    cells.append(md("""### Statistics and sanity checks

Read a few samples by eye before moving on: bad parsing here silently poisons
every later stage. The time-split check is recorded to `results/` rather than
asserted, so the report can quote the actual dates."""))
    cells.append(code('''import subprocess, sys
subprocess.run([sys.executable, (Path(REPO) / "scripts" / "dataset_stats.py").as_posix()
                if ON_COLAB else "../scripts/dataset_stats.py"], check=False)

import pandas as pd
ctx = pd.read_json(P.data / "citation_contexts.jsonl", lines=True)
rw  = pd.read_json(P.data / "rw_pairs.jsonl", lines=True)
print(ctx.groupby("split").agg(rows=("cite_key", "size"), papers=("citing_id", "nunique"),
                               in_rw=("in_related_work", "mean")).round(2))
rw["n_cited"]  = rw["cited"].str.len()
rw["rw_words"] = rw["rw_text"].str.split().str.len()
print(rw.groupby("split")[["n_cited", "rw_words", "abstract_coverage"]].median())

for _, r in ctx.sample(min(3, len(ctx)), random_state=1).iterrows():
    print("\\n---", r.cited_title, "\\n", r.context[:400])
'''))
    cells.append(code('''# Leakage check: the newest train paper must predate the oldest test paper.
ts = json.loads((P.results / "stage1_dataset_stats.json").read_text())["time_split"]
print(json.dumps(ts, indent=1))
assert ts["disjoint"], "time split overlap -- do not evaluate on this benchmark"
'''))
    cells.append(md("""### Next

**02_relation_labels_and_scibert.ipynb** — define the relation taxonomy, label a
few thousand contexts with the configured LLM, hand-check 200 for agreement, then
fine-tune SciBERT (component A).

`rw_pairs.jsonl` (test split) is the frozen evaluation benchmark. Nothing is tuned
on it."""))
    return notebook(cells)


# --------------------------------------------------------------------- 02

def nb02() -> dict:
    cells = [header(
        "02 · Citation-relation labels and the fine-tuned SciBERT classifier (component A)",
        "Label mined citation contexts with the configured LLM against a seven-class relation "
        "taxonomy, measure the LLM against an adjudicated gold set, then fine-tune SciBERT and compare "
        "it with a majority-class, a TF-IDF+LR and a zero-shot LLM baseline.",
        "`data/citation_contexts.jsonl` (notebook 01)",
        "`data/llm_labels.jsonl`, `data/gold_blind.jsonl`, `models/scibert_relations/`, "
        "`results/stage2_labelling.json`, `results/stage2_scibert.json`, "
        "`results/stage2_benchmark.json`, `results/stage2_gold_agreement.json`",
        "labelling 1-3 h (API-bound); SciBERT ~15 min; baselines ~10 min",
        "`GROQ_API_KEY` (or `GEMINI_API_KEY`)",
        "T4 recommended for SciBERT and the zero-shot baseline")]
    cells.append(code(SETUP))
    cells.append(code(f'''{CONFIG_HEADER}
PROVIDER   = "groq"          # groq | gemini | ollama
MODEL      = ""              # "" uses the provider's default from litreview/config.py
N_LABEL    = 4000            # contexts to label this session (append-only, resumable)
BATCH_SIZE = 8               # contexts per request: 8x saving on the request budget
IN_RW_ONLY = True            # label citations inside related-work sections only
N_GOLD     = 200             # size of the adjudicated gold set
EPOCHS, LR, TRAIN_BATCH = 4, 2e-5, 16
USE_SEED_CORPORA = True      # add SciCite + ACL-ARC rows, mapped onto our taxonomy
SEED = 0
# ==============================================================================
'''))
    cells.append(code(PROVIDER_CELL))
    cells.append(md("""### The taxonomy

Seven labels with one definition each and an ordered set of tie-break rules. The
definitions live in `litreview.relations.taxonomy` so the LLM prompt, the human
labelling guide (`docs/labelling_guide.md`) and the training code cannot disagree
about what a label means."""))
    cells.append(code('''from litreview.relations.taxonomy import (ACL_ARC_LIMITS, ACL_ARC_MAP, DEFINITIONS,
                                           DECISION_RULES, SCICITE_LIMITS, SCICITE_MAP)
for k, v in DEFINITIONS.items():
    print(f"{k:18s} {v}")
print("\\nTie-break order:")
for i, r in enumerate(DECISION_RULES, 1):
    print(f"  {i}. {r}")
print("\\nSciCite mapping:", SCICITE_MAP, "\\n  limits:", SCICITE_LIMITS)
print("\\nACL-ARC mapping:", ACL_ARC_MAP, "\\n  limits:", ACL_ARC_LIMITS)
'''))
    cells.append(md("""### Label with the LLM

Append-only and keyed by a hash of (citing paper, cite key, sentence), so a daily
rate limit is not a failure: re-run this cell tomorrow and it continues. Each
request carries `BATCH_SIZE` contexts and asks for a label, a short rationale that
must quote the cue words, and a confidence."""))
    cells.append(code('''import random
from tqdm.auto import tqdm
from litreview.relations.label import label_contexts, load_labels
from litreview.relations.taxonomy import LABELS

rows = []
for line in (P.data / "citation_contexts.jsonl").open():
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if r.get("split") != "train":
        continue
    if IN_RW_ONLY and not r.get("in_related_work"):
        continue
    if len(r.get("cite_sentence") or "") < 40:
        continue
    rows.append(r)
random.Random(SEED).shuffle(rows)
print(f"{len(rows)} candidate contexts")

stats = label_contexts(rows, P.data / "llm_labels.jsonl", batch_size=BATCH_SIZE,
                       limit=N_LABEL, progress=lambda b: tqdm(b, desc="label"))
labelled = load_labels(P.data / "llm_labels.jsonl")
rep = {"stats": stats, "n_labelled_total": len(labelled),
       "label_distribution": {l: sum(1 for r in labelled if r["llm_label"] == l) for l in LABELS},
       "provider": _cfg.SETTINGS.describe(), "usage": usage_summary()}
(P.results / "stage2_labelling.json").write_text(json.dumps(rep, indent=2))
print(json.dumps({k: v for k, v in rep.items() if k != "provider"}, indent=2))
'''))
    cells.append(md("""### The adjudicated gold set

200 contexts, stratified over the LLM's predicted label so the rare relations are
represented at all. The sampler writes a **blind** file with no LLM label in it;
that is what keeps the gold labels independent of the labeller. Label `gold_blind.jsonl` by hand
(fill in `human_label`), save it as `gold_labelled.jsonl`, then run the scoring
cell.

Because the sample is stratified, the resulting agreement is **not** an estimate of
accuracy on a uniformly random citation context, and the report says so."""))
    cells.append(code('''import subprocess, sys
SCRIPTS = (Path(REPO) / "scripts") if ON_COLAB else Path("../scripts")
subprocess.run([sys.executable, str(SCRIPTS / "sample_gold.py"), "--n", str(N_GOLD)], check=False)
print("\\nNow hand-label", P.data / "gold_blind.jsonl",
      "\\n -> save as", P.data / "gold_labelled.jsonl", "with human_label filled in.")
'''))
    cells.append(code('''# Run once gold_labelled.jsonl exists.
if (P.data / "gold_labelled.jsonl").exists():
    subprocess.run([sys.executable, str(SCRIPTS / "score_gold.py")], check=False)
else:
    print("gold_labelled.jsonl not present yet; skipping agreement scoring")
'''))
    cells.append(md("""### Fine-tune SciBERT

Input: the citation context with the target citation marked `[[CITED]]` and any
other citations in the sentence collapsed to `[OTHER]`, paired with the cited
paper's title as a second segment. Class weights are inverse-frequency because
`background` dominates. The dev split is taken **by citing paper**, so no paper
appears on both sides. Gold-set rows are excluded from training."""))
    cells.append(code('''from litreview.relations import seed_data
from litreview.relations.scibert import RelationClassifier, split_by_paper, train

rows = [{**r, "label": r["llm_label"]} for r in load_labels(P.data / "llm_labels.jsonl")]
gold = []
if (P.data / "gold_labelled.jsonl").exists():
    gold = [{**json.loads(l), "label": json.loads(l)["human_label"]}
            for l in (P.data / "gold_labelled.jsonl").open()
            if json.loads(l).get("human_label") in LABELS]
gold_ids = {r["context_id"] for r in gold}
rows = [r for r in rows if r["context_id"] not in gold_ids]

extra = {}
if USE_SEED_CORPORA:
    sc = seed_data.load_scicite(P.cache, limit=2000)
    aa = seed_data.load_acl_arc(P.cache, limit=2000)
    extra = {"scicite": {k: v for k, v in sc.items() if k != "rows"},
             "acl_arc": {k: v for k, v in aa.items() if k != "rows"}}
    rows += sc.get("rows", []) + aa.get("rows", [])
    print(json.dumps(extra, indent=1))

tr, dev = split_by_paper(rows, frac_dev=0.15, seed=SEED)
print(f"train {len(tr)} rows / dev {len(dev)} rows")
report = train(tr, P.models / "scibert_relations", epochs=EPOCHS, lr=LR,
               batch_size=TRAIN_BATCH, dev_rows=dev, seed=SEED,
               progress=lambda dl, d: tqdm(dl, desc=d))
report["seed_corpora"] = extra
if gold:
    report["gold_eval"] = RelationClassifier(P.models / "scibert_relations").evaluate(gold)
    print("gold macro-F1", report["gold_eval"]["macro_f1"],
          "accuracy", report["gold_eval"]["accuracy"])
(P.results / "stage2_scibert.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items()
                  if k not in ("gold_eval", "seed_corpora")}, indent=2))
'''))
    cells.append(md("""### Baselines

Majority class, TF-IDF + logistic regression, and a zero-shot LLM using the same
prompt as the labeller. All four systems are scored on the same adjudicated gold set."""))
    cells.append(code('''subprocess.run([sys.executable, str(SCRIPTS / "run_stage2.py"), "bench"], check=False)
if (P.results / "stage2_benchmark.json").exists():
    b = json.loads((P.results / "stage2_benchmark.json").read_text())
    for k in ("majority", "tfidf_lr", "zero_shot_llm", "scibert"):
        if k in b:
            print(f"{k:16s} acc={b[k]['accuracy']:.3f}  macroF1={b[k]['macro_f1']:.3f}")
'''))
    cells.append(code('''# Expose the classifier the later notebooks use.
from litreview.relations.scibert import RelationClassifier
clf = RelationClassifier(P.models / "scibert_relations")
print(clf.classify("Unlike [[CITED]], which assumes a single reference summary, we ...",
                   "Evaluating the factual consistency of abstractive text summarization"))
print(json.dumps(usage_summary()["overall"], indent=1))
'''))
    cells.append(md("""### Next

**03_crew_extract_verify.ipynb** — the Extractor agent and the NLI-based Verifier,
whose thresholds are fitted on a hand-checked dev set."""))
    return notebook(cells)


# --------------------------------------------------------------------- 03

def nb03() -> dict:
    cells = [header(
        "03 · Extraction and verification (the Extractor agent and the NLI Verifier)",
        "Turn papers into `PaperRecord`s whose every claim quotes its evidence verbatim, then "
        "verify each claim twice: a fuzzy check that the quote really occurs in the source, and "
        "an NLI entailment check that the claim follows from the quote. The two thresholds are "
        "fitted on a hand-checked dev set, not guessed.",
        "`data/papers_train.jsonl` + `data/sources/` (notebook 01)",
        "`data/extractions_*.jsonl`, `data/claim_verdicts_*.json`, "
        "`results/verifier_calibration.json`, `results/stage3_*.json`",
        "1-3 h for 100-300 papers (API-bound); the NLI model loads in ~1 min",
        "`GROQ_API_KEY` (or `GEMINI_API_KEY`)",
        "T4 recommended (the DeBERTa NLI model runs ~10x faster than on CPU)")]
    cells.append(code(SETUP))
    cells.append(code(f'''{CONFIG_HEADER}
PROVIDER  = "groq"
MODEL     = ""
N_PAPERS  = 100              # papers to extract this session (resumable)
SPLIT     = "train"
BACKEND   = "api"            # "api" = direct LLM call; "crew" = CrewAI agent + output_pydantic
N_DEV     = 60               # claim/evidence pairs to hand-check for calibration
SEED      = 0
# ==============================================================================
'''))
    cells.append(code(PROVIDER_CELL))
    cells.append(md("""### The schemas

`PaperRecord` and `ReviewSentence` are the contract between every stage, and they
are what the distilled model in notebook 05 has to emit."""))
    cells.append(code('''from litreview.schemas import PaperRecord, ReviewSentence
print(json.dumps(PaperRecord.model_json_schema()["properties"], indent=1)[:1500])
'''))
    cells.append(md("""### Extract

The Extractor's input is the abstract plus whatever introduction, results,
conclusion and limitations text the LaTeX source gave us, under a character
budget. The exact input string is cached next to the record, because the
Verifier's quote check is defined against *that* string.

`BACKEND = "crew"` runs the same prompt as a CrewAI agent with
`output_pydantic=PaperRecord`; `"api"` calls the model directly and is what the
batch runs use."""))
    cells.append(code('''import sys, subprocess
from tqdm.auto import tqdm
from litreview.extract import ApiExtractor, CrewExtractor, extract_corpus, load_records
SCRIPTS = (Path(REPO) / "scripts") if ON_COLAB else Path("../scripts")
sys.path.insert(0, str(SCRIPTS))
from run_stage3 import papers_from_mining       # same helper the CLI uses

papers = papers_from_mining(N_PAPERS, SPLIT, SEED)
print(f"{len(papers)} papers with usable sections")

ex = CrewExtractor() if BACKEND == "crew" else ApiExtractor()
cache = P.data / f"extractions_{SPLIT}_{ex.name}.jsonl"
stats = extract_corpus(papers, cache, extractor=ex,
                       progress=lambda ps: tqdm(ps, desc="extract"))
stats |= {"backend": ex.name, "cache": str(cache), "usage": usage_summary()}
(P.results / f"stage3_extract_{SPLIT}_{ex.name}.json").write_text(json.dumps(stats, indent=2))
print(json.dumps({k: v for k, v in stats.items() if k != "usage"}, indent=2))
'''))
    cells.append(code('''records = load_records(cache)
pid = next(iter(records))
rec = records[pid]["record"]
print(rec["title"][:90], "\\n  task:", rec.get("task"))
for c in rec["claims"][:3]:
    print(f"  [{c['claim_id']}] {c['text']}\\n      evidence ({c['evidence']['section']}): "
          f"\\"{c['evidence']['passage'][:160]}\\"")
'''))
    cells.append(md("""### Calibrate the Verifier

Sample claim/evidence pairs, label each `supported`, `partially` or `unsupported`
by hand, then fit the quote and entailment thresholds by grid search on macro-F1
over the three verdicts. Macro-F1 rather than accuracy, so a threshold pair is not
chosen for doing well on the majority verdict alone."""))
    cells.append(code('''subprocess.run([sys.executable, str(SCRIPTS / "run_stage3.py"), "make-dev",
                "--n", str(N_DEV), "--split", SPLIT], check=False)
print("\\nHand-label", P.data / "verifier_dev.jsonl",
      '- set "human" to supported | partially | unsupported for each item.')
'''))
    cells.append(code('''# Run once verifier_dev.jsonl is labelled. Until then the defaults in
# litreview.verify.DEFAULT_THRESHOLDS are used, and the report says they are defaults.
if (P.data / "verifier_dev.jsonl").exists():
    subprocess.run([sys.executable, str(SCRIPTS / "run_stage3.py"), "calibrate"], check=False)
if (P.results / "verifier_calibration.json").exists():
    cal = json.loads((P.results / "verifier_calibration.json").read_text())
    print(json.dumps({k: v for k, v in cal.items() if k not in ("grid", "per_item")}, indent=2))
'''))
    cells.append(md("""### Verify every claim

Two independent checks per claim. A claim can quote correctly and still overstate
what the passage says; the entailment check is what catches that."""))
    cells.append(code('''from litreview.verify import Verifier
v = Verifier()
print("thresholds:", v.th, "\\nNLI model:", v.nli.model_name, "on", v.nli.device)

rep = v.verify_corpus(records, P.data / f"claim_verdicts_{SPLIT}_{ex.name}.json")
(P.results / f"stage3_verify_{SPLIT}_{ex.name}.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
'''))
    cells.append(md("""### Next

**04_synthesis_and_gaps.ipynb** — the consensus/contradiction map, gap detection
and the full CrewAI Flow on one real topic."""))
    return notebook(cells)


# --------------------------------------------------------------------- 04

def nb04() -> dict:
    cells = [header(
        "04 · Consensus map, contradictions, gaps, and the full CrewAI Flow",
        "Normalise dataset/metric/method names, group reported numbers by (task, dataset, "
        "metric), flag disagreements and have the Adjudicator explain them from the evidence "
        "alone, build the method-family x dataset matrix and have the Gap Analyst argue both "
        "sides of each candidate gap. Then run the whole Flow end to end on one topic, including "
        "the human approval checkpoint.",
        "`models/scibert_relations/` (02), `data/extractions_*.jsonl` (03), OpenAlex",
        "`results/flow_runs/<topic>/` (review.md, records.json, citation_graph.html, "
        "contradictions.csv, gaps.csv, summary.json), `results/flow_summary.json`",
        "20-50 min",
        "`GROQ_API_KEY`, `OPENALEX_MAILTO`",
        "T4 recommended (NLI + SciBERT inference)")]
    cells.append(code(SETUP))
    cells.append(code(f'''{CONFIG_HEADER}
PROVIDER = "groq"
MODEL    = ""
TOPIC    = "faithfulness and hallucination evaluation in abstractive summarisation"
N_PAPERS = 15                # corpus size for the demonstration run
SEEDS    = []                # optional seed paper titles to snowball from
AUTO_APPROVE = False         # False shows the corpus table and asks
# ==============================================================================
'''))
    cells.append(code(PROVIDER_CELL))
    cells.append(md("""### Normalisation

Three stages, cheapest first: the manual alias table checked into the repo
(`litreview/resources/aliases.json`), then fuzzy matching, then sentence
embeddings for the paraphrases fuzzy matching misses. Anything still unmatched
keeps a tidied version of its original surface form, so nothing is silently
dropped -- and the unmatched names are reported, which is how the alias file gets
extended."""))
    cells.append(code('''from litreview.normalise import Normaliser
n = Normaliser(use_embeddings=False)
for s, kind in [("cnn/dm", "datasets"), ("RougeL", "metrics"), ("bart-large", "methods"),
                ("hallucination detection", "tasks"), ("Zorblax benchmark", "datasets")]:
    print(f"{s:28s} {kind:9s} -> {n.normalise(s, kind)}")
print("\\nmethod families:", {m: n.method_family(m) for m in ["bart", "scibert", "gpt-4", "crewai"]})
'''))
    cells.append(md("""### The Flow

Ten steps: `collect_corpus` → **human approval** → `extract` → `verify_claims` →
`classify_relations` → `build_map` → `find_gaps` → `write_review` →
`verify_review` (one revision pass) → `export`. Everything is cached under the run
directory, so a Flow that dies mid-way continues on the next call.

`as_crewai_flow()` wires the same steps into a real `crewai.flow.Flow` with
`@start` / `@listen`; `run()` calls them directly, which is what the tests use."""))
    cells.append(code('''import pandas as pd
from tqdm.auto import tqdm
from litreview.flow import LitReviewFlow
from litreview.relations.scibert import RelationClassifier

clf = None
if (P.models / "scibert_relations").exists():
    clf = RelationClassifier(P.models / "scibert_relations")
else:
    print("no classifier found; relations will be skipped (run notebook 02 first)")

def approve(table):
    display(pd.DataFrame(table))
    if AUTO_APPROVE:
        return True
    return (input("approve this corpus? [Y/n] ").strip().lower() or "y").startswith("y")

flow = LitReviewFlow(topic=TOPIC, n_papers=N_PAPERS, seeds=SEEDS, approve=approve,
                     classifier=clf, progress=lambda xs: tqdm(xs, desc="step"))
summary = flow.as_crewai_flow().kickoff()
if not isinstance(summary, dict):
    summary = flow.state.exports
summary = dict(summary) | {"usage": usage_summary()}
(P.results / "flow_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "usage"}, indent=2))
'''))
    cells.append(code('''from IPython.display import HTML, Markdown, display
print(flow.state.exports)
display(Markdown(flow.state.review.get("markdown", "(no review)")))
'''))
    cells.append(code('''cons = pd.DataFrame(flow.state.contradictions)
display(cons.head(10) if len(cons) else "no contradictions flagged")
gaps = pd.DataFrame(flow.state.gaps)
display(gaps[["kind", "method_family", "dataset", "verdict", "score",
              "argument_for", "argument_against"]].head(10) if len(gaps) else "no gaps")
'''))
    cells.append(code('''# the typed citation graph, inline
p = Path(flow.state.exports.get("graph_html", ""))
display(HTML(p.read_text()[:400_000]) if p.exists() else HTML("<i>no graph written</i>"))
'''))
    cells.append(md("""### Next

**05_distill_extractor_qlora.ipynb** — distil the API Extractor into a small local
model, using the Verifier as the data filter. That notebook **requires** a GPU."""))
    return notebook(cells)


# --------------------------------------------------------------------- 05

def nb05() -> dict:
    cells = [header(
        "05 · Distil the Extractor into a small local model with QLoRA (component B)",
        "Build a training set from the API Extractor's output on training-split papers, keeping "
        "only extractions where EVERY claim passed the Verifier -- the Verifier is the data "
        "filter, so the student never learns to hallucinate evidence the teacher got away with. "
        "Then QLoRA fine-tune a small instruct model and compare it with the API extractor on "
        "held-out papers.",
        "`data/extractions_train_api.jsonl` and `data/claim_verdicts_train_api.json` (notebook 03)",
        "`data/distill_train.jsonl`, `models/distilled_extractor/`, `results/stage6_distill.json`",
        "data build ~5 min; QLoRA 1.5-3 h for ~400 examples on a T4; comparison ~20 min",
        "`GROQ_API_KEY` (for the API side of the comparison), `HF_TOKEN` if the base model is gated",
        "**T4 required.** 4-bit QLoRA needs bitsandbytes, which is CUDA-only. On CPU/MPS the "
        "trainer falls back to fp32 LoRA and reports `quantised_4bit: false` rather than "
        "pretending it quantised.")]
    cells.append(code(SETUP))
    cells.append(code(f'''{CONFIG_HEADER}
PROVIDER   = "groq"
MODEL      = ""
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # check availability; 3B also fits a T4 at max_len 2048
SPLIT      = "train"
EPOCHS, LR, MAX_LEN = 2, 2e-4, 3072
LORA_R, LORA_ALPHA  = 16, 32
GRAD_ACCUM = 8
ALLOW_PARTIAL = False        # True keeps "partially supported" claims too (more data, looser)
N_COMPARE  = 20              # held-out papers for the head-to-head
SEED = 0
# ==============================================================================
'''))
    cells.append(code('''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "bitsandbytes", "trl"], check=False)   # CUDA-only; harmless elsewhere
import torch
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''))
    cells.append(code(PROVIDER_CELL))
    cells.append(md("""### Build the Verifier-filtered training set

Only papers whose every claim came back `supported` survive. The report records
how many were dropped and why, because that number *is* the stage's argument."""))
    cells.append(code('''from litreview.distill import build_dataset
from litreview.extract import load_records

records = load_records(P.data / f"extractions_{SPLIT}_api.jsonl")
vpath = P.data / f"claim_verdicts_{SPLIT}_api.json"
verdicts = json.loads(vpath.read_text())["per_paper"] if vpath.exists() else {}
if not verdicts:
    raise SystemExit("no claim verdicts found; run notebook 03's verify cell first")

stats = build_dataset(records, verdicts, P.data / "distill_train.jsonl",
                      require_all_supported=True, allow_partial=ALLOW_PARTIAL, seed=SEED)
print(json.dumps(stats, indent=2))
'''))
    cells.append(code('''rows = [json.loads(l) for l in (P.data / "distill_train.jsonl").open()]
print(f"{len(rows)} training examples")
print("--- one example's target ---")
print(rows[0]["messages"][2]["content"][:900])
'''))
    cells.append(md("""### QLoRA fine-tune

4-bit NF4 base with LoRA adapters on the attention and MLP projections, gradient
checkpointing, paged AdamW. Save the adapter to Drive so the evaluation notebook
can load it."""))
    cells.append(code('''from litreview.distill import train_qlora
report = train_qlora(P.data / "distill_train.jsonl", P.models / "distilled_extractor",
                     base_model=BASE_MODEL, epochs=EPOCHS, lr=LR, max_len=MAX_LEN,
                     lora_r=LORA_R, lora_alpha=LORA_ALPHA, grad_accum=GRAD_ACCUM, seed=SEED)
print(json.dumps(report, indent=2))
'''))
    cells.append(md("""### Head to head on held-out papers

JSON validity rate, claim faithfulness through the same Verifier, field-level
agreement, latency and cost per paper. The distilled model runs locally, so its
monetary cost is zero and its latency is hardware-bound; the API column's cost
comes from the central usage log."""))
    cells.append(code('''import sys
from litreview.distill import compare_extractors
from litreview.extract import ApiExtractor, DistilledExtractor
from litreview.verify import Verifier
SCRIPTS = (Path(REPO) / "scripts") if ON_COLAB else Path("../scripts")
sys.path.insert(0, str(SCRIPTS))
from run_stage3 import papers_from_mining

held_out = papers_from_mining(N_COMPARE, "test", SEED)      # test split: never trained on
print(f"{len(held_out)} held-out papers")
cmp = compare_extractors(held_out, ApiExtractor(),
                         DistilledExtractor(P.models / "distilled_extractor"),
                         verifier=Verifier(), out_path=P.results / "stage6_distill.json")
cmp["training"] = report
cmp["dataset"] = stats
(P.results / "stage6_distill.json").write_text(json.dumps(cmp, indent=2))
print(json.dumps({k: v for k, v in cmp.items() if k != "llm_usage"}, indent=2))
'''))
    cells.append(md("""### Next

**06_evaluation.ipynb** — pass `--distilled models/distilled_extractor` so system
S4 is included in the comparison."""))
    return notebook(cells)


# --------------------------------------------------------------------- 06

def nb06() -> dict:
    cells = [header(
        "06 · Evaluation harness (S1-S4 and the ablation)",
        "Given only the papers a real related-work section cites, regenerate that section and "
        "score it: ROUGE and BERTScore against the human text, plus citation coverage, citation "
        "correctness and NLI faithfulness, with bootstrap 95% CIs. Same model and settings "
        "across every system.",
        "`data/rw_pairs.jsonl` test split (frozen), `data/multixscience_test.jsonl`, "
        "`models/scibert_relations/`, optionally `models/distilled_extractor/`",
        "`data/eval_generations_*.jsonl`, `results/eval_arxiv.json`, "
        "`results/eval_multixscience.json`, `figures/*.png`",
        "2-5 h for S1-S4 over 100 examples (API-bound); scoring ~20 min",
        "`GROQ_API_KEY`",
        "T4 recommended (NLI, BERTScore, SciBERT)")]
    cells.append(code(SETUP))
    cells.append(code(f'''{CONFIG_HEADER}
PROVIDER  = "groq"
MODEL     = ""
SYSTEMS   = ["S1", "S2", "S3", "S3_no_relations"]   # add "S4" once notebook 05 has run
LIMIT     = 100              # test examples; the whole split if smaller
BENCHMARK = "arxiv"          # "arxiv" | "multixscience"
DISTILLED = ""               # e.g. str(P.models / "distilled_extractor") to enable S4
BERTSCORE = True
# ==============================================================================
'''))
    cells.append(code(PROVIDER_CELL))
    cells.append(md("""### The benchmark is frozen

The test split comes from a submission window that is later than the training
window **and** later than the knowledge cutoff of every model being evaluated
(`docs/leakage.md`). Nothing in this project is tuned on it: the Verifier's
thresholds, the classifier's dev split and the gold label set all come from the
train split."""))
    cells.append(code('''from litreview.eval.runner import load_benchmark
bench_path = (P.data / "rw_pairs.jsonl" if BENCHMARK == "arxiv"
              else P.data / f"multixscience_test.jsonl")
bench = load_benchmark(bench_path, split="test", limit=LIMIT)
print(f"{len(bench)} examples from {bench_path.name}")
import pandas as pd
print(pd.DataFrame([{"arxiv_id": b["arxiv_id"], "n_cited": len(b["cited"]),
                     "rw_words": len(b["rw_text"].split()),
                     "published": b.get("published")} for b in bench]).describe().round(1))
'''))
    cells.append(md("""### The systems

| | |
|---|---|
| **S1** | one well-prompted LLM call over the same cited abstracts |
| **S2** | crew without the Verifier: extract, then write |
| **S3** | full crew: extract → verify → relations → map → gaps → write → verify/revise |
| **S4** | S3 with the distilled extractor from notebook 05 |
| **S3_no_relations** | ablation: S3 with the relation classifier switched off |

Generation is cached per (system, example), so a run stopped by a rate limit
resumes. Scoring is a separate pass over that cache, so metrics can be recomputed
without regenerating any text."""))
    cells.append(code('''from tqdm.auto import tqdm
from litreview.eval.runner import cost_table, generate, save, score, table
from litreview.relations.scibert import RelationClassifier
from litreview.verify import Verifier

verifier = Verifier()
classifier = (RelationClassifier(P.models / "scibert_relations")
              if (P.models / "scibert_relations").exists() else None)
distilled = None
if DISTILLED:
    from litreview.extract import DistilledExtractor
    distilled = DistilledExtractor(DISTILLED)

cache = P.data / f"eval_generations_{BENCHMARK}.jsonl"
gen = generate(bench, SYSTEMS, cache, verifier=verifier, classifier=classifier,
               distilled=distilled, progress=lambda js: tqdm(js, desc="generate"))
print(json.dumps(gen, indent=1))
'''))
    cells.append(code('''rep = score(bench, cache, nli=verifier.nli, with_bertscore=BERTSCORE,
            progress=lambda rs: tqdm(rs, desc="score"))
rep["benchmark"] = {"path": str(bench_path), "split": "test",
                    "n_examples": len(bench), "source": BENCHMARK}
rep["settings"] = _cfg.SETTINGS.describe()
rep["main_table"] = table(rep["aggregates"])
rep["cost_table"] = cost_table(rep["aggregates"], usage_summary())
save(rep, P.results / f"eval_{BENCHMARK}.json")

print("== main table ==")
display(pd.DataFrame(rep["main_table"]))
print("== cost / latency ==")
display(pd.DataFrame(rep["cost_table"]))
'''))
    cells.append(md("""### Qualitative side by side

Two or three examples, human section against each system's output. These are what
the error analysis in the report is built from."""))
    cells.append(code('''from IPython.display import Markdown, display
rows = [json.loads(l) for l in cache.open()]
by_id = {}
for r in rows:
    by_id.setdefault(r["arxiv_id"], {})[r["system"]] = r
for aid in list(by_id)[:3]:
    gold = next(b for b in bench if b["arxiv_id"] == aid)
    out = [f"### {aid} — {(gold.get('title') or '')[:100]}",
           f"**Human section ({len(gold['rw_text'].split())} words)**\\n\\n{gold['rw_text'][:1200]}"]
    for s in SYSTEMS:
        r = by_id[aid].get(s) or {}
        txt = (r.get("review") or {}).get("markdown") or f"_({r.get('error') or 'not run'})_"
        out.append(f"**{s}**\\n\\n{txt[:1200]}")
    display(Markdown("\\n\\n---\\n\\n".join(out)))
'''))
    cells.append(md("""### Figures

Every figure is generated from `results/*.json` by one script, so no figure can
show a number that is not in a results file."""))
    cells.append(code('''import subprocess, sys
SCRIPTS = (Path(REPO) / "scripts") if ON_COLAB else Path("../scripts")
subprocess.run([sys.executable, str(SCRIPTS / "usage_report.py")], check=False)
subprocess.run([sys.executable, str(SCRIPTS / "make_figures.py")], check=False)
print(json.dumps(json.loads((P.figures / "figures_manifest.json").read_text()), indent=1)[:2000])
'''))
    cells.append(md("""### Done

`results/` now holds every number the report quotes and `figures/` every figure it
embeds. Build the report with `python scripts/build_report.py`."""))
    return notebook(cells)


BUILDERS = {
    "01_mine_arxiv.ipynb": nb01,
    "02_relation_labels_and_scibert.ipynb": nb02,
    "03_crew_extract_verify.ipynb": nb03,
    "04_synthesis_and_gaps.ipynb": nb04,
    "05_distill_extractor_qlora.ipynb": nb05,
    "06_evaluation.ipynb": nb06,
}


def main() -> int:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILDERS.items():
        nb = fn()
        (NB_DIR / name).write_text(json.dumps(nb, indent=1) + "\n")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print(f"wrote {name}  ({len(nb['cells'])} cells, {n_code} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
