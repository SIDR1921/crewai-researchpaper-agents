# Running this project on Google Colab

Six notebooks, in order. All of them mount Google Drive, cache everything under
`/content/drive/MyDrive/litreview/`, and are **resumable**: if the session
disconnects, re-run all cells and each one continues from its cache. Only
notebooks 02 and 05 need a GPU.

The notebooks contain no copy of the project's logic. Each one's first cell
installs this package from the code you put on Drive and then calls it, so the
notebook and the local code can never drift apart.

## Step 1 · Make the bundle

On the machine that has this repository:

```bash
python scripts/make_colab_bundle.py
```

That writes `litreview_colab.zip` (~30 MB) next to the repository. It holds the
code and the data already mined on this machine, so a Colab run starts from that
data instead of spending two to six hours re-mining arXiv:

| In the zip | Why |
|---|---|
| `repo/` | the installable package (`src/`, `scripts/`, `app/`, `tests/`, `docs/`) and the six notebooks |
| `data/*.jsonl`, `data/*.json` | mined contexts, related-work pairs, LLM labels, the gold set, the OpenAlex and arXiv metadata caches |
| `data/sources/` | the 392 cached arXiv LaTeX sources, so Stage 1 re-parses locally and downloads nothing |
| `results/`, `figures/` | so the report can be rebuilt on Colab without re-running anything |

Left out, because Colab re-downloads them cheaply or must build them itself:
`.venv/`, `.git/`, `data/cache/` (the SciCite and ACL-ARC corpora, 322 MB) and
`models/` (the trained checkpoints, 420 MB). `--no-data` gives a code-only zip of
about 1 MB; `--no-sources` drops the LaTeX cache, but then re-running notebook 01
re-downloads every paper, because `write_datasets` rewrites
`citation_contexts.jsonl` from scratch on each run rather than appending.

## Step 2 · Put it on Drive

1. Open [drive.google.com](https://drive.google.com) and create a folder named
   **`litreview`** in *My Drive* (exactly that name, directly under My Drive).
2. Drag `litreview_colab.zip` into it. Do **not** unzip it — the notebook does
   that itself, into the right places.

Two alternatives, if you would rather not use a zip: copy the project folder to
`MyDrive/litreview/repo`, or set `REPO_URL` in the first cell of each notebook to
a git URL it should clone. The first cell tries all three in that order.

## Step 3 · Add the secrets in Colab

In any Colab notebook: **key icon in the left sidebar → Add new secret**, and turn
on *Notebook access* for each one.

| Secret | Needed by | Notes |
|---|---|---|
| `GROQ_API_KEY` | 02, 03, 04, 05, 06 | free tier; the project's default provider |
| `OPENALEX_MAILTO` | 01, 04 | your email; puts you in OpenAlex's faster "polite pool" |
| `OPENALEX_API_KEY` | 01, 04 | optional; raises the daily OpenAlex budget |
| `GEMINI_API_KEY` | optional | alternative provider, set `LITREVIEW_PROVIDER=gemini` |
| `S2_API_KEY` | optional | raises Semantic Scholar rate limits |
| `HF_TOKEN` | optional | only if a gated base model is used in 05 |

Keys are read from Colab secrets at run time. None of them is stored in a
notebook, and nothing writes a key to Drive.

## Step 4 · Open the notebooks

**File → Open notebook → Google Drive**, then pick
`litreview/repo/notebooks/01_mine_arxiv.ipynb` — the zip has already been
unpacked by then if you ran any notebook once, so the first time you may instead
need **File → Upload notebook** and drag one notebook in from your machine; its
first cell unpacks the zip, after which all six are visible on Drive.

Run the first cell. It mounts Drive, finds the zip or folder, `pip install -e`s
the package (about two minutes) and prints which data files are already present:

```
extracting litreview_colab.zip ...
project code: /content/drive/MyDrive/litreview/repo
data root: /content/drive/MyDrive/litreview
already present: {'citation_contexts.jsonl': 28672, 'rw_pairs.jsonl': 2816, ...}
```

Then **Runtime → Run all**. Nothing already cached is paid for twice: with the
bundled data, notebook 01 re-parses the cached LaTeX in a few minutes and makes
no network calls at all, so the first stage that costs real time is notebook 02.

For notebooks 02 and 05: **Runtime → Change runtime type → T4 GPU** first.

## Which notebook to run for what

* **Already have the mined data (the default bundle).** Go straight to 02 to
  fine-tune SciBERT on a GPU, then 05, which is the one stage this project could
  not run at all locally — QLoRA distillation needs CUDA.
* **Want to mine more papers.** Edit the CONFIG cell in 01 and raise
  `N_TRAIN`/`N_TEST`. It adds the new papers to the listings and re-derives both
  datasets from the whole cache, so the existing rows survive.
* **Just want the report rebuilt.** `!python scripts/build_report.py` in any
  notebook, once the first cell has run.

## Drive folder layout

```
MyDrive/litreview/
  data/
    papers_train.jsonl  papers_test.jsonl      # arXiv listings per split
    sources/<arxiv_id>.json                    # cached LaTeX sources ({} = none)
    openalex_cache.json                        # cited title -> metadata
    citation_contexts.jsonl                    # relation-classifier data
    rw_pairs.jsonl                             # the benchmark
    multixscience_test.jsonl                   # external benchmark
    llm_labels.jsonl                           # LLM relation labels (append-only)
    gold_blind.jsonl  gold_labelled.jsonl      # adjudicated gold set
    verifier_dev.jsonl                         # hand-checked calibration items
    extractions_*.jsonl  claim_verdicts_*.json
    distill_train.jsonl                        # Verifier-filtered distillation data
    eval_generations_*.jsonl                   # cached system outputs
    cache/                                     # SciCite / ACL-ARC downloads
  models/
    scibert_relations/                         # component A
    distilled_extractor/                       # component B
  results/*.json                               # every number in the report
  figures/*.png
```

## Notebook order, runtimes and secrets

| # | Notebook | GPU | Expected T4 runtime | Secrets | Writes |
|---|---|---|---|---|---|
| 01 | `01_mine_arxiv.ipynb` | no | 2-6 h at full scale (3000/300 papers); ~45 min at pilot scale (150/50) | `OPENALEX_MAILTO` | `citation_contexts.jsonl`, `rw_pairs.jsonl`, `multixscience_test.jsonl`, `results/stage1_yield.json` |
| 02 | `02_relation_labels_and_scibert.ipynb` | yes | 1.5-3 h (labelling is API-bound, SciBERT is ~15 min on T4) | `GROQ_API_KEY` | `llm_labels.jsonl`, `models/scibert_relations/`, `results/stage2_*.json` |
| 03 | `03_crew_extract_verify.ipynb` | yes (for the NLI model) | 1-3 h for 100-300 papers | `GROQ_API_KEY` | `extractions_*.jsonl`, `claim_verdicts_*.json`, `results/verifier_calibration.json`, `results/stage3_*.json` |
| 04 | `04_synthesis_and_gaps.ipynb` | yes | 20-50 min | `GROQ_API_KEY`, `OPENALEX_MAILTO` | `results/flow_runs/<topic>/`, `results/flow_summary.json` |
| 05 | `05_distill_extractor_qlora.ipynb` | **yes, required** | 2-4 h (data build + QLoRA on ~400 examples) | `GROQ_API_KEY` | `models/distilled_extractor/`, `results/stage6_distill.json` |
| 06 | `06_evaluation.ipynb` | yes | 2-5 h for S1-S4 over 100 test examples | `GROQ_API_KEY` | `results/eval_*.json`, `figures/*.png` |

Runtimes are for the full-scale configuration. Each notebook's first markdown cell
repeats its own purpose, inputs, outputs, runtime and secrets.

## Free-tier budget

Groq's free tier is per organisation, not per key, and the published limits change;
check **console.groq.com -> Settings -> Limits** for yours. At the time of writing
the larger models are roughly 30 requests/minute, 1,000 requests/day and 200k
tokens/day, and the 8B models are considerably higher.

The project is built around those limits:

* `litreview.llm.RateLimiter` is a token bucket on both requests and tokens per
  minute, configured from `PROVIDERS` in `litreview/config.py`.
* Every long-running step is append-only and keyed, so hitting a **daily** limit is
  not a failure: re-run the cell tomorrow and it continues.
* Relation labelling batches 8 contexts per request, which is roughly an 8x saving
  on the request budget.
* `results/llm_calls.jsonl` records tokens, latency and cost for every call;
  `scripts/usage_report.py` aggregates it so you can see what a stage actually cost
  before scaling it up.

Rough full-scale budget, measured from the pilot's per-call token counts: labelling
5,000 contexts is ~625 requests and ~1.6M tokens, which is several days of the
200k/day tier or one day on an 8B model. If that is too slow, set
`LITREVIEW_SMALL_MODEL` and pass `--size small`, or switch
`LITREVIEW_PROVIDER=gemini`.

## If a notebook disconnects

Re-run all cells. Nothing re-downloads, re-labels or re-extracts work that is
already cached. The only cells that cost time again are the ones whose cache you
deleted.

## Running the same code locally

The notebooks `pip install -e` this repository and call the same package, so there
is no second implementation to keep in sync. Locally:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[ml,eval,dev]"
cp .env.example .env            # add GROQ_API_KEY
.venv/bin/python scripts/run_stage1.py --n-train 150 --n-test 50
```

See `README.md` for the full local sequence.
