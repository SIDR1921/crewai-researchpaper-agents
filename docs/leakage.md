# Time split and leakage controls

Every number in this project that compares a system against a human-written
related-work section depends on the test papers being genuinely unseen. This file
records the dates used and why, so the claim can be checked rather than trusted.

## Windows

| Split | Submission window (arXiv) | Purpose |
|---|---|---|
| train | 2023-01-01 to 2024-12-31 | LLM labelling pool, SciBERT training, distillation data |
| test  | 2026-06-01 to 2026-09-30 | the frozen evaluation benchmark |

Categories: `cs.CL` and `cs.IR`, primary or cross-listed.

## Why these dates

1. **After the training window.** The test window starts eighteen months after the
   training window ends, so no test paper can appear in the relation-classifier's
   training data, and no cited paper's related-work text can have been used to fit
   a threshold.

2. **After the evaluated models' knowledge cutoffs.** The generator models are
   accessed through hosted and local APIs whose weights were frozen before the
   test window opens. The brief's suggested window began in March 2026; it was
   narrowed to **June 2026** because the assistant that wrote this code has a
   knowledge cutoff of **May 2026**, and papers from March to May 2026 could in
   principle be memorised. Starting in June removes that overlap entirely.
   The window ends in September 2026 so that papers have had time to receive a
   stable arXiv version before mining (run on 2026-10-20).

3. **Nothing is tuned on the test split.** The Verifier's thresholds are fitted on
   a hand-checked dev set drawn from *train*-split extractions. The relation
   classifier's dev split is carved out of train-split papers by citing paper. The
   gold label set is sampled from train-split contexts. `scripts/run_eval.py`
   reads the test split only to score.

## Residual leakage risks, stated rather than dismissed

* **Cited papers are old by construction.** A 2026 paper cites work from before
  2026, so the *inputs* to every system (the cited abstracts) are plausibly inside
  every model's training data. This is unavoidable for this task -- the benchmark
  asks a system to synthesise known papers the way a 2026 author did -- and it
  does not let a model recover the *target* text, which is what is scored.
* **arXiv versioning.** `arxiv_id` is stored without the version suffix and the
  mined source is whatever `arxiv.org/e-print` served on the mining date. A paper
  revised after mining would differ from the stored copy; `data/sources/` is the
  frozen record of what was actually parsed.
* **Self-citation of this project's own topic.** The demonstration topic
  (faithfulness evaluation in abstractive summarisation) overlaps Chapter 2's
  subject matter. The demonstration run is reported as a qualitative example only;
  no quantitative claim rests on it.

## Verification

`scripts/dataset_stats.py` writes a `time_split` block to
`results/stage1_dataset_stats.json` containing the latest train date, the earliest
test date and whether they are disjoint. The report quotes that block rather than
this file.
