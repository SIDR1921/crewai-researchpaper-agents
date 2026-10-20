"""Evaluation metrics for generated related-work sections (Stage 7).

Overlap metrics (ROUGE, BERTScore) say how close the text is to the human
section. They say nothing about whether the text is *true*, so they are reported
beside three grounding metrics:

* **citation coverage** -- of the papers the real section cites, how many does the
  generated section discuss?
* **citation correctness** -- of the (sentence, cited paper) pairs the system
  asserts, how many are actually supported by that paper's evidence?
* **faithfulness** -- fraction of sentences the NLI verifier calls supported.

All means are reported with bootstrap 95% CIs, because a 30-paper pilot has wide
intervals and a point estimate alone would overclaim.
"""
from __future__ import annotations

import random
import re
from typing import Sequence

_ROUGE = None
_BERTSCORE = {}


def _rouge():
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer
        _ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _ROUGE


def rouge(pred: str, ref: str) -> dict:
    s = _rouge().score(ref or "", pred or "")
    return {"rouge1": round(s["rouge1"].fmeasure, 4),
            "rouge2": round(s["rouge2"].fmeasure, 4),
            "rougeL": round(s["rougeL"].fmeasure, 4)}


def bertscore(preds: Sequence[str], refs: Sequence[str],
              model_type: str = "microsoft/deberta-base-mnli",
              batch_size: int = 8) -> list[float]:
    """BERTScore F1 per pair. Falls back to [] if the scorer cannot be loaded."""
    try:
        from bert_score import BERTScorer
    except ImportError:
        return []
    if model_type not in _BERTSCORE:
        try:
            _BERTSCORE[model_type] = BERTScorer(model_type=model_type, lang="en",
                                                rescale_with_baseline=False)
        except Exception:
            return []
    try:
        _p, _r, f1 = _BERTSCORE[model_type].score(list(preds), list(refs),
                                                  batch_size=batch_size)
        return [round(float(x), 4) for x in f1]
    except Exception:
        return []


# --------------------------------------------------------------- citation side

def _norm_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def citation_coverage(pred_text: str, pred_sentences: Sequence[dict],
                      gold_cited: Sequence[dict]) -> dict:
    """Fraction of the gold section's cited papers that the generated text discusses.

    A paper counts as discussed if its id appears in an inline citation, or if a
    distinctive chunk of its title appears in the text -- otherwise a system that
    paraphrases titles instead of using ids would score zero unfairly.
    """
    asserted = {_norm_id(x) for s in pred_sentences for x in (s.get("cited_paper_ids") or [])}
    asserted |= {_norm_id(x) for x in _inline_ids(pred_text)}
    low = (pred_text or "").lower()
    hit, detail = 0, []
    for c in gold_cited:
        key = _norm_id(c.get("cite_key") or c.get("paper_id") or "")
        title = (c.get("title") or "").lower()
        by_id = bool(key) and key in asserted
        by_title = bool(title) and len(title) > 25 and _title_overlap(title, low) >= 0.6
        ok = by_id or by_title
        hit += ok
        detail.append({"cite_key": c.get("cite_key"), "matched": ok,
                       "by_id": by_id, "by_title": by_title})
    n = len(gold_cited)
    return {"coverage": round(hit / n, 4) if n else 0.0, "n_gold_cited": n,
            "n_discussed": hit, "detail": detail}


def _inline_ids(text: str) -> list[str]:
    out = []
    for m in re.finditer(r"\[([^\]]{1,120})\]", text or ""):
        for part in m.group(1).split(","):
            p = part.strip()
            if p and p.lower() not in ("unverified", "cite"):
                out.append(p)
    return out


def _title_overlap(title: str, text: str) -> float:
    toks = [w for w in re.findall(r"[a-z0-9]+", title) if len(w) > 3]
    if not toks:
        return 0.0
    return sum(1 for w in toks if w in text) / len(toks)


def citation_correctness(pred_sentences: Sequence[dict], evidence_by_paper: dict[str, list[str]],
                         nli, threshold: float = 0.3) -> dict:
    """Of the (sentence, cited paper) pairs asserted, how many that paper actually supports.

    This is the metric that catches a system attaching a real paper id to a
    sentence that paper never said -- the most common way a grounded-looking
    review is wrong.
    """
    pairs, index = [], []
    for i, s in enumerate(pred_sentences):
        for pid in (s.get("cited_paper_ids") or []):
            ev = evidence_by_paper.get(pid) or evidence_by_paper.get(_norm_id(pid)) or []
            if not ev:
                index.append((i, pid, None))
                continue
            for e in ev[:4]:
                pairs.append((e, s["text"]))
                index.append((i, pid, len(pairs) - 1))
    scores = nli.score(pairs) if pairs else []
    best: dict[tuple[int, str], float] = {}
    unknown: set[tuple[int, str]] = set()
    for i, pid, j in index:
        if j is None:
            unknown.add((i, pid))
            continue
        e = scores[j][0]
        key = (i, pid)
        best[key] = max(best.get(key, 0.0), e)
    checked = len(best)
    correct = sum(1 for v in best.values() if v >= threshold)
    total = checked + len(unknown)
    return {"n_asserted_pairs": total, "n_checkable": checked,
            "n_unresolvable_paper_ids": len(unknown),
            "correctness": round(correct / checked, 4) if checked else 0.0,
            "correctness_strict": round(correct / total, 4) if total else 0.0,
            "threshold": threshold}


# -------------------------------------------------------------------- bootstrap

def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Mean with a percentile bootstrap CI. ``n`` < 2 returns the mean and no interval."""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    mean = sum(vals) / n
    if n < 2:
        return {"mean": round(mean, 4), "ci_low": None, "ci_high": None, "n": n}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"mean": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n": n,
            "n_boot": n_boot}


def aggregate(per_example: Sequence[dict], keys: Sequence[str], seed: int = 0) -> dict:
    """Bootstrap each metric across examples."""
    return {k: bootstrap_ci([e.get(k) for e in per_example if e.get(k) is not None], seed=seed)
            for k in keys}
