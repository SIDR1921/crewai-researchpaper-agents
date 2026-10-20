"""Baselines the fine-tuned classifier has to beat: majority class and TF-IDF + LR.

The zero-shot LLM baseline lives in :mod:`litreview.relations.label` because it
reuses the same prompt as the labeller.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from .scibert import encode_text, metrics
from .taxonomy import LABEL_TO_ID, LABELS


def majority_baseline(train_rows: Sequence[dict], test_rows: Sequence[dict]) -> dict:
    top = Counter(r["label"] for r in train_rows).most_common(1)[0][0]
    golds = [LABEL_TO_ID[r["label"]] for r in test_rows]
    return {"predicted_class": top} | metrics(golds, [LABEL_TO_ID[top]] * len(golds))


def tfidf_lr_baseline(train_rows: Sequence[dict], test_rows: Sequence[dict],
                      seed: int = 0) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def text(r):
        a, b = encode_text(r)
        return f"{a} [SEP] {b}"

    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60_000),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    pipe.fit([text(r) for r in train_rows], [r["label"] for r in train_rows])
    preds = pipe.predict([text(r) for r in test_rows])
    return metrics([LABEL_TO_ID[r["label"]] for r in test_rows],
                   [LABEL_TO_ID[p] for p in preds])


def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    """Cohen's kappa between two label sequences (LLM vs human gold)."""
    n = len(a)
    if n == 0:
        return 0.0
    agree = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return round((agree - expected) / (1 - expected), 4) if expected < 1 else 1.0
