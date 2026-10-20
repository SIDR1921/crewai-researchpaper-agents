"""Fine-tune SciBERT for citation-relation classification (component A).

Design points that matter for the numbers:

* **Input.** The citing sentence is marked with ``[[CITED]]`` at the position of
  the citation under question and other citations in the same sentence collapse
  to ``[OTHER]``, so a multi-citation sentence has a well-defined label. The
  cited paper's title is appended as a second segment.
* **Split by citing paper.** Sentences from one paper are highly correlated, so a
  random split would leak. Papers, not rows, are partitioned.
* **Class weights.** ``background`` dominates; weights are inverse-frequency so
  the rare relations (``supports``, ``contrasts``) are learnable.
* A plain PyTorch loop (no ``Trainer``) so the same code runs on MPS and CUDA.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import HF_MODELS
from .taxonomy import ID_TO_LABEL, LABEL_TO_ID, LABELS


def best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_text(row: dict) -> tuple[str, str]:
    """(context with the target citation marked, cited title) -> a sentence pair."""
    ctx = row.get("marked_context") or row.get("context") or row.get("cite_sentence") or ""
    return ctx, (row.get("cited_title") or "")


def split_by_paper(rows: Sequence[dict], frac_dev: float = 0.15, seed: int = 0
                   ) -> tuple[list[dict], list[dict]]:
    """Partition by ``citing_id`` so no paper appears in both halves."""
    papers = sorted({r.get("citing_id") or "" for r in rows})
    rng = random.Random(seed)
    rng.shuffle(papers)
    n_dev = max(1, int(len(papers) * frac_dev))
    dev = set(papers[:n_dev])
    return ([r for r in rows if (r.get("citing_id") or "") not in dev],
            [r for r in rows if (r.get("citing_id") or "") in dev])


def class_weights(labels: Sequence[str]) -> list[float]:
    counts = Counter(labels)
    n, k = len(labels), len(LABELS)
    return [n / (k * counts[l]) if counts[l] else 0.0 for l in LABELS]


# ----------------------------------------------------------------------- train

def train(
    rows: Sequence[dict],
    out_dir: str | Path,
    model_name: str = HF_MODELS["scibert"],
    epochs: int = 4,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_len: int = 256,
    seed: int = 0,
    dev_rows: Sequence[dict] | None = None,
    device: str | None = None,
    progress=None,
) -> dict:
    """Fine-tune and save to ``out_dir``. Returns the training report."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = device or best_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = list(rows)
    dev = list(dev_rows) if dev_rows is not None else None
    if dev is None:
        train_rows, dev = split_by_paper(train_rows, seed=seed)

    tok = AutoTokenizer.from_pretrained(model_name)

    class DS(Dataset):
        def __init__(self, rs):
            self.rs = list(rs)

        def __len__(self):
            return len(self.rs)

        def __getitem__(self, i):
            a, b = encode_text(self.rs[i])
            enc = tok(a, b, truncation=True, max_length=max_len, padding="max_length",
                      return_tensors="pt")
            item = {k: v[0] for k, v in enc.items()}
            item["labels"] = torch.tensor(LABEL_TO_ID[self.rs[i]["label"]])
            return item

    dl_train = DataLoader(DS(train_rows), batch_size=batch_size, shuffle=True)
    dl_dev = DataLoader(DS(dev), batch_size=batch_size) if dev else None

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS),
        id2label=ID_TO_LABEL, label2id=LABEL_TO_ID).to(device)

    w = torch.tensor(class_weights([r["label"] for r in train_rows]),
                     dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = max(1, epochs * len(dl_train))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total,
                                                pct_start=0.1, anneal_strategy="linear")

    history = []
    steps = 0
    for ep in range(epochs):
        model.train()
        running = 0.0
        it = progress(dl_train, f"epoch {ep + 1}/{epochs}") if progress else dl_train
        for batch in it:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = loss_fn(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += float(loss.item())
            steps += 1
        ep_rec = {"epoch": ep + 1, "train_loss": round(running / max(1, len(dl_train)), 4)}
        if dl_dev:
            ep_rec |= {"dev_" + k: v for k, v in
                       _evaluate(model, dl_dev, device).items() if k in ("accuracy", "macro_f1")}
        history.append(ep_rec)

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    report = {
        "model_name": model_name, "device": device, "epochs": epochs, "lr": lr,
        "batch_size": batch_size, "max_len": max_len, "seed": seed,
        "n_train_rows": len(train_rows), "n_dev_rows": len(dev or []),
        "n_train_papers": len({r.get("citing_id") for r in train_rows}),
        "n_dev_papers": len({r.get("citing_id") for r in (dev or [])}),
        "train_label_counts": dict(Counter(r["label"] for r in train_rows)),
        "class_weights": dict(zip(LABELS, [round(x, 3) for x in class_weights(
            [r["label"] for r in train_rows])])),
        "history": history, "steps": steps, "out_dir": str(out_dir),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2))
    return report


def _evaluate(model, dl, device: str) -> dict:
    import torch
    model.eval()
    preds, golds, probs = [], [], []
    with torch.no_grad():
        for batch in dl:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float().cpu()
            p = torch.softmax(logits, -1)
            preds += p.argmax(-1).tolist()
            probs += p.max(-1).values.tolist()
            golds += labels.tolist()
    return metrics(golds, preds) | {"mean_confidence": round(float(np.mean(probs)), 4) if probs else 0.0}


def metrics(golds: Sequence[int], preds: Sequence[int]) -> dict:
    """Accuracy, macro-F1, per-class P/R/F1 and the confusion matrix."""
    k = len(LABELS)
    cm = [[0] * k for _ in range(k)]
    for g, p in zip(golds, preds):
        cm[g][p] += 1
    per = {}
    f1s = []
    for i, name in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(k)) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[name] = {"precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4), "support": sum(cm[i])}
        if sum(cm[i]):                       # macro over classes present in the gold
            f1s.append(f1)
    acc = sum(cm[i][i] for i in range(k)) / max(1, len(golds))
    return {"accuracy": round(acc, 4), "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
            "per_class": per, "confusion": cm, "labels": LABELS, "n": len(golds)}


# ------------------------------------------------------------------- inference

class RelationClassifier:
    """``classify(context) -> (label, prob)`` over a saved checkpoint."""

    def __init__(self, model_dir: str | Path, device: str | None = None, max_len: int = 256) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device or best_device()
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device).eval()
        self._torch = torch
        cfg_labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.labels = cfg_labels if set(cfg_labels) == set(LABELS) else LABELS

    def classify(self, context: str, cited_title: str = "") -> tuple[str, float]:
        return self.classify_batch([{"marked_context": context, "cited_title": cited_title}])[0]

    def classify_batch(self, rows: Sequence[dict], batch_size: int = 32) -> list[tuple[str, float]]:
        torch = self._torch
        out: list[tuple[str, float]] = []
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            pairs = [encode_text(r) for r in chunk]
            enc = self.tok([a for a, _ in pairs], [b for _, b in pairs], truncation=True,
                           max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits.float(), -1).cpu()
            out += [(self.labels[int(j)], round(float(v), 4))
                    for j, v in zip(p.argmax(-1), p.max(-1).values)]
        return out

    def evaluate(self, rows: Sequence[dict]) -> dict:
        preds = self.classify_batch(rows)
        return metrics([LABEL_TO_ID[r["label"]] for r in rows],
                       [LABEL_TO_ID[l] for l, _ in preds])
