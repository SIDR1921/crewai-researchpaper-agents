"""The Verifier: is a claim actually supported by the passage it quotes?

Two independent checks per claim:

(a) **Quote check** -- does the evidence passage really occur in the text the
    Extractor was given? ``rapidfuzz.partial_ratio`` rather than exact matching,
    because LaTeX-to-text conversion leaves whitespace and ligature noise that a
    faithful quote still trips over.
(b) **Entailment check** -- an NLI model scores ``premise = evidence passage``,
    ``hypothesis = claim``. A claim can quote correctly and still overstate what
    the passage says; this is the check that catches it.

Both thresholds are fitted on a small hand-checked dev set
(:func:`calibrate`), not guessed. The same machinery scores review sentences
against their cited claims' evidence at write time, which is where the
faithfulness score comes from.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

from rapidfuzz import fuzz

from .config import HF_MODELS
from .schemas import ClaimVerdict, SentenceVerdict

#: Defaults, overwritten by whatever :func:`calibrate` writes to
#: ``results/verifier_calibration.json``.
DEFAULT_THRESHOLDS = {"quote_min": 85.0, "entail_supported": 0.60, "entail_partial": 0.30}


def normalise_for_match(s: str) -> str:
    s = (s or "").replace("–", "-").replace("—", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"\[CITE:[^\]]*\]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_score(passage: str, source: str) -> float:
    """0-100: how well the passage matches some span of the source."""
    p, s = normalise_for_match(passage), normalise_for_match(source)
    if not p or not s:
        return 0.0
    if p in s:
        return 100.0
    return float(fuzz.partial_ratio(p, s))


class NLI:
    """DeBERTa-v3 MNLI/FEVER/ANLI entailment scorer, loaded once."""

    def __init__(self, model_name: str = HF_MODELS["nli"], device: str | None = None,
                 max_len: int = 512, batch_size: int = 8) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .relations.scibert import best_device
        self.device = device or best_device()
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.max_len, self.batch_size = max_len, batch_size
        self._torch = torch
        labels = [self.model.config.id2label[i].lower()
                  for i in range(self.model.config.num_labels)]
        self.i_entail = next((i for i, l in enumerate(labels) if "entail" in l), 0)
        self.i_contra = next((i for i, l in enumerate(labels) if "contra" in l), None)
        self.labels = labels

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, float]]:
        """``[(premise, hypothesis)]`` -> ``[(P(entail), P(contradiction))]``."""
        torch = self._torch
        out: list[tuple[float, float]] = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i:i + self.batch_size]
            enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], truncation=True,
                           max_length=self.max_len, padding=True,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits.float(), -1).cpu()
            for row in probs:
                e = float(row[self.i_entail])
                c = float(row[self.i_contra]) if self.i_contra is not None else 0.0
                out.append((round(e, 4), round(c, 4)))
        return out


@dataclass
class Thresholds:
    quote_min: float = DEFAULT_THRESHOLDS["quote_min"]
    entail_supported: float = DEFAULT_THRESHOLDS["entail_supported"]
    entail_partial: float = DEFAULT_THRESHOLDS["entail_partial"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Thresholds":
        from .config import paths
        p = Path(path) if path else paths().results / "verifier_calibration.json"
        if p.exists():
            try:
                d = json.loads(p.read_text()).get("thresholds", {})
                return cls(**{k: float(v) for k, v in d.items() if k in DEFAULT_THRESHOLDS})
            except Exception:
                pass
        return cls()

    def verdict(self, quote: float, entail: float) -> str:
        if quote < self.quote_min:
            return "unsupported"
        if entail >= self.entail_supported:
            return "supported"
        if entail >= self.entail_partial:
            return "partially"
        return "unsupported"


class Verifier:
    def __init__(self, nli: NLI | None = None, thresholds: Thresholds | None = None) -> None:
        self.nli = nli or NLI()
        self.th = thresholds or Thresholds.load()

    # ------------------------------------------------------------- claim level

    def verify_claims(self, record: dict, source_text: str) -> list[ClaimVerdict]:
        claims = record.get("claims") or []
        if not claims:
            return []
        quotes = [quote_score(c["evidence"]["passage"], source_text) for c in claims]
        scores = self.nli.score([(c["evidence"]["passage"], c["text"]) for c in claims])
        out = []
        for c, q, (e, contra) in zip(claims, quotes, scores):
            out.append(ClaimVerdict(
                claim_id=c["claim_id"], paper_id=record.get("paper_id", ""),
                quote_found=q >= self.th.quote_min, quote_score=round(q, 2),
                entail_prob=e, contradiction_prob=contra,
                verdict=self.th.verdict(q, e)))
        return out

    def verify_corpus(self, records: dict[str, dict], out_path: str | Path | None = None) -> dict:
        """Verify every cached extraction. ``records`` is :func:`extract.load_records` output."""
        rows, per_paper = [], {}
        for pid, row in records.items():
            rec = row["record"]
            src = (row.get("meta") or {}).get("input_text") or ""
            vs = self.verify_claims(rec, src)
            per_paper[pid] = [v.model_dump() for v in vs]
            rows += vs
        counts = {k: sum(1 for v in rows if v.verdict == k)
                  for k in ("supported", "partially", "unsupported")}
        report = {
            "thresholds": asdict(self.th),
            "nli_model": self.nli.model_name,
            "n_papers": len(records), "n_claims": len(rows),
            "verdict_counts": counts,
            "verdict_rates": {k: round(v / max(1, len(rows)), 4) for k, v in counts.items()},
            "quote_found_rate": round(sum(v.quote_found for v in rows) / max(1, len(rows)), 4),
            "mean_quote_score": round(sum(v.quote_score for v in rows) / max(1, len(rows)), 2),
            "mean_entail_prob": round(sum(v.entail_prob for v in rows) / max(1, len(rows)), 4),
            "fully_supported_papers": sum(
                1 for vs in per_paper.values()
                if vs and all(v["verdict"] == "supported" for v in vs)),
        }
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps({"report": report, "per_paper": per_paper},
                                                 indent=2))
        return report

    # ---------------------------------------------------------- sentence level

    def verify_sentences(self, sentences: Sequence[dict], evidence_by_claim: dict[str, str],
                         evidence_by_paper: dict[str, list[str]] | None = None
                         ) -> list[SentenceVerdict]:
        """Score each review sentence against the evidence of the claims it cites.

        A sentence with no resolvable claim falls back to any evidence from the
        papers it cites; a sentence that cites nothing at all is unsupported by
        construction, which is the behaviour we want for a grounded review.
        """
        out: list[SentenceVerdict] = []
        pairs: list[tuple[str, str]] = []
        index: list[tuple[int, str]] = []
        for i, s in enumerate(sentences):
            prem = [evidence_by_claim[c] for c in (s.get("supporting_claim_ids") or [])
                    if c in evidence_by_claim]
            if not prem and evidence_by_paper:
                for pid in (s.get("cited_paper_ids") or []):
                    prem += (evidence_by_paper.get(pid) or [])[:4]
            for p in prem[:8]:
                pairs.append((p, s["text"]))
                index.append((i, p))
            out.append(SentenceVerdict(text=s["text"],
                                       cited_paper_ids=s.get("cited_paper_ids") or []))
        if pairs:
            scored = self.nli.score(pairs)
            for (i, prem), (e, _c) in zip(index, scored):
                if e > out[i].entail_prob:
                    out[i].entail_prob = e
                    out[i].best_evidence = prem[:400]
        for v in out:
            v.verdict = ("supported" if v.entail_prob >= self.th.entail_supported
                         else "partially" if v.entail_prob >= self.th.entail_partial
                         else "unsupported")
        return out

    def faithfulness(self, verdicts: Sequence[SentenceVerdict]) -> dict:
        n = len(verdicts)
        sup = sum(v.verdict == "supported" for v in verdicts)
        part = sum(v.verdict == "partially" for v in verdicts)
        return {"n_sentences": n,
                "supported": sup, "partially": part, "unsupported": n - sup - part,
                "faithfulness": round(sup / n, 4) if n else 0.0,
                "faithfulness_lenient": round((sup + part) / n, 4) if n else 0.0,
                "mean_entail_prob": round(sum(v.entail_prob for v in verdicts) / n, 4) if n else 0.0}


# ---------------------------------------------------------------- calibration

def calibrate(dev_items: Sequence[dict], nli: NLI | None = None,
              out_path: str | Path | None = None) -> dict:
    """Fit the two thresholds on hand-checked (claim, evidence, source, label) items.

    ``dev_items``: ``{claim, passage, source, human}`` where ``human`` is
    ``supported`` / ``partially`` / ``unsupported``. The grid is scored by macro-F1
    over the three verdicts, so a threshold pair is not chosen for accuracy on the
    majority verdict alone.
    """
    nli = nli or NLI()
    quotes = [quote_score(d["passage"], d["source"]) for d in dev_items]
    ent = [e for e, _ in nli.score([(d["passage"], d["claim"]) for d in dev_items])]
    gold = [d["human"] for d in dev_items]
    kinds = ("supported", "partially", "unsupported")

    def macro_f1(pred: Sequence[str]) -> float:
        f1s = []
        for k in kinds:
            tp = sum(p == k and g == k for p, g in zip(pred, gold))
            fp = sum(p == k and g != k for p, g in zip(pred, gold))
            fn = sum(p != k and g == k for p, g in zip(pred, gold))
            if tp + fn == 0:
                continue
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn)
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        return sum(f1s) / len(f1s) if f1s else 0.0

    grid = []
    for qm in [60, 70, 75, 80, 85, 90, 95, 100]:
        for es in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for ep in [0.1, 0.2, 0.3, 0.4]:
                if ep >= es:
                    continue
                th = Thresholds(qm, es, ep)
                pred = [th.verdict(q, e) for q, e in zip(quotes, ent)]
                acc = sum(p == g for p, g in zip(pred, gold)) / max(1, len(gold))
                grid.append({"quote_min": qm, "entail_supported": es, "entail_partial": ep,
                             "macro_f1": round(macro_f1(pred), 4), "accuracy": round(acc, 4)})
    best = max(grid, key=lambda g: (g["macro_f1"], g["accuracy"]))
    report = {
        "n_dev_items": len(dev_items),
        "nli_model": nli.model_name,
        "gold_distribution": {k: gold.count(k) for k in kinds},
        "thresholds": {k: best[k] for k in ("quote_min", "entail_supported", "entail_partial")},
        "best_macro_f1": best["macro_f1"], "best_accuracy": best["accuracy"],
        "defaults_macro_f1": next(
            (g["macro_f1"] for g in grid
             if g["quote_min"] == DEFAULT_THRESHOLDS["quote_min"]
             and g["entail_supported"] == DEFAULT_THRESHOLDS["entail_supported"]
             and g["entail_partial"] == DEFAULT_THRESHOLDS["entail_partial"]), None),
        "grid": sorted(grid, key=lambda g: -g["macro_f1"])[:15],
        "per_item": [{"claim": d["claim"][:160], "human": g, "quote_score": round(q, 1),
                      "entail_prob": e} for d, g, q, e in zip(dev_items, gold, quotes, ent)],
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
    return report
