"""The evaluation harness: run systems over the frozen benchmark and score them.

Every (system, example) outcome is cached as one JSONL row, so a run that stops
at a rate limit resumes. Scoring is a separate pass over the cache, so metrics can
be recomputed (for instance with a different NLI model) without re-generating any
text.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from ..llm import usage_summary
from . import metrics as M
from .systems import SYSTEM_LABELS, run_system

GEN_KEYS = ["rouge1", "rouge2", "rougeL", "bertscore", "coverage", "correctness",
            "faithfulness", "faithfulness_lenient", "wall_s", "n_sentences"]


def load_benchmark(path: str | Path, split: str = "test", limit: int | None = None,
                   min_cited: int = 3) -> list[dict]:
    """Read ``rw_pairs.jsonl`` (or the Multi-XScience file) and filter to one split."""
    rows = []
    for line in Path(path).open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if split and r.get("split") != split:
            continue
        if len([c for c in (r.get("cited") or []) if c.get("abstract")]) < min_cited:
            continue
        if not (r.get("rw_text") or "").strip():
            continue
        rows.append(r)
    rows.sort(key=lambda r: r.get("arxiv_id") or "")
    return rows[:limit] if limit else rows


def _key(system: str, pair: dict) -> str:
    return f"{system}||{pair.get('arxiv_id')}"


def generate(benchmark: Sequence[dict], systems: Sequence[str], cache_path: str | Path,
             verifier=None, classifier=None, distilled=None, progress=None,
             **kw) -> dict:
    """Run each system over each example, appending to the cache. Resumable."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if cache_path.exists():
        for line in cache_path.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("review") or row.get("skipped"):
                done.add(f"{row['system']}||{row['arxiv_id']}")

    jobs = [(s, p) for s in systems for p in benchmark if _key(s, p) not in done]
    it = progress(jobs) if progress else jobs
    stats = {"already_done": len(done), "attempted": 0, "failed": 0}
    with cache_path.open("a") as f:
        for system, pair in it:
            stats["attempted"] += 1
            t0 = time.time()
            try:
                out = run_system(system, pair, verifier=verifier, classifier=classifier,
                                 distilled=distilled, **kw)
            except Exception as e:  # noqa: BLE001 - never lose the rest of the run
                stats["failed"] += 1
                out = {"system": system, "error": str(e)[:400], "review": {},
                       "records": [], "evidence_by_paper": {},
                       "wall_s": round(time.time() - t0, 2), "notes": ["exception"]}
            row = {k: v for k, v in out.items() if k != "records"}
            row |= {"arxiv_id": pair.get("arxiv_id"), "source": pair.get("source"),
                    "n_cited": len(pair.get("cited") or [])}
            f.write(json.dumps(row) + "\n")
            f.flush()
    return stats


def score(benchmark: Sequence[dict], cache_path: str | Path, nli=None,
          with_bertscore: bool = True, progress=None) -> dict:
    """Score every cached generation. Returns per-example rows and aggregates."""
    by_id = {p.get("arxiv_id"): p for p in benchmark}
    rows = []
    for line in Path(cache_path).open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("arxiv_id") in by_id:
            rows.append(row)

    per_example = []
    it = progress(rows) if progress else rows
    for row in it:
        pair = by_id[row["arxiv_id"]]
        rev = row.get("review") or {}
        pred = (rev.get("markdown") or "").strip()
        sents = rev.get("sentences") or []
        rec = {"system": row["system"], "arxiv_id": row["arxiv_id"],
               "source": row.get("source"), "n_cited": row.get("n_cited"),
               "wall_s": row.get("wall_s"), "n_sentences": len(sents),
               "skipped": bool(row.get("skipped")), "error": row.get("error")}
        if not pred:
            per_example.append(rec)
            continue
        rec |= M.rouge(pred, pair.get("rw_text") or "")
        cov = M.citation_coverage(pred, sents, pair.get("cited") or [])
        rec["coverage"] = cov["coverage"]
        rec["n_gold_cited"] = cov["n_gold_cited"]
        if nli is not None:
            ev = row.get("evidence_by_paper") or {}
            if ev:
                cc = M.citation_correctness(sents, ev, nli)
                rec |= {"correctness": cc["correctness"],
                        "correctness_strict": cc["correctness_strict"],
                        "n_unresolvable_ids": cc["n_unresolvable_paper_ids"]}
            prem = {}
            for pid, passages in ev.items():
                prem[pid] = passages
            vs = _sentence_faithfulness(sents, prem, nli)
            rec |= vs
        per_example.append(rec)

    if with_bertscore:
        idx = [i for i, r in enumerate(per_example) if r.get("rouge1") is not None]
        if idx:
            preds, refs = [], []
            for i in idx:
                r = per_example[i]
                row = next(x for x in rows if x["system"] == r["system"]
                           and x["arxiv_id"] == r["arxiv_id"])
                preds.append((row.get("review") or {}).get("markdown") or "")
                refs.append(by_id[r["arxiv_id"]].get("rw_text") or "")
            f1s = M.bertscore(preds, refs)
            for i, v in zip(idx, f1s):
                per_example[i]["bertscore"] = v

    systems = sorted({r["system"] for r in per_example})
    aggregates = {}
    for s in systems:
        sub = [r for r in per_example if r["system"] == s and not r.get("skipped")]
        aggregates[s] = {"label": SYSTEM_LABELS.get(s, s), "n_examples": len(sub),
                         "n_errors": sum(1 for r in sub if r.get("error"))}
        aggregates[s] |= M.aggregate(sub, GEN_KEYS)
    return {"per_example": per_example, "aggregates": aggregates,
            "systems": systems, "n_benchmark": len(benchmark)}


def _sentence_faithfulness(sents: Sequence[dict], evidence_by_paper: dict, nli) -> dict:
    """NLI faithfulness over review sentences, using the cited papers' evidence."""
    from ..verify import Thresholds
    th = Thresholds.load()
    pairs, index = [], []
    for i, s in enumerate(sents):
        prem = []
        for pid in (s.get("cited_paper_ids") or []):
            prem += (evidence_by_paper.get(pid) or [])[:4]
        if not prem:
            prem = [p for ps in list(evidence_by_paper.values())[:6] for p in ps[:1]]
        for p in prem[:8]:
            pairs.append((p, s.get("text") or ""))
            index.append(i)
    best = [0.0] * len(sents)
    if pairs:
        for i, (e, _c) in zip(index, nli.score(pairs)):
            best[i] = max(best[i], e)
    sup = sum(1 for e in best if e >= th.entail_supported)
    part = sum(1 for e in best if th.entail_partial <= e < th.entail_supported)
    n = len(sents) or 1
    return {"faithfulness": round(sup / n, 4),
            "faithfulness_lenient": round((sup + part) / n, 4),
            "mean_entail_prob": round(sum(best) / n, 4)}


def save(report: dict, out_path: str | Path, extra: dict | None = None) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["llm_usage"] = usage_summary()
    if extra:
        payload |= extra
    p.write_text(json.dumps(payload, indent=2))
    return p


def table(aggregates: dict, keys: Sequence[str] = ("rouge1", "rouge2", "rougeL", "bertscore",
                                                   "coverage", "correctness", "faithfulness")
          ) -> list[dict]:
    """Main results table: mean [CI] per system per metric."""
    rows = []
    for s, agg in aggregates.items():
        row = {"system": agg.get("label", s), "n": agg.get("n_examples")}
        for k in keys:
            d = agg.get(k) or {}
            row[k] = (f"{d['mean']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"
                      if d.get("mean") is not None and d.get("ci_low") is not None
                      else (f"{d['mean']:.3f}" if d.get("mean") is not None else "-"))
        rows.append(row)
    return rows


def cost_table(aggregates: dict, usage: dict | None = None) -> list[dict]:
    """Latency and token/cost columns, from the central LLM usage log."""
    usage = usage or usage_summary()
    rows = []
    for s, agg in aggregates.items():
        lat = (agg.get("wall_s") or {}).get("mean")
        rows.append({"system": agg.get("label", s), "n": agg.get("n_examples"),
                     "wall_s_per_example": None if lat is None else round(lat, 1)})
    rows.append({"system": "TOTAL (all LLM calls this log)",
                 "n": usage["overall"]["calls"],
                 "wall_s_per_example": usage["overall"]["latency_s_total"],
                 "prompt_tokens": usage["overall"]["prompt_tokens"],
                 "completion_tokens": usage["overall"]["completion_tokens"],
                 "cost_usd": usage["overall"]["cost_usd"]})
    return rows
