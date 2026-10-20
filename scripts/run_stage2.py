#!/usr/bin/env python
"""Stage 2: LLM-label citation contexts, fine-tune SciBERT, run the baselines.

    python scripts/run_stage2.py label  --limit 1200
    python scripts/run_stage2.py train  --epochs 4
    python scripts/run_stage2.py bench            # baselines + gold-set evaluation
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm.auto import tqdm

from litreview.config import paths
from litreview.llm import SETTINGS, probe, usage_summary
from litreview.relations import baselines, seed_data
from litreview.relations.label import label_contexts, load_labels, zero_shot_label
from litreview.relations.scibert import RelationClassifier, metrics, split_by_paper, train
from litreview.relations.taxonomy import LABELS, LABEL_TO_ID


def read_contexts(path: Path, split: str | None = "train", in_rw_only: bool = True) -> list[dict]:
    rows = []
    for line in path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if split and r.get("split") != split:
            continue
        if in_rw_only and not r.get("in_related_work"):
            continue
        if len(r.get("cite_sentence") or "") < 40:
            continue
        rows.append(r)
    return rows


def cmd_label(args) -> int:
    P = paths()
    print("provider probe:", json.dumps(probe(), indent=1))
    rows = read_contexts(P.data / "citation_contexts.jsonl", split=args.split,
                         in_rw_only=not args.all_sections)
    random.Random(args.seed).shuffle(rows)
    print(f"{len(rows)} candidate contexts; labelling up to {args.limit}")
    stats = label_contexts(rows, P.data / "llm_labels.jsonl", batch_size=args.batch_size,
                           limit=args.limit, progress=lambda b: tqdm(b, desc="label"))
    labelled = load_labels(P.data / "llm_labels.jsonl")
    report = {"stats": stats, "n_labelled_total": len(labelled),
              "label_distribution": {l: sum(1 for r in labelled if r["llm_label"] == l)
                                     for l in LABELS},
              "provider": SETTINGS.describe(), "usage": usage_summary()}
    (P.results / "stage2_labelling.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "provider"}, indent=2))
    return 0


def _gold_rows(P) -> list[dict]:
    p = P.data / "gold_labelled.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.open():
        r = json.loads(line)
        if r.get("human_label") in LABELS:
            out.append({**r, "label": r["human_label"]})
    return out


def cmd_train(args) -> int:
    P = paths().mkdirs()
    rows = [{**r, "label": r["llm_label"]} for r in load_labels(P.data / "llm_labels.jsonl")]
    gold_ids = {r["context_id"] for r in _gold_rows(P)}
    rows = [r for r in rows if r["context_id"] not in gold_ids]   # never train on gold
    extra = {"scicite": {"available": False}, "acl_arc": {"available": False}}
    if args.seed_corpora:
        sc = seed_data.load_scicite(P.cache, limit=args.seed_limit)
        aa = seed_data.load_acl_arc(P.cache, limit=args.seed_limit)
        extra = {"scicite": {k: v for k, v in sc.items() if k != "rows"},
                 "acl_arc": {k: v for k, v in aa.items() if k != "rows"}}
        rows += sc.get("rows", []) + aa.get("rows", [])
    if len(rows) < 50:
        print(f"only {len(rows)} labelled rows; label more first")
        return 1
    tr, dev = split_by_paper(rows, frac_dev=0.15, seed=args.seed)
    report = train(tr, P.models / "scibert_relations", epochs=args.epochs, lr=args.lr,
                   batch_size=args.batch_size, dev_rows=dev, seed=args.seed,
                   progress=lambda dl, d: tqdm(dl, desc=d))
    report["seed_corpora"] = extra
    gold = _gold_rows(P)
    if gold:
        clf = RelationClassifier(P.models / "scibert_relations")
        report["gold_eval"] = clf.evaluate(gold)
    (P.results / "stage2_scibert.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("gold_eval", "seed_corpora")}, indent=2))
    if "gold_eval" in report:
        print("gold macro-F1:", report["gold_eval"]["macro_f1"],
              "accuracy:", report["gold_eval"]["accuracy"])
    return 0


def cmd_bench(args) -> int:
    P = paths()
    gold = _gold_rows(P)
    if not gold:
        print("no gold set; run scripts/sample_gold.py and label it first")
        return 1
    rows = [{**r, "label": r["llm_label"]} for r in load_labels(P.data / "llm_labels.jsonl")]
    gold_ids = {r["context_id"] for r in gold}
    train_rows = [r for r in rows if r["context_id"] not in gold_ids]
    out = {"n_gold": len(gold), "n_train_rows": len(train_rows),
           "gold_distribution": {l: sum(1 for r in gold if r["label"] == l) for l in LABELS}}
    out["majority"] = baselines.majority_baseline(train_rows, gold)
    out["tfidf_lr"] = baselines.tfidf_lr_baseline(train_rows, gold, seed=args.seed)
    if args.zero_shot:
        zs = zero_shot_label(gold, progress=lambda b: tqdm(b, desc="zero-shot"))
        by_id = {z["context_id"]: z.get("llm_label") for z in zs}
        pairs = [(r["label"], by_id.get(r["context_id"])) for r in gold]
        pairs = [(h, p) for h, p in pairs if p in LABELS]
        out["zero_shot_llm"] = metrics([LABEL_TO_ID[h] for h, _ in pairs],
                                       [LABEL_TO_ID[p] for _, p in pairs]) | {
            "n_unparsed": len(gold) - len(pairs),
            "cohen_kappa": baselines.cohen_kappa([p for _, p in pairs],
                                                 [h for h, _ in pairs])}
    model_dir = P.models / "scibert_relations"
    if model_dir.exists():
        out["scibert"] = RelationClassifier(model_dir).evaluate(gold)
    out["usage"] = usage_summary()
    (P.results / "stage2_benchmark.json").write_text(json.dumps(out, indent=2))
    for k in ("majority", "tfidf_lr", "zero_shot_llm", "scibert"):
        if k in out:
            print(f"{k:16s} acc={out[k]['accuracy']:.3f} macroF1={out[k]['macro_f1']:.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("label")
    l.add_argument("--limit", type=int, default=1200)
    l.add_argument("--batch-size", type=int, default=8)
    l.add_argument("--split", default="train")
    l.add_argument("--all-sections", action="store_true",
                   help="label citations anywhere, not just in related-work sections")
    l.add_argument("--seed", type=int, default=0)
    l.set_defaults(fn=cmd_label)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--seed-corpora", action="store_true", default=True)
    t.add_argument("--no-seed-corpora", dest="seed_corpora", action="store_false")
    t.add_argument("--seed-limit", type=int, default=2000)
    t.set_defaults(fn=cmd_train)
    b = sub.add_parser("bench")
    b.add_argument("--zero-shot", action="store_true", default=True)
    b.add_argument("--no-zero-shot", dest="zero_shot", action="store_false")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_bench)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
