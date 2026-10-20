#!/usr/bin/env python
"""Sample a stratified gold set of citation contexts for human labelling.

Writes two files:

* ``data/gold_blind.jsonl``  -- the contexts with **no** LLM label, for the human
  (that is what makes the human labels independent);
* ``data/gold_key.json``     -- the mapping back to the LLM's labels, which is only
  read after the human labels exist.

Stratification is over the LLM's predicted label, so the rare relations are
present in the gold set at all; the resulting agreement numbers are therefore
*not* an estimate of accuracy on a random context, and the report says so.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from litreview.config import paths
from litreview.relations.label import load_labels
from litreview.relations.taxonomy import LABELS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--labels", default=None, help="path to llm_labels.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    P = paths()
    rows = load_labels(args.labels or (P.data / "llm_labels.jsonl"))
    if not rows:
        print("no LLM labels found; run stage 2 labelling first")
        return 1
    rng = random.Random(args.seed)
    by_label = {l: [r for r in rows if r["llm_label"] == l] for l in LABELS}
    for v in by_label.values():
        rng.shuffle(v)

    # Proportional, but with a floor so rare classes appear; then top up from the pool.
    quota = max(8, args.n // (2 * len(LABELS)))
    picked, seen = [], set()
    for l in LABELS:
        for r in by_label[l][:quota]:
            picked.append(r)
            seen.add(r["context_id"])
    pool = [r for r in rows if r["context_id"] not in seen]
    rng.shuffle(pool)
    picked += pool[: max(0, args.n - len(picked))]
    picked = picked[: args.n]
    rng.shuffle(picked)

    blind = P.data / "gold_blind.jsonl"
    with blind.open("w") as f:
        for i, r in enumerate(picked, 1):
            f.write(json.dumps({
                "n": i, "context_id": r["context_id"], "citing_id": r["citing_id"],
                "cite_key": r["cite_key"], "cited_title": r["cited_title"],
                "marked_context": r["marked_context"], "human_label": None}) + "\n")
    (P.data / "gold_key.json").write_text(json.dumps(
        {r["context_id"]: {"llm_label": r["llm_label"], "llm_rationale": r["llm_rationale"],
                           "llm_confidence": r["llm_confidence"]} for r in picked}, indent=2))
    print(json.dumps({
        "sampled": len(picked), "blind_path": str(blind),
        "llm_label_distribution_in_sample":
            {l: sum(1 for r in picked if r["llm_label"] == l) for l in LABELS},
        "pool_size": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
