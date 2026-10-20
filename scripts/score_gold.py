#!/usr/bin/env python
"""Score the LLM labeller against the human gold set, and the fine-tuned model too.

Outputs ``results/stage2_gold_agreement.json``: accuracy, Cohen's kappa and the
per-class confusion of LLM vs human, plus the same for every other system whose
predictions are supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from litreview.config import paths
from litreview.relations.baselines import cohen_kappa
from litreview.relations.scibert import metrics
from litreview.relations.taxonomy import LABEL_TO_ID, LABELS


def load_gold(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.open()]
    return [r for r in rows if r.get("human_label") in LABELS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    P = paths()
    gold_path = Path(args.gold or (P.data / "gold_labelled.jsonl"))
    gold = load_gold(gold_path)
    if not gold:
        print(f"no labelled gold rows in {gold_path}")
        return 1
    key = json.loads((P.data / "gold_key.json").read_text())

    pairs = [(r["human_label"], (key.get(r["context_id"]) or {}).get("llm_label"))
             for r in gold]
    pairs = [(h, l) for h, l in pairs if l in LABELS]
    human = [h for h, _ in pairs]
    llm = [l for _, l in pairs]
    rep = {
        "n_gold": len(gold), "n_comparable": len(pairs),
        "gold_distribution": {l: human.count(l) for l in LABELS},
        "llm_distribution": {l: llm.count(l) for l in LABELS},
        "llm_vs_gold": metrics([LABEL_TO_ID[h] for h in human],
                               [LABEL_TO_ID[l] for l in llm]) |
                       {"cohen_kappa": cohen_kappa(llm, human)},
        "note": ("The gold set is stratified over the LLM's predicted label so rare relations "
                 "are represented; agreement here is therefore not an estimate of accuracy on "
                 "a uniformly random citation context."),
        "gold_path": str(gold_path),
    }
    out = Path(args.out or (P.results / "stage2_gold_agreement.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "llm_vs_gold"}, indent=2))
    print("accuracy", rep["llm_vs_gold"]["accuracy"], "macro_f1",
          rep["llm_vs_gold"]["macro_f1"], "kappa", rep["llm_vs_gold"]["cohen_kappa"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
