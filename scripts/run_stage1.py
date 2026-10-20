#!/usr/bin/env python
"""Stage 1 at pilot scale: mine arXiv related-work sections and citation contexts.

    python scripts/run_stage1.py --n-train 150 --n-test 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

from litreview.config import paths
from litreview.mining import multixscience
from litreview.mining.pipeline import CATEGORIES, TEST_WINDOW, TRAIN_WINDOW, run_stage1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--train-window", nargs=2, default=list(TRAIN_WINDOW))
    ap.add_argument("--test-window", nargs=2, default=list(TEST_WINDOW))
    ap.add_argument("--multixscience", type=int, default=200,
                    help="examples to keep from the Multi-XScience test split (0 to skip)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P = paths().mkdirs()
    print(f"root={P.root}")
    report = run_stage1(
        args.n_train, args.n_test, args.categories,
        tuple(args.train_window), tuple(args.test_window), seed=args.seed,
        fetch_progress=lambda ps, s: tqdm(ps, desc=f"fetch {s}"),
        progress=lambda ps, s: tqdm(ps, desc=f"parse {s}"),
        arxiv_progress=lambda bs: tqdm(bs, desc="arxiv meta"),
        oa_progress=lambda ts: tqdm(ts, desc="openalex"),
    )
    print(json.dumps(report, indent=2))

    if args.multixscience:
        try:
            mx = multixscience.build(P.data, split="test", limit=args.multixscience)
            print("multi-xscience:", json.dumps(mx, indent=2))
            report["multixscience"] = mx
        except Exception as e:
            report["multixscience"] = {"available": False, "error": str(e)[:300]}
            print("multi-xscience unavailable:", e, file=sys.stderr)
        (P.results / "stage1_yield.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
