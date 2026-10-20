#!/usr/bin/env python
"""Stage 7: run the evaluation harness over the frozen test benchmark.

    python scripts/run_eval.py --systems S1 S2 S3 --limit 30
    python scripts/run_eval.py --benchmark multixscience --systems S1 S3 --limit 20
    python scripts/run_eval.py --score-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from litreview.config import SETTINGS, paths
from litreview.eval.runner import cost_table, generate, load_benchmark, save, score, table
from litreview.llm import probe, usage_summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=["S1", "S2", "S3"])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--benchmark", default="arxiv", choices=["arxiv", "multixscience"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--distilled", default=None, help="path to the Stage 6 model, enables S4")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    P = paths().mkdirs()
    bench_path = (P.data / "rw_pairs.jsonl" if args.benchmark == "arxiv"
                  else P.data / f"multixscience_{args.split}.jsonl")
    if not bench_path.exists():
        print(f"{bench_path} missing; run stage 1 first")
        return 1
    bench = load_benchmark(bench_path, split=args.split, limit=args.limit)
    print(f"benchmark: {len(bench)} examples from {bench_path.name} ({args.split} split)")
    if not bench:
        return 1

    tag = args.tag or args.benchmark
    cache = P.data / f"eval_generations_{tag}.jsonl"

    verifier = classifier = distilled = None
    needs_verifier = any(s in ("S3", "S4", "S3_no_relations") for s in args.systems)
    if needs_verifier or True:
        from litreview.verify import Verifier
        verifier = Verifier()
    md = P.models / "scibert_relations"
    if md.exists() and "S3" in args.systems:
        from litreview.relations.scibert import RelationClassifier
        classifier = RelationClassifier(md)
    elif "S3" in args.systems:
        print(f"note: no classifier at {md}; S3 runs without relation edges")
    if args.distilled:
        from litreview.extract import DistilledExtractor
        distilled = DistilledExtractor(args.distilled)

    if not args.score_only:
        print("provider probe:", json.dumps(probe(), indent=1))
        gen = generate(bench, args.systems, cache, verifier=verifier, classifier=classifier,
                       distilled=distilled, progress=lambda js: tqdm(js, desc="generate"))
        print("generation:", json.dumps(gen, indent=1))

    rep = score(bench, cache, nli=verifier.nli if verifier else None,
                with_bertscore=not args.no_bertscore,
                progress=lambda rs: tqdm(rs, desc="score"))
    rep["benchmark"] = {"path": str(bench_path), "split": args.split,
                        "n_examples": len(bench), "source": args.benchmark}
    rep["settings"] = SETTINGS.describe()
    rep["main_table"] = table(rep["aggregates"])
    rep["cost_table"] = cost_table(rep["aggregates"], usage_summary())
    out = save(rep, P.results / f"eval_{tag}.json")

    print("\n== main table ==")
    import pandas as pd
    print(pd.DataFrame(rep["main_table"]).to_string(index=False))
    print("\n== cost / latency ==")
    print(pd.DataFrame(rep["cost_table"]).to_string(index=False))
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
