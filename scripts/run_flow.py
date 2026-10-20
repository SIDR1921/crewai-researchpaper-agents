#!/usr/bin/env python
"""Stage 5: run the whole Flow on one topic.

    python scripts/run_flow.py --topic "faithfulness and hallucination evaluation in abstractive summarisation" --n-papers 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from litreview.config import paths
from litreview.flow import LitReviewFlow
from litreview.llm import probe, usage_summary

DEFAULT_TOPIC = "faithfulness and hallucination evaluation in abstractive summarisation"


def interactive_approve(table: list[dict]) -> bool:
    import pandas as pd
    print(pd.DataFrame(table).to_string(index=False))
    return (input("\napprove this corpus? [Y/n] ").strip().lower() or "y").startswith("y")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default=DEFAULT_TOPIC)
    ap.add_argument("--n-papers", type=int, default=15)
    ap.add_argument("--seeds", nargs="*", default=[])
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument("--no-verifier", action="store_true")
    ap.add_argument("--no-relations", action="store_true")
    ap.add_argument("--crewai-flow", action="store_true",
                    help="drive the steps through a real CrewAI Flow object")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    P = paths().mkdirs()
    print("provider probe:", json.dumps(probe(), indent=1))
    clf = None
    if not args.no_relations:
        md = P.models / "scibert_relations"
        if md.exists():
            from litreview.relations.scibert import RelationClassifier
            clf = RelationClassifier(md)
        else:
            print(f"note: no classifier at {md}; relations will be skipped")

    flow = LitReviewFlow(
        topic=args.topic, n_papers=args.n_papers, seeds=args.seeds,
        run_dir=args.run_dir,
        approve=(lambda t: True) if args.auto_approve else interactive_approve,
        classifier=clf, use_relations=not args.no_relations,
        use_verifier=not args.no_verifier,
        progress=lambda xs: tqdm(xs, desc="step"))

    summary = flow.as_crewai_flow().kickoff() if args.crewai_flow else flow.run()
    if not isinstance(summary, dict):
        summary = flow.state.exports
    summary = dict(summary)
    summary["usage"] = usage_summary()
    out = P.results / "flow_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "usage"}, indent=2))
    print("\nwrote", out)
    print("review:", flow.state.exports.get("review_md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
