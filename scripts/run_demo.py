#!/usr/bin/env python
"""Run the full crew on one benchmark example and save it for the console.

This is the demonstration the whole project builds towards: take a real paper,
hide its related-work section, hand the crew only the abstracts that section
cites, and put what the crew writes beside what the author wrote -- with every
generated sentence carrying the evidence it rests on.

It needs no corpus search, so it works when OpenAlex's daily budget is spent.

    python scripts/run_demo.py                       # first test-split example
    python scripts/run_demo.py --arxiv-id 2609.00491
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litreview.config import SETTINGS, paths            # noqa: E402
from litreview.eval.runner import load_benchmark        # noqa: E402
from litreview.eval.systems import run_system, topic_of  # noqa: E402
from litreview.llm import install_call_logger, tagged, usage_summary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv-id", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--system", default="S3", choices=["S1", "S2", "S3", "S3_no_relations"])
    ap.add_argument("--max-cited", type=int, default=14,
                    help="cap the cited papers, to keep one demo inside the token budget")
    args = ap.parse_args()

    P = paths().mkdirs()
    install_call_logger(P.results / "llm_calls.jsonl")
    bench = load_benchmark(P.data / "rw_pairs.jsonl", split=args.split)
    if not bench:
        print(f"no benchmark examples in the {args.split} split; run stage 1 first")
        return 1
    pair = next((b for b in bench if b["arxiv_id"] == args.arxiv_id), None) if args.arxiv_id \
        else min(bench, key=lambda b: len(b.get("cited") or []))
    if pair is None:
        print(f"{args.arxiv_id} is not in the {args.split} split")
        return 1
    if args.max_cited:
        with_abs = [c for c in pair["cited"] if c.get("abstract")]
        pair = {**pair, "cited": with_abs[: args.max_cited]}

    print(f"demo on {pair['arxiv_id']}: {(pair.get('title') or '')[:70]}")
    print(f"  {len(pair['cited'])} cited papers in, "
          f"{len((pair.get('rw_text') or '').split())} words of human section hidden")

    verifier = classifier = None
    from litreview.verify import Verifier
    verifier = Verifier()
    md = P.models / "scibert_relations"
    if md.exists():
        from litreview.relations.scibert import RelationClassifier
        classifier = RelationClassifier(md)
    else:
        print("  (no relation classifier yet; running without typed edges)")

    t0 = time.time()
    with tagged("demo"):
        out = run_system(args.system, pair, verifier=verifier, classifier=classifier)
    print(f"  generated in {time.time() - t0:.0f}s")

    rev = out.get("review") or {}
    records = out.get("records") or []
    ev_by_paper = out.get("evidence_by_paper") or {}

    # sentence-level verdicts, the thing the console renders
    evidence = {f"{r['paper_id']}:{c['claim_id']}": c["evidence"]["passage"]
                for r in records for c in (r.get("claims") or [])}
    sents = rev.get("sentences") or []
    verdicts = verifier.verify_sentences(sents, evidence, ev_by_paper)
    faith = verifier.faithfulness(verdicts)

    d = P.results / "flow_runs" / f"demo_{pair['arxiv_id'].replace('/', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "review.md").write_text(rev.get("markdown") or "")
    (d / "records.json").write_text(json.dumps(records, indent=2, default=str))
    (d / "review_verdicts.json").write_text(json.dumps(
        {"after_revision": faith,
         "sentence_verdicts": [v.model_dump() for v in verdicts]}, indent=2, default=str))
    (d / "reference.json").write_text(json.dumps(
        {"arxiv_id": pair["arxiv_id"], "title": pair.get("title"),
         "rw_text": pair.get("rw_text"),
         "cited": [{"cite_key": c.get("cite_key"), "title": c.get("title")}
                   for c in pair["cited"]]}, indent=2))
    (d / "summary.json").write_text(json.dumps({
        "topic": topic_of(pair)[:160], "demo_of": pair["arxiv_id"], "system": args.system,
        "settings": SETTINGS.describe(),
        "n_records": len(records),
        "n_claims": sum(len(r.get("claims") or []) for r in records),
        "n_contradictions": out.get("n_contradictions"),
        "relation_counts": out.get("relation_counts"),
        "faithfulness": {"after_revision": faith},
        "wall_s": out.get("wall_s"), "notes": out.get("notes"),
        "usage": usage_summary(),
    }, indent=2, default=str))

    print(f"  {len(sents)} sentences, faithfulness {faith['faithfulness']:.0%} "
          f"({faith['supported']} supported, {faith['unsupported']} unsupported)")
    print(f"  wrote {d}")
    print("  open the console's Grounded review tab to read it against the human section")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
