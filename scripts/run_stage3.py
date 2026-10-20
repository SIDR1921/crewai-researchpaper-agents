#!/usr/bin/env python
"""Stage 3: extract records from papers, calibrate the Verifier, verify everything.

    python scripts/run_stage3.py extract   --n 20
    python scripts/run_stage3.py calibrate                 # needs data/verifier_dev.jsonl
    python scripts/run_stage3.py verify
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm.auto import tqdm

from litreview.config import paths
from litreview.extract import ApiExtractor, CrewExtractor, extract_corpus, load_records
from litreview.llm import probe, usage_summary


def papers_from_mining(n: int, split: str = "train", seed: int = 0) -> list[dict]:
    """Papers with LaTeX sections available, built from the Stage 1 cache."""
    from litreview.mining import arxiv, latex_mining as lm
    P = paths()
    meta = [json.loads(l) for l in (P.data / f"papers_{split}.jsonl").open()]
    random.Random(seed).shuffle(meta)
    out = []
    for p in meta:
        files = arxiv.fetch_source(p["arxiv_id"], root=P.sources)
        if not files:
            continue
        try:
            mined = lm.mine_paper(files)
        except Exception:
            continue
        if not mined or not mined.get("sections"):
            continue
        out.append({"paper_id": p["arxiv_id"], "title": p["title"],
                    "year": int(p["published"][:4]), "abstract": p["abstract"],
                    "published": p["published"], "sections": mined["sections"]})
        if len(out) >= n:
            break
    return out


def cmd_extract(args) -> int:
    P = paths().mkdirs()
    print("provider probe:", json.dumps(probe(), indent=1))
    papers = papers_from_mining(args.n, args.split, args.seed)
    print(f"{len(papers)} papers with usable sections")
    ex = CrewExtractor() if args.backend == "crew" else ApiExtractor()
    cache = P.data / f"extractions_{args.split}_{ex.name}.jsonl"
    stats = extract_corpus(papers, cache, extractor=ex, progress=lambda ps: tqdm(ps, desc="extract"))
    stats |= {"backend": ex.name, "cache": str(cache), "usage": usage_summary()}
    (P.results / f"stage3_extract_{args.split}_{ex.name}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "usage"}, indent=2))
    return 0


def cmd_make_dev(args) -> int:
    """Sample (claim, evidence, source) triples for the human verifier dev set."""
    P = paths()
    rows = load_records(P.data / f"extractions_{args.split}_api.jsonl")
    items = []
    for pid, row in rows.items():
        src = (row.get("meta") or {}).get("input_text") or ""
        for c in (row["record"].get("claims") or []):
            items.append({"paper_id": pid, "claim_id": c["claim_id"], "claim": c["text"],
                          "passage": c["evidence"]["passage"], "section": c["evidence"]["section"],
                          "source": src, "human": None})
    random.Random(args.seed).shuffle(items)
    items = items[: args.n]
    out = P.data / "verifier_dev.jsonl"
    with out.open("w") as f:
        for i, it in enumerate(items, 1):
            f.write(json.dumps({"n": i, **it}) + "\n")
    print(f"wrote {len(items)} items to {out} (fill in \"human\": supported|partially|unsupported)")
    return 0


def cmd_calibrate(args) -> int:
    from litreview.verify import NLI, calibrate
    P = paths()
    path = P.data / "verifier_dev.jsonl"
    if not path.exists():
        print(f"{path} missing; run `make-dev` and label it")
        return 1
    items = [json.loads(l) for l in path.open()]
    items = [i for i in items if i.get("human") in ("supported", "partially", "unsupported")]
    if len(items) < 15:
        print(f"only {len(items)} labelled dev items; label more")
        return 1
    rep = calibrate(items, NLI(), P.results / "verifier_calibration.json")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("grid", "per_item")}, indent=2))
    return 0


def cmd_verify(args) -> int:
    from litreview.verify import Verifier
    P = paths()
    cache = P.data / f"extractions_{args.split}_{args.backend}.jsonl"
    rows = load_records(cache)
    if not rows:
        print(f"no extractions in {cache}")
        return 1
    v = Verifier()
    rep = v.verify_corpus(rows, P.data / f"claim_verdicts_{args.split}_{args.backend}.json")
    (P.results / f"stage3_verify_{args.split}_{args.backend}.json").write_text(
        json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("extract", cmd_extract), ("make-dev", cmd_make_dev),
                     ("calibrate", cmd_calibrate), ("verify", cmd_verify)):
        s = sub.add_parser(name)
        s.add_argument("--n", type=int, default=20)
        s.add_argument("--split", default="train")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--backend", default="api", choices=["api", "crew"])
        s.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
