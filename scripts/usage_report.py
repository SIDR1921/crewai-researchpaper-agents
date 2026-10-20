#!/usr/bin/env python
"""Aggregate results/llm_calls.jsonl into results/llm_usage.json.

Every LLM call in the project appends to that log through a LiteLLM callback, so
this is the single source for the token, latency and cost numbers in the report.
"""
from __future__ import annotations

import json

from litreview.config import paths
from litreview.llm import usage_summary


def main() -> int:
    P = paths()
    rep = usage_summary(P.results / "llm_calls.jsonl")
    out = P.results / "llm_usage.json"
    out.write_text(json.dumps(rep, indent=2))
    o = rep["overall"]
    print(f"{o['calls']} calls ({o['failed']} failed), {o['total_tokens']:,} tokens, "
          f"${o['cost_usd']:.4f}, {o['latency_s_total']}s total")
    for tag, v in sorted(rep["by_tag"].items(), key=lambda kv: -kv[1]["total_tokens"]):
        print(f"  {tag:26s} {v['calls']:5d} calls  {v['total_tokens']:9,d} tok  "
              f"{v['latency_s_mean']}s mean")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
