#!/usr/bin/env python
"""MCP server exposing the literature-review project to Claude.

Lets you ask about the project in conversation instead of reading the dashboard:
what has run, what the benchmark looks like, which relations the labeller
assigned, whether a generated review's sentences survived verification, and what
it all cost.

Read-only. It serves the same views as the web console, through the same
:mod:`litreview.service` module, so Claude, the console and the report cannot
disagree about a number. Nothing here starts a run or changes a file.

Register it with Claude Desktop (see docs/MCP.md):

    {"mcpServers": {"litreview": {
        "command": "/ABSOLUTE/PATH/.venv/bin/python",
        "args": ["/ABSOLUTE/PATH/app/mcp_server.py"]}}}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mcp.server.fastmcp import FastMCP          # noqa: E402

from litreview import service as S              # noqa: E402

mcp = FastMCP("litreview")


def _j(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


@mcp.tool()
def pipeline_status() -> str:
    """Which pipeline stages have run, with headline numbers for each.

    Start here. A stage reported as not done has no results file, which means it
    genuinely has not been run -- not that the number is missing.
    """
    return _j(S.pipeline_status())


@mcp.tool()
def dataset_overview() -> str:
    """Sizes and descriptive statistics of the mined datasets.

    Covers the benchmark examples, the citation contexts, the labelled subset and
    the external Multi-XScience benchmark, plus the time-split check that the
    train and test windows do not overlap.
    """
    return _j(S.dataset_summary())


@mcp.tool()
def list_benchmark(limit: int = 20, split: str = "") -> str:
    """List mined benchmark examples. `split` is 'train', 'test' or empty for both.

    Each example is a real paper whose related-work section is hidden and whose
    cited papers' abstracts are the only input a system gets.
    """
    return _j(S.benchmark_examples(limit=limit, split=split or None))


@mcp.tool()
def get_benchmark_example(arxiv_id: str) -> str:
    """One benchmark example in full: the hidden human related-work section and
    every paper it cites, with abstracts where they could be retrieved."""
    r = S.benchmark_example(arxiv_id)
    return _j(r) if r else f"No benchmark example with arXiv id {arxiv_id!r}."


@mcp.tool()
def relation_labels() -> str:
    """The seven citation-relation classes: definition, how often the labeller
    assigned each, and real labelled examples.

    The distinction that matters downstream is `compares-against` (a measurement
    against a baseline) versus `contrasts` (an asserted disagreement); only the
    latter is evidence that two papers conflict.
    """
    return _j(S.label_summary())


@mcp.tool()
def grounded_review(run: str = "") -> str:
    """A generated review with, for each sentence, its verdict, the entailment
    probability and the passage it was checked against.

    Also returns the contradictions and gaps from that run, and the human
    reference section when the run was a benchmark demo. Leave `run` empty for
    the most recent.
    """
    r = S.review(run or None)
    if not r:
        return ("No review has been generated yet. Run `python scripts/run_demo.py` "
                "to regenerate one benchmark paper's related-work section.")
    return _j(r)


@mcp.tool()
def list_runs() -> str:
    """Every completed Flow or demo run available to inspect."""
    return _j({"runs": S.runs()})


@mcp.tool()
def usage_and_cost() -> str:
    """Tokens, latency and cost for every model call the project has made,
    broken down by pipeline stage and by model."""
    return _j(S.usage())


@mcp.tool()
def verified_references() -> str:
    """The references confirmed to exist via OpenAlex, Semantic Scholar or arXiv.

    The report cites only these; anything a source could not confirm was dropped
    rather than cited.
    """
    return _j(S.references())


@mcp.tool()
def report_status() -> str:
    """Whether the report has been built, and what it still needs from a human:
    unresolved placeholders, missing figures and stages that could not be
    reported because they have not run."""
    return _j(S.report_status())


if __name__ == "__main__":
    mcp.run()
