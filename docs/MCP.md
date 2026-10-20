# Using the project from Claude (MCP connector)

`app/mcp_server.py` exposes the project to Claude as a set of tools, so you can
ask about it in conversation instead of reading the dashboard:

> *"What's the state of my literature-review pipeline?"*
> *"Show me a benchmark example and the section the author actually wrote."*
> *"Which citation relation does the labeller never assign?"*
> *"Did the generated review's sentences survive verification?"*
> *"What has this cost in tokens so far?"*

It is **read-only**. It reports what the pipeline has produced; it cannot start a
run, change a file, or spend your API budget. Use the console's Run tab, or the
command line, for anything that writes.

## Setup

1. Find your absolute path:

```bash
cd ~/Claude/litreview && pwd
```

2. Open Claude Desktop → **Settings → Developer → Edit Config**. That opens
   `claude_desktop_config.json`. Add the `litreview` entry, replacing
   `/ABSOLUTE/PATH` with what `pwd` printed:

```json
{
  "mcpServers": {
    "litreview": {
      "command": "/ABSOLUTE/PATH/.venv/bin/python",
      "args": ["/ABSOLUTE/PATH/app/mcp_server.py"]
    }
  }
}
```

If you already have other servers, add `litreview` alongside them rather than
replacing the file.

3. Quit Claude Desktop completely and reopen it. The tools appear under the
   connector icon in the message box.

Use the venv's Python (`.venv/bin/python`), not the system one — the system
Python does not have the project's dependencies.

## The tools

| Tool | Answers |
|---|---|
| `pipeline_status` | Which of the eight stages have run, with headline numbers. **Start here.** |
| `dataset_overview` | Sizes and statistics of the mined datasets, and the train/test time-split check |
| `list_benchmark` | Mined benchmark examples, filterable by split |
| `get_benchmark_example` | One example in full: the hidden human section and every cited paper |
| `relation_labels` | The seven relation classes, their definitions, counts and real examples |
| `grounded_review` | A generated review with each sentence's verdict, entailment score and evidence |
| `list_runs` | Completed Flow and demo runs |
| `usage_and_cost` | Tokens, latency and cost per pipeline stage |
| `verified_references` | References confirmed to exist, which are the only ones the report cites |
| `report_status` | Whether the report built, and what it still needs from you |

## A good first conversation

> **You:** What's the state of my literature review pipeline, and what should I run next?

Claude calls `pipeline_status`, sees which stages are missing, and tells you
which script to run. Then:

> **You:** Pick a benchmark example and show me what the author wrote versus what the system has to work from.

`get_benchmark_example` returns both, so Claude can lay the human section beside
the abstracts and point out what the system has to infer.

> **You:** Of the sentences the crew generated, which ones failed verification and why?

`grounded_review` returns per-sentence verdicts with the closest supporting
passage, so Claude can show you the sentence, the evidence, and the gap between
them.

## If the tools do not appear

* **Server not listed** — the paths in the config must be absolute; `~` does not
  expand there.
* **Server errors on start** — run it by hand and read the error:
  `/ABSOLUTE/PATH/.venv/bin/python /ABSOLUTE/PATH/app/mcp_server.py`. It should
  sit silently waiting on stdin; a traceback means a missing dependency.
* **Tools return "not run yet"** — that is the honest answer, not a bug. Run the
  stage it names.

## Console or connector?

They serve the same data and neither is authoritative over the other.

* The **console** is better for looking: the sentence-by-sentence evidence
  reader, the side-by-side comparison and the yield chart are visual.
* The **connector** is better for asking: anything that starts "why", "which" or
  "compare", and anything where you want the answer written up rather than
  displayed.
