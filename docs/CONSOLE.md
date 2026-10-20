# The console

A local web app for the project. It shows what the pipeline has produced, lets
you read a generated review against the human-written one it is imitating, and
starts pipeline runs.

```bash
.venv/bin/python app/server.py
# litreview console on http://127.0.0.1:8000
```

Then open <http://127.0.0.1:8000>. That is the whole setup — no build step, no
npm, no extra packages. The server is Python's standard library, the front end is
one HTML file, one stylesheet and one script.

It binds to `127.0.0.1`, so it is reachable only from your machine. Use
`--port 9000` if 8000 is taken.

## What each tab is for

### Overview
The eight pipeline stages, each marked done or pending, with a couple of real
numbers from the results file it wrote. A stage with no results file says
**pending** rather than showing an estimate — the same rule the report follows.
Below that: corpus totals and the mining yield, so you can see how 200 listed
papers became 80 benchmark examples.

### Benchmark
Every mined example. Click a row and you get the thing this project is really
about:

* **the human related-work section** the paper's author wrote, which is hidden
  from every system and used only for scoring, and
* **the papers it cites** — the abstracts, and only the abstracts, that a system
  is given to work from. Greyed-out entries are references whose abstract could
  not be retrieved.

The `Section` column tells you where the related work was found. Most are a
`\section`; the ones marked `subsection` or `paragraph` are papers the parser
recovers that a naive implementation drops.

### Relations
The seven citation-relation classes, each with its definition, how often the
labeller assigned it, and a real labelled example with the cited paper marked
`[[CITED]]`. The split that matters is `compares-against` (running a baseline —
a measurement) against `contrasts` (saying the cited work is wrong — a
disagreement). Only the second is evidence that two papers conflict, and the
contradiction map depends on telling them apart.

A class the labeller never assigned is dimmed and says so, because that is a
finding, not a blank.

### Grounded review
The point of the system. Each sentence is underlined by its verdict — green
supported, orange partially, red unsupported — and **clicking a sentence shows
the passage it was checked against and the entailment probability the Verifier
gave it**. A sentence with no supporting passage is flagged rather than quietly
deleted.

Switch to **Side by side** to put the generated section next to the one the
paper's author actually wrote, for the same cited papers.

Below: the contradictions the Adjudicator explained, and the gaps the Analyst
argued for and against.

### Cost & usage
Tokens, latency and cost for every model call the project has made, broken down
by pipeline stage. This is the same log the report's cost table is built from, so
the price of the agent pipeline sits beside its quality.

### Run
Starts the project's own scripts. Output streams below, and because every stage
caches, a run you interrupt resumes next time.

| Command | What it does | Needs |
|---|---|---|
| **Demo** | Regenerates one benchmark paper's related-work section with the full crew, then saves it for the Grounded review tab | an LLM key |
| **Run the full Flow** | Collects a corpus on a topic, extracts, verifies, maps, finds gaps, writes and verifies | an LLM key **and** OpenAlex budget |
| **Extract and verify** | Extracts records from mined papers and verifies every claim | an LLM key |
| **Regenerate figures** | Rebuilds `figures/` from `results/` | nothing |
| **Rebuild the report** | Rebuilds the .docx from `results/` and `figures/` | nothing |

Only these five can be started from the browser, with validated arguments.
Nothing else is runnable through the app.

## The quickest way to see what this project does

1. Start the server and open **Benchmark**. Click any paper. Read the human
   section, then look at how little the system is given.
2. Go to **Run**, pick **Demo**, leave the fields alone, press Start. It takes a
   few minutes — most of it waiting on the free tier's rate limit.
3. Go to **Grounded review**. Click sentences and watch the evidence change.
   Then press **Side by side**.

That sequence is the project's argument in about five minutes.

## If something is empty

Every panel degrades to a sentence explaining which stage has not run and what to
run. Nothing is ever invented to fill a gap. The usual causes:

* **No benchmark examples** — Stage 1 has not run: `python scripts/run_stage1.py`.
* **No labels** — Stage 2 has not run: `python scripts/run_stage2.py label`.
* **No review** — run the **Demo** command.
* **Provider says "server unreachable"** — the server stopped; restart it.

## Using it from Claude instead

`app/mcp_server.py` exposes the same views as tools over MCP, so you can ask
Claude about the project's state in conversation rather than reading the
dashboard. See [MCP.md](MCP.md).
