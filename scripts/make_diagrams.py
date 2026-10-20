#!/usr/bin/env python
"""The three structural diagrams for Chapter 3, drawn with matplotlib.

Graphviz's `dot` binary is not available in this environment, so the diagrams are
drawn directly rather than shelled out to. They are generated, not hand-drawn, so
they stay in step with the code: the node labels below are the actual module and
Flow-step names.

    python scripts/make_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from litreview.config import paths

FIG = paths().figures
FIG.mkdir(parents=True, exist_ok=True)

INK = "#1f2933"
AGENT = "#dce9f5"
AGENT_E = "#2f6fa8"
MODEL = "#e6f2e6"
MODEL_E = "#3f8f5f"
DATA = "#fdf1dc"
DATA_E = "#c88b2a"
GATE = "#fae3e3"
GATE_E = "#c0504d"
plt.rcParams.update({"figure.dpi": 220, "savefig.dpi": 220, "font.size": 8,
                     "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, fc=AGENT, ec=AGENT_E, fs=8, bold=False, radius=0.035):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}",
                                linewidth=1.0, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, a, b, side="right", label="", style="-|>", ls="-", rad=0.0, colour=INK, fs=6.8):
    def port(bx, where):
        x, y, w, h = bx
        return {"right": (x + w, y + h / 2), "left": (x, y + h / 2),
                "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}[where]
    opposite = {"right": "left", "left": "right", "top": "bottom", "bottom": "top"}
    p1, p2 = port(a, side), port(b, opposite[side])
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, linestyle=ls, linewidth=0.9,
                                 color=colour, mutation_scale=9,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.012, label, ha="center", va="bottom", fontsize=fs, color=colour)


def legend(ax, items, x=0.01, y=0.005, fs=6.8):
    for i, (label, fc, ec) in enumerate(items):
        ax.add_patch(FancyBboxPatch((x + i * 0.195, y), 0.022, 0.022,
                                    boxstyle="round,pad=0.004,rounding_size=0.006",
                                    linewidth=0.8, edgecolor=ec, facecolor=fc))
        ax.text(x + i * 0.195 + 0.03, y + 0.011, label, va="center", fontsize=fs, color=INK)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    out = FIG / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------- architecture

def architecture():
    fig, ax = canvas(7.8, 4.4)
    ax.text(0.5, 0.985, "System architecture", ha="center", fontsize=10.5,
            fontweight="bold", color=INK)

    # column 1: data
    src = box(ax, 0.015, 0.775, 0.155, 0.125,
              "arXiv API\nOpenAlex\nSemantic Scholar", DATA, DATA_E, fs=7.4, bold=True)
    mine = box(ax, 0.015, 0.605, 0.155, 0.105,
               "mining.latex_mining\nmining.pipeline", MODEL, MODEL_E, fs=7.0)
    ds = box(ax, 0.015, 0.415, 0.155, 0.125,
             "citation_contexts\nrw_pairs\n(+ Multi-XScience)", DATA, DATA_E, fs=7.0)
    dis = box(ax, 0.015, 0.205, 0.155, 0.125,
              "Distilled extractor\nQLoRA (component B)\nswaps in for the Extractor",
              MODEL, MODEL_E, fs=6.6)

    # column 2: the grounded core
    lib = box(ax, 0.225, 0.790, 0.175, 0.095, "Librarian\ncorpus.py", fs=7.6)
    gate = box(ax, 0.225, 0.645, 0.175, 0.085, "Human approval\ncheckpoint",
               GATE, GATE_E, fs=7.4, bold=True)
    ext = box(ax, 0.225, 0.480, 0.175, 0.105,
              "Extractor\nextract.py\noutput_pydantic", fs=7.4)
    ver = box(ax, 0.225, 0.300, 0.175, 0.115,
              "Verifier\nverify.py\nquote + NLI", GATE, GATE_E, fs=7.4)

    # column 3: synthesis
    clf = box(ax, 0.455, 0.645, 0.175, 0.115,
              "Relation classifier\nSciBERT, 7 classes\n(component A)", MODEL, MODEL_E, fs=7.0)
    syn = box(ax, 0.455, 0.455, 0.175, 0.115,
              "Synthesist /\nAdjudicator\nsynthesis.py", fs=7.4)
    gap = box(ax, 0.455, 0.280, 0.175, 0.105, "Gap Analyst\ngaps.py", fs=7.6)

    # column 4: writing
    wri = box(ax, 0.685, 0.560, 0.150, 0.100, "Writer\nreview.py", fs=7.6)
    rev = box(ax, 0.685, 0.380, 0.150, 0.105,
              "verify_review\nrevise once", GATE, GATE_E, fs=7.4)

    # column 5: exports
    out = box(ax, 0.865, 0.400, 0.120, 0.235,
              "review.md\nrecords.json\ncitation_graph\n.html\ngaps.csv\ncontradictions\n.csv",
              DATA, DATA_E, fs=6.6)

    arrow(ax, src, mine, "bottom")
    arrow(ax, mine, ds, "bottom")
    arrow(ax, ds, dis, "bottom")
    arrow(ax, src, lib, "right", rad=0.0)
    arrow(ax, lib, gate, "bottom")
    arrow(ax, gate, ext, "bottom")
    arrow(ax, ext, ver, "bottom")
    arrow(ax, dis, ext, "right", rad=0.0, ls="--")
    arrow(ax, ver, syn, "right", rad=0.08)
    arrow(ax, clf, syn, "bottom")
    arrow(ax, syn, gap, "bottom")
    arrow(ax, syn, wri, "right", rad=0.06)
    arrow(ax, gap, wri, "right", rad=-0.30)
    arrow(ax, wri, rev, "bottom")
    arrow(ax, rev, out, "right", rad=0.10)

    # off-box annotations, placed in empty space
    ax.text(0.0925, 0.372, "mined training data \u2192 components A and B",
            fontsize=6.0, color=MODEL_E, ha="center", va="center")
    ax.text(0.760, 0.345, "unsupported sentences\ngo back once", fontsize=6.3,
            color=GATE_E, ha="center", va="top")
    ax.add_patch(FancyArrowPatch((0.760, 0.380), (0.760, 0.560), arrowstyle="-|>",
                                 linestyle="--", linewidth=0.9, color=GATE_E,
                                 mutation_scale=9,
                                 connectionstyle="arc3,rad=0.9", shrinkA=1, shrinkB=1))

    # one LLM config, as a footer rather than as arrows into every agent
    box(ax, 0.015, 0.080, 0.970, 0.070,
        "llm.py \u2014 one configuration for every agent and every direct call: CrewAI LLM over "
        "LiteLLM (Groq / Gemini / local Ollama),\na token-bucket rate limiter for free tiers, and "
        "a callback that logs tokens, latency and cost for every call to results/llm_calls.jsonl",
        "#f3f4f6", "#9aa5b1", fs=7.0)

    legend(ax, [("CrewAI agent", AGENT, AGENT_E), ("fine-tuned model", MODEL, MODEL_E),
                ("data / output", DATA, DATA_E), ("verification gate", GATE, GATE_E)],
           y=0.005)
    save(fig, "fig_architecture.png")


# ------------------------------------------------------------------- flow

def flow_diagram():
    fig, ax = canvas(7.6, 3.1)
    ax.text(0.5, 0.965, "CrewAI Flow: ten steps with one human gate and one revision loop",
            ha="center", fontsize=9.5, fontweight="bold", color=INK)
    steps = [
        ("1 collect_corpus", AGENT, AGENT_E), ("2 approve", GATE, GATE_E),
        ("3 extract", AGENT, AGENT_E), ("4 verify_claims", GATE, GATE_E),
        ("5 classify_relations", MODEL, MODEL_E), ("6 build_map", AGENT, AGENT_E),
        ("7 find_gaps", AGENT, AGENT_E), ("8 write_review", AGENT, AGENT_E),
        ("9 verify_review", GATE, GATE_E), ("10 export", DATA, DATA_E),
    ]
    w, h = 0.168, 0.155
    xs = [0.015, 0.212, 0.409, 0.606, 0.803]
    rows_y = [0.640, 0.330]
    boxes = []
    for i, (label, fc, ec) in enumerate(steps):
        row, col = divmod(i, 5)
        boxes.append(box(ax, xs[col], rows_y[row], w, h, label, fc, ec, fs=7.4))
    for i in list(range(4)) + list(range(5, 9)):
        arrow(ax, boxes[i], boxes[i + 1], "right")

    # wrap from 5 to 6 through a clear lane between the rows
    lane = (rows_y[0] + rows_y[1] + h) / 2
    x5, x6 = xs[4] + w / 2, xs[0] + w / 2
    ax.plot([x5, x5, x6, x6], [rows_y[0], lane, lane, rows_y[1] + h],
            color=INK, linewidth=0.9, solid_capstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch((x6, lane - 0.004), (x6, rows_y[1] + h),
                                 arrowstyle="-|>", linewidth=0.9, color=INK,
                                 mutation_scale=9, shrinkA=0, shrinkB=1))

    # revision loop, routed below the second row
    under = rows_y[1] - 0.105
    x9, x8 = xs[3] + w / 2, xs[2] + w / 2
    ax.plot([x9, x9, x8, x8], [rows_y[1], under, under, rows_y[1]],
            color=GATE_E, linewidth=0.9, linestyle="--", zorder=1)
    ax.add_patch(FancyArrowPatch((x8, under + 0.004), (x8, rows_y[1]),
                                 arrowstyle="-|>", linestyle="--", linewidth=0.9,
                                 color=GATE_E, mutation_scale=9, shrinkA=0, shrinkB=1))
    ax.text((x8 + x9) / 2, under - 0.018,
            "any unsupported sentence goes back to the Writer, once",
            ha="center", va="top", fontsize=6.6, color=GATE_E)

    ax.text(xs[1] + w / 2, rows_y[0] + h + 0.030,
            "pipeline stops here until a human approves",
            ha="center", va="bottom", fontsize=6.6, color=GATE_E)
    ax.text(0.015, 0.075,
            "Every step writes its output to the run directory, so a Flow interrupted at any "
            "step resumes from its cache on the next kickoff().",
            fontsize=6.8, color="#52606d")
    legend(ax, [("agent step", AGENT, AGENT_E), ("verification / gate", GATE, GATE_E),
                ("fine-tuned model", MODEL, MODEL_E), ("export", DATA, DATA_E)], y=0.005)
    save(fig, "fig_flow.png")


# --------------------------------------------------------------- mining pipeline

def mining_diagram():
    fig, ax = canvas(7.6, 3.3)
    ax.text(0.5, 0.965, "Data-mining pipeline: from arXiv submissions to two supervised datasets",
            ha="center", fontsize=9.5, fontweight="bold", color=INK)
    chain = [
        ("list submissions\ncs.CL + cs.IR\nper time window", DATA, DATA_E),
        ("download LaTeX\nsource\n(e-print tarball)", DATA, DATA_E),
        ("pick main .tex,\ninline \\input,\nstrip comments", MODEL, MODEL_E),
        ("find related work:\nsection, subsection\nor \\paragraph run", MODEL, MODEL_E),
        ("citations \u2192\n[CITE:key]\nmarkers", MODEL, MODEL_E),
        ("cite key \u2192 title\n.bbl (natbib and\nbiblatex), .bib", MODEL, MODEL_E),
        ("title \u2192 OpenAlex\nfuzzy match \u2265 88\nabstract, year", DATA, DATA_E),
    ]
    w, h = 0.158, 0.195
    xs = [0.015, 0.195, 0.375, 0.555]
    rows_y = [0.610, 0.280]
    boxes = []
    for i, (label, fc, ec) in enumerate(chain):
        row, col = divmod(i, 4)
        boxes.append(box(ax, xs[col], rows_y[row], w, h, label, fc, ec, fs=6.5))
    for i in list(range(3)) + list(range(4, 6)):
        arrow(ax, boxes[i], boxes[i + 1], "right")

    lane = (rows_y[0] + rows_y[1] + h) / 2
    x4, x5 = xs[3] + w / 2, xs[0] + w / 2
    ax.plot([x4, x4, x5, x5], [rows_y[0], lane, lane, rows_y[1] + h],
            color=INK, linewidth=0.9, zorder=1)
    ax.add_patch(FancyArrowPatch((x5, lane - 0.004), (x5, rows_y[1] + h), arrowstyle="-|>",
                                 linewidth=0.9, color=INK, mutation_scale=9,
                                 shrinkA=0, shrinkB=1))

    d1 = box(ax, 0.775, 0.520, 0.210, 0.160,
             "citation_contexts.jsonl\none row per\n(citing sentence \u00d7 cited key)",
             DATA, DATA_E, fs=6.4)
    d2 = box(ax, 0.775, 0.280, 0.210, 0.185,
             "rw_pairs.jsonl\nrelated-work text +\ncited abstracts\n(drop < 60% coverage)",
             DATA, DATA_E, fs=6.4)
    arrow(ax, boxes[6], d1, "right", rad=-0.18)
    arrow(ax, boxes[6], d2, "right", rad=0.08)

    ax.text(0.015, 0.195,
            "Every paper's extracted source files are cached as one JSON file, and an empty "
            "cache entry records \u2018no usable LaTeX\u2019, so a PDF-only submission is never "
            "re-downloaded.\nThe OpenAlex cache is keyed by the queried title, so a re-run "
            "resolves nothing twice. Together these are what make the stage resumable after a "
            "Colab disconnection.",
            fontsize=6.6, color="#52606d", va="top", linespacing=1.5)
    legend(ax, [("retrieval, cached", DATA, DATA_E),
                ("parsing (pure functions, unit-tested)", MODEL, MODEL_E)], y=0.005)
    save(fig, "fig_mining_pipeline.png")


def main() -> int:
    architecture()
    flow_diagram()
    mining_diagram()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
