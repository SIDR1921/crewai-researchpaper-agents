#!/usr/bin/env python
"""Every figure in the report, rendered from ``results/*.json`` only.

Nothing here invents a number: each function opens a results file, and skips
itself (printing why) if that file does not exist. Run after the stages.

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from litreview.config import paths

PALETTE = ["#2f6fa8", "#e1803a", "#4f9d6b", "#9b5fa8", "#c0504d", "#7f7f7f", "#2a9d8f"]
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.autolayout": True, "font.family": "DejaVu Sans",
})

P = paths()
FIG = P.figures
FIG.mkdir(parents=True, exist_ok=True)
MADE: list[dict] = []
SKIPPED: list[dict] = []


def _load(name: str):
    p = P.results / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _save(fig, name: str, caption: str, source: str) -> None:
    out = FIG / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    MADE.append({"file": name, "caption": caption, "source": source})
    print("wrote", out)


def _skip(name: str, why: str) -> None:
    SKIPPED.append({"file": name, "reason": why})
    print(f"skip {name}: {why}")


# ------------------------------------------------------------------ Stage 1

def fig_yield():
    d = _load("stage1_yield.json")
    if not d:
        return _skip("fig_stage1_yield.png", "results/stage1_yield.json missing")
    y = d["yield"]
    ds = d["datasets"]
    steps = [("listed", y["listed"]),
             ("LaTeX source", y["listed"] - y["no_source"]),
             ("parsed", y["parsed"]),
             ("related work found", y["with_related_work"]),
             ("kept in rw_pairs", ds["rw_pairs_kept"])]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    names = [s[0] for s in steps]
    vals = [s[1] for s in steps]
    bars = ax.barh(range(len(steps))[::-1], vals, color=PALETTE[0], height=0.6)
    ax.set_yticks(range(len(steps))[::-1], names)
    ax.set_xlabel("papers")
    for b, v in zip(bars, vals):
        pct = 100 * v / max(1, vals[0])
        ax.text(b.get_width() + max(vals) * 0.015, b.get_y() + b.get_height() / 2,
                f"{v} ({pct:.0f}%)", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.25)
    _save(fig, "fig_stage1_yield.png",
          "Stage 1 mining yield: papers surviving each step of the pipeline.",
          "results/stage1_yield.json")


def fig_dataset_stats():
    d = _load("stage1_dataset_stats.json")
    if not d:
        return _skip("fig_dataset_stats.png", "results/stage1_dataset_stats.json missing")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
    for ax, key, xlabel in zip(
            axes, ("rw_words", "n_cited", "abstract_coverage"),
            ("words in related-work section", "papers cited", "abstract coverage")):
        hist = (d.get("histograms") or {}).get(key)
        if not hist:
            ax.set_visible(False)
            continue
        ax.bar(hist["centers"], hist["counts"], width=np.diff(hist["edges"]).mean() * 0.9,
               color=PALETTE[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("papers" if key == "rw_words" else "")
    _save(fig, "fig_dataset_stats.png",
          "Distribution of related-work length, number of cited papers and abstract coverage.",
          "results/stage1_dataset_stats.json")


# ------------------------------------------------------------------ Stage 2

def _confusion(ax, cm, labels, title):
    cm = np.array(cm, dtype=float)
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.55 else "#222")
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title, fontsize=8.5)
    ax.grid(False)
    return im


def fig_confusion():
    bench = _load("stage2_benchmark.json")
    agree = _load("stage2_gold_agreement.json")
    panels = []
    if agree and agree.get("llm_vs_gold", {}).get("confusion"):
        panels.append(("LLM labeller vs human gold", agree["llm_vs_gold"]))
    if bench and bench.get("scibert", {}).get("confusion"):
        panels.append(("Fine-tuned SciBERT vs human gold", bench["scibert"]))
    if not panels:
        return _skip("fig_relation_confusion.png",
                     "no confusion matrix in results/stage2_*.json")
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4))
    axes = np.atleast_1d(axes)
    for ax, (title, m) in zip(axes, panels):
        _confusion(ax, m["confusion"], m["labels"], f"{title}\nmacro-F1 {m['macro_f1']:.3f}")
    _save(fig, "fig_relation_confusion.png",
          "Confusion matrices on the human gold set, row-normalised; cell text is the count.",
          "results/stage2_benchmark.json, results/stage2_gold_agreement.json")


def fig_relation_models():
    d = _load("stage2_benchmark.json")
    if not d:
        return _skip("fig_relation_models.png", "results/stage2_benchmark.json missing")
    order = [("majority", "majority class"), ("tfidf_lr", "TF-IDF + LR"),
             ("zero_shot_llm", "zero-shot LLM"), ("scibert", "fine-tuned SciBERT")]
    rows = [(lab, d[k]["accuracy"], d[k]["macro_f1"]) for k, lab in order if k in d]
    if not rows:
        return _skip("fig_relation_models.png", "no systems scored")
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    x = np.arange(len(rows))
    ax.bar(x - 0.19, [r[1] for r in rows], 0.36, label="accuracy", color=PALETTE[0])
    ax.bar(x + 0.19, [r[2] for r in rows], 0.36, label="macro-F1", color=PALETTE[1])
    for i, r in enumerate(rows):
        ax.text(i - 0.19, r[1] + 0.012, f"{r[1]:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.19, r[2] + 0.012, f"{r[2]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score on human gold set")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig_relation_models.png",
          "Citation-relation classification on the human gold set: baselines vs fine-tuned SciBERT.",
          "results/stage2_benchmark.json")


def fig_label_distribution():
    d = _load("stage2_labelling.json")
    if not d:
        return _skip("fig_label_distribution.png", "results/stage2_labelling.json missing")
    dist = d.get("label_distribution") or {}
    if not any(dist.values()):
        return _skip("fig_label_distribution.png", "empty label distribution")
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    keys = list(dist)
    ax.bar(keys, [dist[k] for k in keys], color=PALETTE[2])
    for i, k in enumerate(keys):
        ax.text(i, dist[k] + max(dist.values()) * 0.02, str(dist[k]), ha="center", fontsize=7)
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("contexts")
    _save(fig, "fig_label_distribution.png",
          "Distribution of LLM-assigned citation relations over the labelled pool.",
          "results/stage2_labelling.json")


# ------------------------------------------------------------------ Stage 3

def fig_verifier_calibration():
    d = _load("verifier_calibration.json")
    if not d:
        return _skip("fig_verifier_calibration.png", "results/verifier_calibration.json missing")
    items = d.get("per_item") or []
    if not items:
        return _skip("fig_verifier_calibration.png", "no per-item scores")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    colours = {"supported": PALETTE[2], "partially": PALETTE[1], "unsupported": PALETTE[4]}
    for kind, c in colours.items():
        xs = [i["quote_score"] for i in items if i["human"] == kind]
        ys = [i["entail_prob"] for i in items if i["human"] == kind]
        if xs:
            ax1.scatter(xs, ys, s=16, alpha=0.8, color=c, label=kind, edgecolor="none")
    th = d["thresholds"]
    ax1.axvline(th["quote_min"], ls="--", lw=0.9, color="#555")
    ax1.axhline(th["entail_supported"], ls="--", lw=0.9, color="#555")
    ax1.axhline(th["entail_partial"], ls=":", lw=0.9, color="#888")
    ax1.set_xlabel("quote match score")
    ax1.set_ylabel("P(entailment)")
    ax1.legend(frameon=False, fontsize=7)
    ax1.set_title("hand-checked dev items vs fitted thresholds", fontsize=8.5)

    grid = d.get("grid") or []
    if grid:
        ax2.plot(range(1, len(grid) + 1), [g["macro_f1"] for g in grid], marker="o", ms=3,
                 color=PALETTE[0], label="macro-F1")
        ax2.plot(range(1, len(grid) + 1), [g["accuracy"] for g in grid], marker="s", ms=3,
                 color=PALETTE[1], label="accuracy")
        ax2.set_xlabel("threshold setting (best first)")
        ax2.set_ylabel("score on dev set")
        ax2.legend(frameon=False, fontsize=7)
        ax2.set_title("threshold grid", fontsize=8.5)
    _save(fig, "fig_verifier_calibration.png",
          "Verifier calibration: hand-checked dev items against the fitted thresholds, and the "
          "top of the threshold grid.",
          "results/verifier_calibration.json")


def fig_claim_verdicts():
    files = sorted(P.results.glob("stage3_verify_*.json"))
    data = [(f.stem.replace("stage3_verify_", ""), json.loads(f.read_text())) for f in files]
    data = [(n, d) for n, d in data if d.get("verdict_rates")]
    if not data:
        return _skip("fig_claim_verdicts.png", "no results/stage3_verify_*.json")
    fig, ax = plt.subplots(figsize=(5.0, 0.7 + 0.7 * len(data)))
    kinds = ["supported", "partially", "unsupported"]
    colours = [PALETTE[2], PALETTE[1], PALETTE[4]]
    left = np.zeros(len(data))
    for k, c in zip(kinds, colours):
        vals = np.array([d["verdict_rates"].get(k, 0) for _, d in data])
        ax.barh(range(len(data)), vals, left=left, color=c, label=k, height=0.5)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0.06:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        color="white")
        left += vals
    ax.set_yticks(range(len(data)), [n for n, _ in data], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of claims")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.02))
    _save(fig, "fig_claim_verdicts.png",
          "Verifier verdicts over extracted claims, by extraction run.",
          "results/stage3_verify_*.json")


# ------------------------------------------------------------------ Stage 4

def fig_gap_heatmap():
    runs = sorted((P.results / "flow_runs").glob("*/gaps.json")) if (P.results / "flow_runs").exists() else []
    if not runs:
        return _skip("fig_gap_heatmap.png", "no flow run with gaps.json")
    d = json.loads(runs[0].read_text())
    m = d["matrix"]
    mat = np.array(m["matrix"], dtype=float)
    if mat.size == 0:
        return _skip("fig_gap_heatmap.png", "empty matrix")
    rows, cols = m["rows"], m["columns"]
    mat, rows = mat[:12], rows[:12]
    mat, cols = mat[:, :14], cols[:14]
    fig, ax = plt.subplots(figsize=(max(4.6, 0.46 * len(cols) + 2.4),
                                    max(2.6, 0.38 * len(rows) + 1.4)))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, int(mat[i, j]) if mat[i, j] else "·", ha="center", va="center",
                    fontsize=6.5, color="white" if mat[i, j] > mat.max() * 0.6 else "#333")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="papers")
    ax.set_title("method family x dataset coverage ('·' = empty cell = candidate gap)",
                 fontsize=8.5)
    _save(fig, "fig_gap_heatmap.png",
          "Method-family by dataset coverage matrix for the demonstration topic; empty cells "
          "are candidate research gaps.",
          str(runs[0].relative_to(P.root)))


def fig_relation_counts():
    f = P.results / "flow_summary.json"
    if not f.exists():
        return _skip("fig_relation_counts.png", "results/flow_summary.json missing")
    d = json.loads(f.read_text())
    counts = {k: v for k, v in (d.get("relation_counts") or {}).items() if k}
    if not counts:
        return _skip("fig_relation_counts.png", "no relation edges in the flow run")
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    keys = sorted(counts, key=lambda k: -counts[k])
    ax.bar(keys, [counts[k] for k in keys], color=PALETTE[5])
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("edges")
    _save(fig, "fig_relation_counts.png",
          "Typed relation edges found in the demonstration corpus.",
          "results/flow_summary.json")


# ------------------------------------------------------------------ Stage 7

def _ci_plot(ax, aggs, metric, label):
    systems = list(aggs)
    means = [(aggs[s].get(metric) or {}).get("mean") for s in systems]
    los = [(aggs[s].get(metric) or {}).get("ci_low") for s in systems]
    his = [(aggs[s].get(metric) or {}).get("ci_high") for s in systems]
    xs, ys, err = [], [], [[], []]
    names = []
    for i, s in enumerate(systems):
        if means[i] is None:
            continue
        xs.append(len(xs))
        ys.append(means[i])
        err[0].append(means[i] - (los[i] if los[i] is not None else means[i]))
        err[1].append((his[i] if his[i] is not None else means[i]) - means[i])
        names.append(aggs[s].get("label", s).replace("full crew", "full\ncrew"))
    if not xs:
        ax.set_visible(False)
        return
    ax.errorbar(xs, ys, yerr=err, fmt="o", ms=5, capsize=3, lw=1.1,
                color=PALETTE[0], ecolor="#888")
    ax.set_xticks(xs, [n.split()[0] for n in names], fontsize=8)
    ax.set_title(label, fontsize=8.5)
    ax.set_ylim(bottom=0)


def fig_eval_main():
    for tag in ("arxiv", "multixscience"):
        d = _load(f"eval_{tag}.json")
        if not d:
            _skip(f"fig_eval_{tag}.png", f"results/eval_{tag}.json missing")
            continue
        aggs = d["aggregates"]
        panels = [("rougeL", "ROUGE-L"), ("bertscore", "BERTScore F1"),
                  ("coverage", "citation coverage"), ("faithfulness", "faithfulness (NLI)")]
        fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels) + 0.6, 2.5))
        for ax, (k, lab) in zip(np.atleast_1d(axes), panels):
            _ci_plot(ax, aggs, k, lab)
        n = d["benchmark"]["n_examples"]
        fig.suptitle(f"{tag} benchmark, {n} examples, bootstrap 95% CIs", fontsize=9)
        _save(fig, f"fig_eval_{tag}.png",
              f"System comparison on the {tag} benchmark ({n} examples) with bootstrap 95% "
              f"confidence intervals.",
              f"results/eval_{tag}.json")


def fig_cost_latency():
    d = _load("eval_arxiv.json")
    if not d:
        return _skip("fig_cost_latency.png", "results/eval_arxiv.json missing")
    aggs = d["aggregates"]
    rows = [(a.get("label", s), (a.get("wall_s") or {}).get("mean"),
             (a.get("n_sentences") or {}).get("mean"))
            for s, a in aggs.items()]
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return _skip("fig_cost_latency.png", "no latency recorded")
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    names = [r[0].split()[0] for r in rows]
    ax.bar(names, [r[1] for r in rows], color=PALETTE[3], width=0.55)
    for i, r in enumerate(rows):
        ax.text(i, r[1] * 1.02, f"{r[1]:.0f}s", ha="center", fontsize=7)
    ax.set_ylabel("wall-clock seconds per example")
    _save(fig, "fig_cost_latency.png",
          "Wall-clock cost per benchmark example by system.",
          "results/eval_arxiv.json")


def fig_usage_by_stage():
    d = _load("llm_usage.json")
    if not d:
        return _skip("fig_usage_by_stage.png", "results/llm_usage.json missing")
    by_tag = {k: v for k, v in (d.get("by_tag") or {}).items() if v.get("total_tokens")}
    if not by_tag:
        return _skip("fig_usage_by_stage.png", "no tagged usage recorded")
    keys = sorted(by_tag, key=lambda k: -by_tag[k]["total_tokens"])[:10]
    fig, ax = plt.subplots(figsize=(5.2, 0.5 + 0.34 * len(keys)))
    ax.barh(range(len(keys))[::-1], [by_tag[k]["total_tokens"] for k in keys],
            color=PALETTE[0], height=0.6)
    ax.set_yticks(range(len(keys))[::-1], keys, fontsize=8)
    ax.set_xlabel("tokens")
    for i, k in enumerate(keys):
        v = by_tag[k]
        ax.text(v["total_tokens"] * 1.01, len(keys) - 1 - i,
                f"{v['total_tokens']:,} ({v['calls']} calls)", va="center", fontsize=7)
    ax.set_xlim(0, max(by_tag[k]["total_tokens"] for k in keys) * 1.45)
    _save(fig, "fig_usage_by_stage.png",
          "LLM tokens consumed per pipeline stage, from the central call log.",
          "results/llm_usage.json")


FIGURES = [fig_yield, fig_dataset_stats, fig_label_distribution, fig_confusion,
           fig_relation_models, fig_verifier_calibration, fig_claim_verdicts,
           fig_gap_heatmap, fig_relation_counts, fig_eval_main, fig_cost_latency,
           fig_usage_by_stage]


def main() -> int:
    for fn in FIGURES:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - one bad figure must not block the rest
            _skip(fn.__name__, f"error: {e}")
    (P.figures / "figures_manifest.json").write_text(json.dumps(
        {"made": MADE, "skipped": SKIPPED}, indent=2))
    print(f"\n{len(MADE)} figures written, {len(SKIPPED)} skipped -> "
          f"{P.figures / 'figures_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
