"""Consensus / contradiction map over the extracted records (Stage 4).

Two independent sources of disagreement are combined:

* **Reported numbers.** Records are grouped by ``(task, dataset, metric)`` after
  normalisation. Within a group, a spread wider than ``rel_tol`` of the group
  median is a *value disagreement*; two papers that order the same pair of
  methods oppositely are a *ranking disagreement*.
* **The relation classifier.** A ``contrasts`` edge between two papers is a
  disagreement the authors themselves asserted, which numbers alone cannot show.

Flagged contradictions are then handed to the Adjudicator agent, which must
explain each one using only the evidence passages -- different setting, different
split, or a genuine disagreement.
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Sequence

from .llm import complete_json, tagged
from .schemas import Contradiction

REL_TOL = 0.15       # a >15% spread of the group median counts as a disagreement
MIN_GROUP = 2


def _metric_rows(records: Sequence[dict]) -> list[dict]:
    """Flatten every (paper, metric) pair that has a number and a normalised name."""
    rows = []
    for r in records:
        for m in r.get("metrics_norm") or r.get("metrics") or []:
            if m.get("value") is None:
                continue
            rows.append({
                "paper_id": r.get("paper_id"), "title": r.get("title"), "year": r.get("year"),
                "task": r.get("task_norm") or r.get("task"),
                "dataset": m.get("setting_dataset") or (r.get("datasets_norm") or [None])[0],
                "metric": m.get("name_norm") or m.get("name"),
                "value": float(m["value"]),
                "setting": m.get("setting"),
                "higher_is_better": bool(m.get("higher_is_better", True)),
                "methods": r.get("methods_norm") or r.get("methods") or [],
            })
    return rows


def consensus_map(records: Sequence[dict]) -> dict:
    """Group reported numbers by ``(task, dataset, metric)``; summarise each group."""
    rows = _metric_rows(records)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["dataset"] and r["metric"]:
            groups[(r["task"], r["dataset"], r["metric"])].append(r)
    out = []
    for (task, ds, met), rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        vals = sorted(r["value"] for r in rs)
        med = vals[len(vals) // 2]
        spread = (max(vals) - min(vals)) / abs(med) if med else 0.0
        out.append({
            "task": task, "dataset": ds, "metric": met,
            "n_papers": len({r["paper_id"] for r in rs}), "n_values": len(vals),
            "min": min(vals), "median": med, "max": max(vals),
            "relative_spread": round(spread, 4),
            "agrees": spread <= REL_TOL,
            "papers": sorted({r["paper_id"] for r in rs}),
            "values": [{"paper_id": r["paper_id"], "value": r["value"],
                        "setting": r["setting"], "methods": r["methods"][:3]} for r in rs],
        })
    return {"groups": out, "rel_tol": REL_TOL,
            "n_groups": len(out),
            "n_agreeing": sum(1 for g in out if g["agrees"]),
            "n_comparable_groups": sum(1 for g in out if g["n_papers"] >= MIN_GROUP)}


def find_contradictions(records: Sequence[dict], edges: Sequence[dict] | None = None,
                        rel_tol: float = REL_TOL) -> list[Contradiction]:
    """Value disagreements, ranking disagreements, and classifier ``contrasts`` edges."""
    out: list[Contradiction] = []
    cmap = consensus_map(records)
    for g in cmap["groups"]:
        if g["n_papers"] < MIN_GROUP or g["agrees"]:
            continue
        out.append(Contradiction(
            task=g["task"], dataset=g["dataset"], metric=g["metric"],
            papers=g["papers"], values=g["values"],
            kind="value-disagreement", spread=g["relative_spread"]))

    # ranking disagreements: two papers that order the same method pair oppositely
    best: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in _metric_rows(records):
        if not (r["dataset"] and r["metric"] and r["methods"]):
            continue
        key = (r["task"], r["dataset"], r["metric"], r["paper_id"])
        for m in r["methods"]:
            prev = best[key].get(m)
            if prev is None or (r["value"] > prev) == r["higher_is_better"]:
                best[key][m] = r["value"]
    orders: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
    for (task, ds, met, pid), by_m in best.items():
        for a, b in combinations(sorted(by_m), 2):
            if by_m[a] == by_m[b]:
                continue
            winner = a if by_m[a] > by_m[b] else b
            orders[(task, ds, met, a, b)].append((pid, winner, f"{by_m[a]} vs {by_m[b]}"))
    for (task, ds, met, a, b), obs in orders.items():
        winners = {w for _, w, _ in obs}
        if len(winners) > 1:
            out.append(Contradiction(
                task=task, dataset=ds, metric=met,
                papers=sorted({p for p, _, _ in obs}),
                values=[{"paper_id": p, "winner": w, "detail": d} for p, w, d in obs],
                kind="ranking-disagreement", spread=None))

    for e in edges or []:
        if e.get("relation") == "contrasts":
            out.append(Contradiction(
                papers=[e.get("citing_id", ""), e.get("cited_id", "")],
                values=[{"relation": "contrasts", "prob": e.get("prob"),
                         "sentence": (e.get("sentence") or "")[:400]}],
                kind="classifier-contrasts", spread=None))
    return out


ADJUDICATOR_SYSTEM = (
    "You adjudicate apparent disagreements between scientific papers. You may use ONLY the "
    "evidence given to you. Decide whether the disagreement is explained by a different "
    "experimental setting, a different data split or metric definition, or whether it is a "
    "genuine disagreement. If the evidence does not let you tell, say so."
)


def adjudicate(contradictions: Sequence[Contradiction], records: dict[str, dict],
               limit: int | None = None, progress=None) -> list[Contradiction]:
    """Ask the Adjudicator agent to explain each flagged contradiction."""
    items = list(contradictions)[: limit or None]
    it = progress(items) if progress else items
    for c in it:
        ev = []
        for pid in c.papers[:4]:
            rec = (records.get(pid) or {}).get("record") or records.get(pid) or {}
            for cl in (rec.get("claims") or [])[:3]:
                ev.append(f"[{pid}] {cl['text']} (quoted: \"{cl['evidence']['passage'][:220]}\")")
        prompt = (
            f"Apparent disagreement ({c.kind}).\n"
            f"Task: {c.task}; dataset: {c.dataset}; metric: {c.metric}\n"
            f"Reported: {json.dumps(c.values)[:1200]}\n\n"
            "Evidence passages from the papers involved:\n" + ("\n".join(ev) or "(none)") +
            '\n\nReturn JSON: {"explanation": "<=60 words, citing the evidence",'
            ' "category": "different-setting|different-split-or-metric|genuine-disagreement|'
            'insufficient-evidence"}'
        )
        try:
            with tagged("adjudicate"):
                data = complete_json(prompt, system=ADJUDICATOR_SYSTEM, est_tokens=1200)
            c.adjudication = f"[{data.get('category', '?')}] {str(data.get('explanation', ''))[:400]}"
        except Exception as e:  # noqa: BLE001
            c.adjudication = f"[error] {str(e)[:160]}"
    return items


# ------------------------------------------------------------------- the graph

def build_graph(records: Sequence[dict], edges: Sequence[dict],
                contradictions: Sequence[Contradiction] | None = None):
    """A typed citation graph: nodes are papers, edges carry the relation label."""
    import networkx as nx
    g = nx.MultiDiGraph()
    by_id = {r.get("paper_id"): r for r in records}
    for pid, r in by_id.items():
        g.add_node(pid, label=(r.get("title") or pid)[:70], year=r.get("year"),
                   task=r.get("task_norm") or r.get("task"),
                   families=", ".join(r.get("method_families") or []),
                   n_claims=len(r.get("claims") or []), kind="paper")
    for e in edges:
        a, b = e.get("citing_id"), e.get("cited_id")
        if not a or not b:
            continue
        for n in (a, b):
            if n not in g:
                g.add_node(n, label=n[:70], kind="cited-only")
        g.add_edge(a, b, key=e.get("relation"), relation=e.get("relation"),
                   prob=round(float(e.get("prob") or 0.0), 3),
                   title=(e.get("sentence") or "")[:200])
    for c in (contradictions or []):
        if c.kind == "classifier-contrasts" or len(c.papers) < 2:
            continue
        for a, b in combinations(c.papers[:4], 2):
            if a in g and b in g:
                g.add_edge(a, b, key="contradiction",
                           relation="contradiction", prob=1.0,
                           title=f"{c.kind}: {c.metric} on {c.dataset}")
    return g


RELATION_COLOURS = {
    "background": "#9aa5b1", "uses-method": "#2f80ed", "uses-data": "#27ae60",
    "extends": "#8e44ad", "compares-against": "#f2994a", "supports": "#16a085",
    "contrasts": "#eb5757", "contradiction": "#c0392b",
}


def export_graph_html(g, out_path: str | Path, height: str = "800px") -> Path:
    """Interactive pyvis rendering of the typed citation graph."""
    from pyvis.network import Network
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    net = Network(height=height, width="100%", directed=True, bgcolor="#ffffff",
                  font_color="#222222", notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-12000, spring_length=140)
    for n, d in g.nodes(data=True):
        deg = g.degree(n)
        net.add_node(n, label=(d.get("label") or n)[:40],
                     title=f"{d.get('label', n)}\nyear: {d.get('year')}\n"
                           f"task: {d.get('task')}\nfamilies: {d.get('families')}",
                     size=10 + min(30, 3 * deg),
                     color="#1f3a93" if d.get("kind") == "paper" else "#cfd8dc")
    for a, b, k, d in g.edges(data=True, keys=True):
        rel = d.get("relation") or k or "background"
        net.add_edge(a, b, title=f"{rel}: {d.get('title', '')}",
                     color=RELATION_COLOURS.get(rel, "#9aa5b1"),
                     width=1 + 3 * float(d.get("prob") or 0.0), label="")
    net.set_options(json.dumps({
        "physics": {"stabilization": {"iterations": 200}},
        "edges": {"smooth": {"type": "continuous"}, "arrows": {"to": {"scaleFactor": 0.5}}},
        "interaction": {"hover": True, "tooltipDelay": 120},
    }))
    net.write_html(str(out_path), open_browser=False, notebook=False)
    # Legend, so the colours mean something without reading the code.
    legend = "".join(
        f'<span style="display:inline-block;margin:0 10px 6px 0;">'
        f'<span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'vertical-align:middle;"></span> {r}</span>'
        for r, c in RELATION_COLOURS.items())
    html = out_path.read_text().replace(
        "<body>", f'<body><div style="font:13px system-ui;padding:8px 12px;">'
                  f'<b>Typed citation graph</b> &mdash; {legend}</div>', 1)
    out_path.write_text(html)
    return out_path


def contradictions_table(contradictions: Sequence[Contradiction]) -> list[dict]:
    """Flat rows for the report table."""
    rows = []
    for i, c in enumerate(contradictions, 1):
        rows.append({
            "id": f"C{i}", "kind": c.kind, "task": c.task, "dataset": c.dataset,
            "metric": c.metric, "n_papers": len(c.papers),
            "papers": ", ".join(c.papers[:4]),
            "spread": None if c.spread is None else round(c.spread, 3),
            "detail": json.dumps(c.values)[:300],
            "adjudication": c.adjudication,
        })
    return rows
