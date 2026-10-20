"""Gap detection (Stage 4b).

Three mechanical signals, then an LLM Gap Analyst that must argue both sides:

* an **empty cell** in the method-family x dataset matrix -- a family nobody has
  tried on a dataset that is otherwise well used;
* a **sparse cell** -- one paper where the row and column averages suggest more;
* an **untested comparison** -- two method families that appear on the same
  dataset but never have a ``compares-against`` edge between their papers.

Counts alone would be misleading (an empty cell is often empty for a good
reason), so the Analyst's job is to argue *against* each candidate too, and its
reasoning is kept in the output rather than collapsed to a score.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Sequence

from .llm import complete_json, tagged
from .schemas import Gap

SPARSE_MAX = 1


def build_matrix(records: Sequence[dict], axis: str = "datasets_norm") -> dict:
    """Method-family x dataset (or task) counts, plus the per-year trend."""
    cells: dict[tuple[str, str], set[str]] = defaultdict(set)
    years: dict[str, Counter] = defaultdict(Counter)
    fams: Counter = Counter()
    cols: Counter = Counter()
    for r in records:
        pid = r.get("paper_id")
        families = r.get("method_families") or ["Other"]
        if axis == "task_norm":
            values = [r.get("task_norm") or r.get("task")]
        else:
            values = r.get(axis) or r.get("datasets") or []
        values = [v for v in values if v]
        for f in families:
            fams[f] += 1
            for v in values:
                cells[(f, v)].add(pid)
                cols[v] += 1
            if r.get("year"):
                years[f][int(r["year"])] += 1
    rows = sorted(fams, key=lambda f: -fams[f])
    columns = sorted(cols, key=lambda c: -cols[c])
    matrix = [[len(cells.get((f, c), ())) for c in columns] for f in rows]
    return {"axis": axis, "rows": rows, "columns": columns, "matrix": matrix,
            "row_totals": {f: fams[f] for f in rows},
            "column_totals": {c: cols[c] for c in columns},
            "cell_papers": {f"{f}||{c}": sorted(v) for (f, c), v in cells.items()},
            "trend_by_year": {f: dict(sorted(years[f].items())) for f in rows if years[f]},
            "n_records": len(records)}


def candidate_gaps(matrix: dict, edges: Sequence[dict] | None = None,
                   records: Sequence[dict] | None = None,
                   min_row_total: int = 2, min_col_total: int = 2,
                   max_candidates: int = 40) -> list[Gap]:
    """Mechanical candidates, scored so the Analyst sees the most plausible first."""
    rows, cols, m = matrix["rows"], matrix["columns"], matrix["matrix"]
    out: list[Gap] = []
    for i, fam in enumerate(rows):
        if matrix["row_totals"][fam] < min_row_total:
            continue
        for j, col in enumerate(cols):
            if matrix["column_totals"][col] < min_col_total:
                continue
            n = m[i][j]
            if n > SPARSE_MAX:
                continue
            # Expected count under independence of row and column activity.
            total = sum(sum(r) for r in m) or 1
            exp = (sum(m[i]) * sum(r[j] for r in m)) / total
            score = round(max(0.0, exp - n), 3)
            out.append(Gap(
                kind="empty-cell" if n == 0 else "sparse-cell",
                method_family=fam, dataset=col, n_papers=n,
                candidate=(f"No paper in this corpus applies {fam} methods to {col}"
                           if n == 0 else
                           f"Only {n} paper in this corpus applies {fam} methods to {col}"),
                score=score))

    # untested comparisons between families that co-occur on a dataset
    if records is not None:
        fam_of: dict[str, set[str]] = {r.get("paper_id"): set(r.get("method_families") or [])
                                       for r in records}
        ds_of: dict[str, set[str]] = {r.get("paper_id"): set(r.get("datasets_norm") or [])
                                      for r in records}
        compared: set[frozenset] = set()
        for e in edges or []:
            if e.get("relation") == "compares-against":
                for a in fam_of.get(e.get("citing_id"), ()):
                    for b in fam_of.get(e.get("cited_id"), ()):
                        compared.add(frozenset((a, b)))
        by_ds: dict[str, set[str]] = defaultdict(set)
        for pid, dss in ds_of.items():
            for d in dss:
                by_ds[d] |= fam_of.get(pid, set())
        for d, families in by_ds.items():
            for a, b in combinations(sorted(families), 2):
                if frozenset((a, b)) in compared:
                    continue
                out.append(Gap(
                    kind="untested-comparison", method_family=f"{a} vs {b}", dataset=d,
                    n_papers=0,
                    candidate=(f"{a} and {b} methods are both evaluated on {d} but no paper in "
                               f"this corpus compares them directly"),
                    score=1.0))
    out.sort(key=lambda g: -g.score)
    return out[:max_candidates]


ANALYST_SYSTEM = (
    "You assess whether a candidate research gap is real. A cell can be empty because nobody "
    "has tried it, or because the combination makes no sense, or because the corpus is small. "
    "You must argue both sides from the evidence you are given and then decide. You never "
    "claim knowledge of papers that are not in the evidence."
)


def rank_gaps(candidates: Sequence[Gap], matrix: dict, records: Sequence[dict],
              limit: int = 12, progress=None) -> list[Gap]:
    """Have the Gap Analyst argue for and against each candidate, keeping the reasoning."""
    recs_by_id = {r.get("paper_id"): r for r in records}
    items = list(candidates)[:limit]
    it = progress(items) if progress else items
    for g in items if progress is None else it:
        key = f"{g.method_family}||{g.dataset}"
        near = []
        for k, pids in (matrix.get("cell_papers") or {}).items():
            fam, col = k.split("||", 1)
            if fam == g.method_family or col == g.dataset:
                for pid in pids[:2]:
                    r = recs_by_id.get(pid) or {}
                    near.append(f"[{pid}] {fam} on {col}: {(r.get('title') or '')[:90]}")
        prompt = (
            f"Candidate gap ({g.kind}): {g.candidate}\n"
            f"Corpus size: {matrix['n_records']} papers.\n"
            f"Method family '{g.method_family}' appears in "
            f"{matrix['row_totals'].get(g.method_family, 0)} papers; "
            f"'{g.dataset}' appears in {matrix['column_totals'].get(g.dataset, 0)} papers.\n"
            f"Papers in neighbouring cells:\n" + ("\n".join(near[:12]) or "(none)") + "\n\n"
            'Return JSON: {"argument_for": "<=40 words why this is a real, worthwhile gap>",'
            ' "argument_against": "<=40 words why it may not be: infeasible, already covered '
            'elsewhere, or an artefact of a small corpus>",'
            ' "verdict": "real-gap|not-a-gap|uncertain", "score": <0.0-1.0 priority>}'
        )
        try:
            with tagged("gap_analysis"):
                data = complete_json(prompt, system=ANALYST_SYSTEM, est_tokens=1200)
            g.argument_for = str(data.get("argument_for", ""))[:400]
            g.argument_against = str(data.get("argument_against", ""))[:400]
            v = str(data.get("verdict", "uncertain")).strip().lower()
            g.verdict = v if v in ("real-gap", "not-a-gap", "uncertain") else "uncertain"
            try:
                g.score = round(float(data.get("score", g.score)), 3)
            except (TypeError, ValueError):
                pass
        except Exception as e:  # noqa: BLE001
            g.argument_for = g.argument_against = None
            g.verdict = "uncertain"
            g.candidate += f" [analyst error: {str(e)[:100]}]"
    items.sort(key=lambda x: (x.verdict != "real-gap", -x.score))
    return items


def save(matrix: dict, gaps: Sequence[Gap], out_path: str | Path) -> dict:
    report = {
        "matrix": {k: v for k, v in matrix.items() if k != "cell_papers"},
        "n_candidates": len(gaps),
        "verdicts": dict(Counter(g.verdict or "unscored" for g in gaps)),
        "gaps": [g.model_dump() for g in gaps],
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    return report
