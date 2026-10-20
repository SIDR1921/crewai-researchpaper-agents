"""LLM labelling of citation contexts, with a rationale, cached and resumable.

Contexts are labelled in small batches to stay inside free-tier request limits,
and every labelled row is appended to a JSONL cache keyed by a stable context id.
Re-running skips what is already there, so hitting a daily quota just means
running again later.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

from ..llm import complete_json, tagged
from .taxonomy import LABELS, normalise_label, taxonomy_prompt

SYSTEM = (
    "You label the relation between a citing paper and a paper it cites, using only the "
    "sentence that contains the citation and its immediate context. You are precise and "
    "you never invent information that is not in the text."
)

MARKER = "[[CITED]]"


def context_id(row: dict) -> str:
    """Stable id for a (citing paper, cite key, sentence) triple."""
    raw = f"{row.get('citing_id','')}|{row.get('cite_key','')}|{row.get('cite_sentence','')}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def mark_target(row: dict) -> str:
    """Rewrite the context so the cited paper under question is unambiguous.

    A sentence often cites several papers; the target's marker becomes
    ``[[CITED]]`` and the others collapse to ``[OTHER]``, otherwise the label
    would be undefined for multi-citation sentences.
    """
    ctx = row.get("context") or row.get("cite_sentence") or ""
    key = re.escape(row.get("cite_key", ""))

    def repl(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        return MARKER if any(re.fullmatch(key, k) for k in keys) else "[OTHER]"

    out = re.sub(r"\[CITE:([^\]]+)\]", repl, ctx)
    return out if MARKER in out else out + f" {MARKER}"


def _prompt(batch: Sequence[dict]) -> str:
    items = []
    for i, r in enumerate(batch, 1):
        items.append(
            f"### Item {i}\n"
            f"Cited paper title: {r.get('cited_title') or 'unknown'}\n"
            f"Citing context (the citation in question is marked {MARKER}):\n"
            f"{mark_target(r)[:1400]}\n"
        )
    return (
        taxonomy_prompt() + "\n\n"
        f"Label each of the {len(batch)} items below.\n\n" + "\n".join(items) +
        "\nReturn ONLY a JSON array with one object per item, in the same order:\n"
        '[{"item": 1, "label": "<one label>", "rationale": "<at most 15 words, quoting the '
        'cue words from the context>", "confidence": <0.0-1.0>}]'
    )


def label_batch(batch: Sequence[dict], size_hint: str = "large") -> list[dict]:
    """Label one batch. Rows the model fails to label come back with ``label=None``."""
    with tagged("relation_labelling"):
        data = complete_json(_prompt(batch), system=SYSTEM, size=size_hint,
                             est_tokens=400 * len(batch) + 600)
    if isinstance(data, dict):
        data = data.get("items") or data.get("labels") or [data]
    by_item: dict[int, dict] = {}
    for i, obj in enumerate(data if isinstance(data, list) else [], 1):
        if not isinstance(obj, dict):
            continue
        idx = int(obj.get("item", i) or i)
        by_item[idx] = obj
    out = []
    for i, row in enumerate(batch, 1):
        obj = by_item.get(i, {})
        out.append({
            "context_id": context_id(row),
            "citing_id": row.get("citing_id"),
            "cite_key": row.get("cite_key"),
            "cited_title": row.get("cited_title"),
            "split": row.get("split"),
            "cite_sentence": row.get("cite_sentence"),
            "marked_context": mark_target(row),
            "llm_label": normalise_label(obj.get("label")),
            "llm_rationale": str(obj.get("rationale", ""))[:300],
            "llm_confidence": float(obj.get("confidence", 0.0) or 0.0),
            "raw_label": obj.get("label"),
        })
    return out


def label_contexts(rows: Iterable[dict], out_path: str | Path, batch_size: int = 8,
                   limit: int | None = None, progress=None, size_hint: str = "large") -> dict:
    """Label ``rows``, appending to ``out_path``; already-labelled ids are skipped."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("llm_label"):
                done.add(rec["context_id"])

    todo = [r for r in rows if context_id(r) not in done]
    if limit:
        todo = todo[:limit]
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    it = progress(batches) if progress else batches
    n_ok, n_fail, n_err = 0, 0, 0
    with out_path.open("a") as f:
        for batch in it:
            try:
                recs = label_batch(batch, size_hint=size_hint)
            except Exception as e:  # noqa: BLE001 - one bad batch must not kill the run
                n_err += len(batch)
                f.write(json.dumps({"context_id": context_id(batch[0]),
                                    "llm_label": None, "error": str(e)[:300]}) + "\n")
                f.flush()
                continue
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
                n_ok += rec["llm_label"] is not None
                n_fail += rec["llm_label"] is None
            f.flush()
    return {"already_done": len(done), "attempted": len(todo), "labelled": n_ok,
            "unparsed": n_fail, "errored": n_err, "path": str(out_path)}


def load_labels(path: str | Path) -> list[dict]:
    """Deduplicated, successfully-labelled rows from a label cache."""
    p = Path(path)
    if not p.exists():
        return []
    best: dict[str, dict] = {}
    for line in p.open():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("llm_label") and rec["llm_label"] in LABELS:
            best[rec["context_id"]] = rec
    return list(best.values())


def zero_shot_label(rows: Sequence[dict], batch_size: int = 8, progress=None) -> list[dict]:
    """Zero-shot LLM baseline for Stage 2, evaluated on the gold set."""
    out: list[dict] = []
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    it = progress(batches) if progress else batches
    with tagged("relation_zeroshot_baseline"):
        for b in it:
            try:
                out += label_batch(b)
            except Exception:
                out += [{"context_id": context_id(r), "llm_label": None} for r in b]
    return out
