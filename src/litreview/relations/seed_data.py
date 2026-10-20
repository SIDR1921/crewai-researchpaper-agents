"""SciCite and ACL-ARC, mapped onto our taxonomy.

Both corpora are coarser than our seven labels, so mapped rows are used as extra
*training* signal only; the evaluation gold set is always our own hand labels.
The mapping and its limits are recorded in the returned report, and repeated in
``docs/labelling_guide.md``.

If a corpus cannot be downloaded the loader returns an empty list with
``available: False`` rather than silently substituting something else.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import requests

from .taxonomy import ACL_ARC_LIMITS, ACL_ARC_MAP, SCICITE_LIMITS, SCICITE_MAP

SCICITE_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"
ACL_ARC_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/acl-arc.tar.gz"


def _download_tar(url: str, dest: Path) -> Path | None:
    dest.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    tgz = dest / name
    marker = dest / (name.replace(".tar.gz", "") + ".extracted")
    if marker.exists():
        return dest
    try:
        if not tgz.exists() or tgz.stat().st_size < 1000:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            tgz.write_bytes(r.content)
        with tarfile.open(tgz, "r:gz") as tar:
            members = [m for m in tar.getmembers()
                       if m.isfile() and not m.name.startswith(("/", ".."))]
            for m in members:
                tar.extract(m, dest, filter="data")
        marker.write_text("ok")
        return dest
    except Exception:
        return None


def _read_jsonl_tree(root: Path, want: tuple[str, ...] = ("train", "dev", "test")) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in root.rglob("*.jsonl"):
        stem = p.stem.lower()
        for w in want:
            if stem.startswith(w):
                rows = []
                for line in p.open():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                out.setdefault(w, []).extend(rows)
    return out


def load_scicite(cache_dir: Path, splits: tuple[str, ...] = ("train",),
                 limit: int | None = None) -> dict:
    """SciCite (Cohan et al., NAACL 2019) mapped onto our labels."""
    root = _download_tar(SCICITE_URL, cache_dir / "scicite")
    if root is None:
        return {"available": False, "reason": "download failed", "rows": [],
                "mapping": SCICITE_MAP, "limits": SCICITE_LIMITS}
    data = _read_jsonl_tree(root, want=splits)
    rows, dropped = [], 0
    for split, rs in data.items():
        for r in rs:
            mapped = SCICITE_MAP.get((r.get("label") or "").lower())
            if mapped is None:
                dropped += 1
                continue
            text = r.get("string") or r.get("text") or ""
            rows.append({
                "context_id": f"scicite:{r.get('unique_id') or r.get('id') or len(rows)}",
                "citing_id": f"scicite:{r.get('citingPaperId') or 'unk'}",
                "cite_key": r.get("citedPaperId") or "",
                "cited_title": "",
                "marked_context": text,
                "label": mapped,
                "source": "scicite",
                "orig_label": r.get("label"),
                "split": split,
            })
    if limit:
        rows = rows[:limit]
    return {"available": True, "rows": rows, "n": len(rows), "dropped": dropped,
            "mapping": SCICITE_MAP, "limits": SCICITE_LIMITS, "citation":
            "Cohan, Ammar, van Zuylen & Cady (2019), Structural Scaffolds for Citation "
            "Intent Classification in Scientific Publications, NAACL."}


def load_acl_arc(cache_dir: Path, splits: tuple[str, ...] = ("train",),
                 limit: int | None = None) -> dict:
    """ACL-ARC citation-function data (Jurgens et al., TACL 2018) mapped onto our labels."""
    root = _download_tar(ACL_ARC_URL, cache_dir / "acl_arc")
    if root is None:
        return {"available": False, "reason": "download failed", "rows": [],
                "mapping": ACL_ARC_MAP, "limits": ACL_ARC_LIMITS}
    data = _read_jsonl_tree(root, want=splits)
    rows, dropped = [], 0
    for split, rs in data.items():
        for r in rs:
            mapped = ACL_ARC_MAP.get((r.get("intent") or r.get("label") or "").strip())
            if mapped is None:
                dropped += 1
                continue
            rows.append({
                "context_id": f"aclarc:{r.get('unique_id') or r.get('id') or len(rows)}",
                "citing_id": f"aclarc:{r.get('citing_paper_id') or r.get('citingPaperId') or 'unk'}",
                "cite_key": r.get("cited_paper_id") or "",
                "cited_title": "",
                "marked_context": r.get("text") or r.get("string") or "",
                "label": mapped,
                "source": "acl-arc",
                "orig_label": r.get("intent") or r.get("label"),
                "split": split,
            })
    if limit:
        rows = rows[:limit]
    return {"available": True, "rows": rows, "n": len(rows), "dropped": dropped,
            "mapping": ACL_ARC_MAP, "limits": ACL_ARC_LIMITS, "citation":
            "Jurgens, Kumar, Hoover, McFarland & Jurafsky (2018), Measuring the Evolution "
            "of a Scientific Field through Citation Frames, TACL 6:391-406."}
