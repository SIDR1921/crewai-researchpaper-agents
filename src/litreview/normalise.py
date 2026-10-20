"""Name normalisation for datasets, metrics, methods and tasks.

Three stages, cheapest first:

1. the manual alias table in ``resources/aliases.json``, which is checked into the
   repo and is the only place a human intervenes;
2. fuzzy matching against known canonical names (``token_set_ratio``);
3. sentence embeddings, for paraphrases fuzzy matching misses
   ("extreme summarisation" vs "XSum") -- optional, since it pulls in a model.

Anything still unmatched keeps a tidied version of its original surface form, so
nothing is silently dropped.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path

from rapidfuzz import fuzz, process

KINDS = ("datasets", "metrics", "methods", "tasks")
FUZZ_FLOOR = 88
EMBED_FLOOR = 0.72


@lru_cache(maxsize=1)
def alias_table(path: str | None = None) -> dict:
    if path:
        return json.loads(Path(path).read_text())
    with resources.files("litreview.resources").joinpath("aliases.json").open() as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _lookup() -> dict[str, dict[str, str]]:
    """``kind -> {lowercased surface form: canonical}``."""
    tbl = alias_table()
    out: dict[str, dict[str, str]] = {}
    for kind in KINDS:
        m: dict[str, str] = {}
        for canon, forms in (tbl.get(kind) or {}).items():
            m[canon.lower()] = canon
            for f in forms:
                m[f.lower()] = canon
        out[kind] = m
    return out


def clean_surface(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    # Strip trailing punctuation first, so a trailing "(2018)." is still seen as a year.
    s = s.strip(" .,:;–-\"'")
    s = re.sub(r"\s*\((?:19|20)\d\d\)\s*$", "", s)
    return s.strip(" .,:;–-()[]\"'")


class Normaliser:
    """Reusable normaliser; embeddings are loaded lazily and only if enabled."""

    def __init__(self, use_embeddings: bool = False, model_name: str | None = None) -> None:
        self.use_embeddings = use_embeddings
        self._model = None
        self._model_name = model_name
        self._canon_emb: dict[str, tuple] = {}
        self.unmatched: dict[str, set[str]] = {k: set() for k in KINDS}

    def _embedder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from .config import HF_MODELS
            self._model = SentenceTransformer(self._model_name or HF_MODELS["embed"])
        return self._model

    def canonical_names(self, kind: str) -> list[str]:
        return sorted((alias_table().get(kind) or {}).keys())

    def normalise(self, name: str, kind: str) -> tuple[str, str]:
        """Return ``(canonical_or_tidied_name, how)`` where ``how`` records the stage."""
        s = clean_surface(name)
        if not s:
            return "", "empty"
        low = s.lower()
        table = _lookup().get(kind, {})
        if low in table:
            return table[low], "alias"
        canon = self.canonical_names(kind)
        if canon:
            hit = process.extractOne(s, canon, scorer=fuzz.token_set_ratio)
            if hit and hit[1] >= FUZZ_FLOOR:
                return hit[0], f"fuzzy:{int(hit[1])}"
            hit2 = process.extractOne(low, list(table.keys()), scorer=fuzz.token_set_ratio)
            if hit2 and hit2[1] >= FUZZ_FLOOR:
                return table[hit2[0]], f"fuzzy-alias:{int(hit2[1])}"
        if self.use_embeddings and canon:
            import numpy as np
            m = self._embedder()
            if kind not in self._canon_emb:
                self._canon_emb[kind] = (canon, m.encode(canon, normalize_embeddings=True))
            names, mat = self._canon_emb[kind]
            v = m.encode([s], normalize_embeddings=True)[0]
            sims = mat @ v
            j = int(np.argmax(sims))
            if float(sims[j]) >= EMBED_FLOOR:
                return names[j], f"embed:{float(sims[j]):.2f}"
        self.unmatched[kind].add(s)
        return s, "unmatched"

    def normalise_list(self, names, kind: str) -> list[str]:
        out, seen = [], set()
        for n in names or []:
            c, _ = self.normalise(n, kind)
            if c and c.lower() not in seen:
                seen.add(c.lower())
                out.append(c)
        return out

    def method_family(self, method: str) -> str:
        """Group a canonical method into its family, for the gap matrix."""
        fams = alias_table().get("method_families") or {}
        canon, _ = self.normalise(method, "methods")
        for fam, members in fams.items():
            if canon in members:
                return fam
        hit = process.extractOne(canon, list(fams.keys()), scorer=fuzz.token_set_ratio)
        if hit and hit[1] >= 90:
            return hit[0]
        return "Other"

    def normalise_record(self, rec: dict) -> dict:
        """Add ``*_norm`` fields to a :class:`PaperRecord` dict, leaving originals intact."""
        out = dict(rec)
        out["methods_norm"] = self.normalise_list(rec.get("methods"), "methods")
        out["datasets_norm"] = self.normalise_list(rec.get("datasets"), "datasets")
        out["task_norm"] = self.normalise(rec.get("task") or "", "tasks")[0] or None
        out["method_families"] = sorted({self.method_family(m) for m in out["methods_norm"]})
        mets = []
        for m in rec.get("metrics") or []:
            mm = dict(m)
            mm["name_norm"] = self.normalise(m.get("name") or "", "metrics")[0]
            mm["setting_dataset"] = (
                self.normalise(m.get("setting") or "", "datasets")[0]
                if m.get("setting") else None)
            mets.append(mm)
        out["metrics_norm"] = mets
        return out

    def report(self) -> dict:
        return {"unmatched_counts": {k: len(v) for k, v in self.unmatched.items()},
                "unmatched_examples": {k: sorted(v)[:25] for k, v in self.unmatched.items()},
                "canonical_counts": {k: len(self.canonical_names(k)) for k in KINDS},
                "fuzz_floor": FUZZ_FLOOR, "embed_floor": EMBED_FLOOR,
                "embeddings_used": self.use_embeddings}
