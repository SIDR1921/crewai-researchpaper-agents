"""The Extractor: paper text -> :class:`PaperRecord` with quoted evidence.

One prompt, three interchangeable backends, so Stage 6 can swap the API model for
the distilled one without touching the Flow:

* :class:`ApiExtractor`    -- the configured hosted/Ollama model via CrewAI's ``LLM``
* :class:`CrewExtractor`   -- the same model driven as a CrewAI agent with ``output_pydantic``
* :class:`DistilledExtractor` -- the QLoRA-tuned small model from Stage 6

Every claim must quote its evidence verbatim from the text we passed in; the
Verifier (Stage 3) checks that, and the check is only meaningful because the
input text is recorded alongside the record.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol, Sequence

from pydantic import ValidationError

from .llm import complete_json, tagged
from .schemas import PaperRecord

#: Sections given to the Extractor, in priority order, with a per-section budget.
SECTION_BUDGET: list[tuple[str, int]] = [
    ("abstract", 2500),
    ("introduction", 3000),
    ("results", 4000),
    ("conclusion", 2000),
    ("limitations", 1500),
]
MAX_INPUT_CHARS = 12_000

SYSTEM = (
    "You extract structured records from scientific papers. You only state what the given "
    "text says. Every claim you emit must be backed by a passage copied verbatim from that "
    "text -- never paraphrased, never invented. If the text does not support a field, leave "
    "it empty."
)

PROMPT = """Extract a structured record from the paper text below.

Return ONLY JSON of this exact shape:
{{
  "paper_id": "{paper_id}",
  "title": "{title}",
  "year": {year},
  "task": "<the problem the paper addresses, one short phrase>",
  "methods": ["<named model/algorithm/technique the paper proposes or uses>"],
  "datasets": ["<named dataset/benchmark/corpus used>"],
  "metrics": [{{"name": "<metric>", "value": <number or null>,
               "setting": "<dataset/split this value is for, or null>",
               "higher_is_better": true}}],
  "claims": [{{"claim_id": "c1", "text": "<one sentence stating the claim>",
              "type": "finding|method|limitation|comparison",
              "evidence": {{"section": "<which section below it came from>",
                           "passage": "<VERBATIM span copied from the text, 10-60 words>"}}}}],
  "limitations": ["<a limitation the paper itself states>"]
}}

Rules:
- Emit between 3 and 5 claims. Prefer the paper's main results and comparisons.
- `passage` must be a character-for-character copy of a span of the text below.
  Do not fix typos, do not shorten with ellipses, do not join two distant spans.
- Only list datasets and metrics that are actually named in the text.
- `value` is a plain number (e.g. 44.2), not a string, and null when no number is given.

--- PAPER TEXT ---
{text}
--- END PAPER TEXT ---
Reply with JSON only."""


def build_input_text(paper: dict) -> str:
    """Assemble the Extractor's input from the abstract plus available LaTeX sections.

    Also returned verbatim into the record cache, because the Verifier's quote
    check is defined against exactly this string.
    """
    secs: dict[str, str] = dict(paper.get("sections") or {})
    if paper.get("abstract") and not secs.get("abstract"):
        secs["abstract"] = paper["abstract"]
    parts, used = [], 0
    for name, budget in SECTION_BUDGET:
        body = (secs.get(name) or "").strip()
        if not body:
            continue
        chunk = body[:budget]
        if used + len(chunk) > MAX_INPUT_CHARS:
            chunk = chunk[: max(0, MAX_INPUT_CHARS - used)]
        if len(chunk) < 80:
            continue
        parts.append(f"[{name}]\n{chunk}")
        used += len(chunk)
        if used >= MAX_INPUT_CHARS:
            break
    if not parts and paper.get("abstract"):
        parts = [f"[abstract]\n{paper['abstract'][:MAX_INPUT_CHARS]}"]
    return "\n\n".join(parts)


def build_prompt(paper: dict, text: str | None = None) -> str:
    text = text if text is not None else build_input_text(paper)
    return PROMPT.format(
        paper_id=paper.get("paper_id") or paper.get("arxiv_id") or "unknown",
        title=(paper.get("title") or "").replace('"', "'")[:300],
        year=paper.get("year") or (paper.get("published") or "")[:4] or "null",
        text=text,
    )


def coerce_record(data: dict, paper: dict) -> PaperRecord:
    """Validate, repairing the mistakes small models reliably make."""
    data = dict(data or {})
    data.setdefault("paper_id", paper.get("paper_id") or paper.get("arxiv_id") or "unknown")
    data.setdefault("title", paper.get("title") or "")
    if not data.get("year"):
        data["year"] = paper.get("year") or (paper.get("published") or "")[:4] or None
    claims = []
    for i, c in enumerate(data.get("claims") or [], 1):
        if not isinstance(c, dict):
            continue
        ev = c.get("evidence") or {}
        if isinstance(ev, str):
            ev = {"section": "unknown", "passage": ev}
        c = dict(c)
        c["claim_id"] = str(c.get("claim_id") or f"c{i}")
        c["evidence"] = {"section": str(ev.get("section") or "unknown"),
                         "passage": str(ev.get("passage") or "")}
        t = str(c.get("type") or "finding").lower().strip()
        c["type"] = t if t in ("finding", "method", "limitation", "comparison") else "finding"
        if c.get("text") and c["evidence"]["passage"]:
            claims.append(c)
    data["claims"] = claims
    mets = []
    for m in data.get("metrics") or []:
        if not isinstance(m, dict) or not m.get("name"):
            continue
        v = m.get("value")
        if isinstance(v, str):
            try:
                v = float(v.strip().rstrip("%"))
            except ValueError:
                v = None
        mets.append({"name": str(m["name"])[:80], "value": v,
                     "setting": (str(m["setting"])[:160] if m.get("setting") else None),
                     "higher_is_better": bool(m.get("higher_is_better", True))})
    data["metrics"] = mets
    return PaperRecord.model_validate(data)


# ---------------------------------------------------------------- backends

class Extractor(Protocol):
    name: str

    def extract(self, paper: dict) -> tuple[PaperRecord | None, dict]:
        ...


class ApiExtractor:
    """The configured LLM, called directly. The default backend."""

    name = "api"

    def __init__(self, size: str = "large", tag: str = "extract") -> None:
        self.size, self.tag = size, tag

    def extract(self, paper: dict) -> tuple[PaperRecord | None, dict]:
        text = build_input_text(paper)
        meta = {"backend": self.name, "input_chars": len(text), "input_text": text}
        if len(text) < 200:
            return None, meta | {"error": "not enough source text"}
        t0 = time.time()
        try:
            with tagged(self.tag):
                data = complete_json(build_prompt(paper, text), system=SYSTEM, size=self.size,
                                     est_tokens=len(text) // 3 + 900, est_output=850)
            rec = coerce_record(data if isinstance(data, dict) else {}, paper)
            return rec, meta | {"latency_s": round(time.time() - t0, 2), "json_valid": True}
        except (ValueError, ValidationError) as e:
            return None, meta | {"latency_s": round(time.time() - t0, 2), "json_valid": False,
                                 "error": str(e)[:300]}


class CrewExtractor:
    """The same prompt run as a CrewAI agent with ``output_pydantic=PaperRecord``."""

    name = "crew"

    def __init__(self, tag: str = "extract_crew") -> None:
        self.tag = tag

    def extract(self, paper: dict) -> tuple[PaperRecord | None, dict]:
        from .crew import extractor_task
        text = build_input_text(paper)
        meta = {"backend": self.name, "input_chars": len(text), "input_text": text}
        if len(text) < 200:
            return None, meta | {"error": "not enough source text"}
        t0 = time.time()
        try:
            with tagged(self.tag):
                out = extractor_task(paper, text)
            rec = out if isinstance(out, PaperRecord) else coerce_record(out, paper)
            return rec, meta | {"latency_s": round(time.time() - t0, 2), "json_valid": True}
        except Exception as e:  # noqa: BLE001
            return None, meta | {"latency_s": round(time.time() - t0, 2), "json_valid": False,
                                 "error": str(e)[:300]}


class DistilledExtractor:
    """The Stage 6 QLoRA model, loaded locally through transformers."""

    name = "distilled"

    def __init__(self, model_dir: str | Path, max_new_tokens: int = 900,
                 device: str | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .relations.scibert import best_device
        self.device = device or best_device()
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.model_dir = str(model_dir)

    def extract(self, paper: dict) -> tuple[PaperRecord | None, dict]:
        import torch
        from .llm import parse_json
        text = build_input_text(paper)
        meta = {"backend": self.name, "input_chars": len(text), "input_text": text,
                "model_dir": self.model_dir}
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(paper, text)}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        reply = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        meta |= {"latency_s": round(time.time() - t0, 2), "raw_chars": len(reply)}
        try:
            rec = coerce_record(parse_json(reply), paper)
            return rec, meta | {"json_valid": True}
        except (ValueError, ValidationError) as e:
            return None, meta | {"json_valid": False, "error": str(e)[:300]}


# ---------------------------------------------------------------- cached runner

def extract_corpus(papers: Sequence[dict], cache_path: str | Path,
                   extractor: Extractor | None = None, progress=None,
                   limit: int | None = None) -> dict:
    """Extract every paper, caching one JSON line each. Resumable."""
    extractor = extractor or ApiExtractor()
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if cache_path.exists():
        for line in cache_path.open():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record"):
                done.add(rec["paper_id"])

    todo = [p for p in papers
            if (p.get("paper_id") or p.get("arxiv_id")) not in done][: limit or None]
    it = progress(todo) if progress else todo
    stats = {"already_done": len(done), "attempted": 0, "ok": 0, "failed": 0,
             "latency_s": 0.0, "claims": 0}
    with cache_path.open("a") as f:
        for p in it:
            pid = p.get("paper_id") or p.get("arxiv_id")
            rec, meta = extractor.extract(p)
            stats["attempted"] += 1
            stats["latency_s"] += meta.get("latency_s") or 0.0
            if rec is None:
                stats["failed"] += 1
            else:
                stats["ok"] += 1
                stats["claims"] += len(rec.claims)
            f.write(json.dumps({"paper_id": pid, "backend": extractor.name,
                                "record": rec.model_dump(mode="json") if rec else None,
                                "meta": meta}) + "\n")
            f.flush()
    stats["json_validity_rate"] = round(stats["ok"] / max(1, stats["attempted"]), 4)
    stats["latency_s_per_paper"] = round(stats["latency_s"] / max(1, stats["attempted"]), 2)
    stats["claims_per_paper"] = round(stats["claims"] / max(1, stats["ok"]), 2)
    return stats


def load_records(cache_path: str | Path) -> dict[str, dict]:
    """``paper_id -> {record, meta}`` for the successful extractions in a cache."""
    p = Path(cache_path)
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    for line in p.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("record"):
            out[row["paper_id"]] = row
    return out
