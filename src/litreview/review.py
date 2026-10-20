"""Writing the review: the grounded Writer, the revise loop, and the S1 baseline.

The Writer never sees raw paper text. It sees *verified claims* -- each with an
id, a paper id and a quoted passage -- and must attach the ids it used to every
sentence it writes. That is what makes sentence-level verification possible at
all: the Verifier knows exactly which passage each sentence is supposed to follow
from.

:func:`single_call_review` is system S1 in the evaluation: one well-prompted call
with the same inputs and no agents, no claims and no verification.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

from .llm import complete, complete_json, tagged
from .schemas import Review, ReviewSentence

WRITER_SYSTEM = (
    "You write the related-work section of a scientific paper. Every sentence you write must "
    "follow from the claims you are given, and you must list the claim ids and paper ids it "
    "rests on. You never state anything the claims do not support, and you never cite a paper "
    "whose claims you did not use."
)

WRITER_PROMPT = """Write the related-work section for a paper on: {topic}

You may use ONLY the verified claims below. Each claim has an id, the paper it came from,
and the passage it was verified against.

{claims_block}
{extra_block}
Requirements:
- {n_sent} to {n_sent_max} sentences, grouped into 2-4 thematic paragraphs.
- Every sentence must be supported by at least one claim id.
- Cite papers inline as [paper_id].
- Group papers by what they have in common; do not write one sentence per paper.
- Where the claims disagree, say so explicitly.
- Neutral academic prose. No bullet points, no headings, no filler.

Return ONLY JSON:
{{"sentences": [{{"text": "<one sentence, with inline [paper_id] citations>",
                 "supporting_claim_ids": ["c1"], "cited_paper_ids": ["<paper_id>"]}}]}}"""


def claims_block(records: Sequence[dict], verdicts: dict[str, list[dict]] | None = None,
                 only_supported: bool = True, max_per_paper: int = 5,
                 max_chars: int = 14_000) -> tuple[str, dict[str, str]]:
    """Render claims for the prompt, and return ``claim_id -> evidence passage``.

    Claim ids are prefixed with the paper id, so they stay unique across papers.
    """
    lines, evidence = [], {}
    used = 0
    for r in records:
        pid = r.get("paper_id")
        ok = None
        if verdicts and only_supported:
            ok = {v["claim_id"] for v in verdicts.get(pid, [])
                  if v["verdict"] in ("supported", "partially")}
        head = f"\n[{pid}] {(r.get('title') or '')[:140]} ({r.get('year') or 'n.d.'})"
        body = []
        for c in (r.get("claims") or [])[:max_per_paper]:
            if ok is not None and c["claim_id"] not in ok:
                continue
            gid = f"{pid}:{c['claim_id']}"
            evidence[gid] = c["evidence"]["passage"]
            body.append(f"  - {gid} ({c['type']}): {c['text']}")
        if not body:
            continue
        chunk = head + "\n" + "\n".join(body)
        if used + len(chunk) > max_chars:
            break
        lines.append(chunk)
        used += len(chunk)
    return "\n".join(lines), evidence


def _extra_block(contradictions: Sequence[dict] | None, gaps: Sequence[dict] | None,
                 relations: Sequence[dict] | None) -> str:
    parts = []
    if relations:
        rel = [f"  - {e['citing_id']} {e['relation']} {e['cited_id']}"
               for e in relations
               if e.get("relation") in ("contrasts", "supports", "extends", "compares-against")][:25]
        if rel:
            parts.append("Typed relations found between these papers (from the citation-relation "
                         "classifier):\n" + "\n".join(rel))
    if contradictions:
        con = [f"  - {c.get('kind')}: {c.get('metric')} on {c.get('dataset')} "
               f"({', '.join((c.get('papers') or [])[:3])}) "
               f"{(c.get('adjudication') or '')[:160]}" for c in contradictions[:8]]
        parts.append("Disagreements the Adjudicator confirmed:\n" + "\n".join(con))
    if gaps:
        gp = [f"  - {g.get('candidate')} [{g.get('verdict')}]" for g in gaps[:6]
              if g.get("verdict") == "real-gap"]
        if gp:
            parts.append("Gaps the Gap Analyst judged real (mention at most two, at the end):\n"
                         + "\n".join(gp))
    return ("\n" + "\n\n".join(parts) + "\n") if parts else "\n"


def write_review(topic: str, records: Sequence[dict],
                 verdicts: dict[str, list[dict]] | None = None,
                 contradictions: Sequence[dict] | None = None,
                 gaps: Sequence[dict] | None = None,
                 relations: Sequence[dict] | None = None,
                 n_sent: int = 10, n_sent_max: int = 18,
                 only_supported: bool = True) -> tuple[Review, dict[str, str]]:
    """One Writer pass. Returns the review and the ``claim_id -> evidence`` map."""
    block, evidence = claims_block(records, verdicts, only_supported=only_supported)
    prompt = WRITER_PROMPT.format(
        topic=topic, claims_block=block or "(no verified claims)",
        extra_block=_extra_block(contradictions, gaps, relations),
        n_sent=n_sent, n_sent_max=n_sent_max)
    with tagged("write_review"):
        data = complete_json(prompt, system=WRITER_SYSTEM, est_tokens=len(prompt) // 3 + 1200)
    sents = []
    for s in (data.get("sentences") if isinstance(data, dict) else data) or []:
        if not isinstance(s, dict) or not (s.get("text") or "").strip():
            continue
        sents.append(ReviewSentence(
            text=str(s["text"]).strip(),
            supporting_claim_ids=[str(x) for x in (s.get("supporting_claim_ids") or [])],
            cited_paper_ids=[str(x) for x in (s.get("cited_paper_ids") or [])]))
    rev = Review(topic=topic, sentences=sents, markdown=to_markdown(topic, sents))
    return rev, evidence


REVISE_PROMPT = """Some sentences of your related-work section are not supported by the
evidence behind the claim ids you attached. Rewrite ONLY those sentences so each one follows
directly from its evidence, or drop a sentence if no claim supports it.

Unsupported sentences, with the closest evidence the verifier found:
{bad}

The claims you may use:
{claims_block}

Return ONLY JSON in the same shape as before, containing the FULL revised section:
{{"sentences": [{{"text": "...", "supporting_claim_ids": ["..."], "cited_paper_ids": ["..."]}}]}}"""


def revise_review(review: Review, bad: Sequence[dict], records: Sequence[dict],
                  verdicts: dict[str, list[dict]] | None = None) -> Review:
    """One revision pass over the sentences the Verifier rejected."""
    block, _ = claims_block(records, verdicts)
    bad_block = "\n".join(
        f"  - \"{b['text'][:220]}\" (entailment {b.get('entail_prob', 0):.2f}; closest evidence: "
        f"\"{(b.get('best_evidence') or 'none')[:200]}\")" for b in bad)
    with tagged("revise_review"):
        data = complete_json(REVISE_PROMPT.format(bad=bad_block, claims_block=block),
                             system=WRITER_SYSTEM, est_tokens=len(block) // 3 + 1500)
    sents = []
    for s in (data.get("sentences") if isinstance(data, dict) else data) or []:
        if isinstance(s, dict) and (s.get("text") or "").strip():
            sents.append(ReviewSentence(
                text=str(s["text"]).strip(),
                supporting_claim_ids=[str(x) for x in (s.get("supporting_claim_ids") or [])],
                cited_paper_ids=[str(x) for x in (s.get("cited_paper_ids") or [])]))
    return Review(topic=review.topic, sentences=sents or review.sentences,
                  markdown=to_markdown(review.topic, sents or review.sentences))


def to_markdown(topic: str, sentences: Sequence[ReviewSentence],
                flags: Sequence[str] | None = None) -> str:
    """Join sentences into paragraphs, flagging any that stayed unsupported."""
    flagged = set(flags or ())
    paras, cur = [], []
    for i, s in enumerate(sentences):
        txt = s.text + (" _[unverified]_" if s.text in flagged else "")
        cur.append(txt)
        if len(cur) >= 4 and i < len(sentences) - 1:
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    return f"## Related Work: {topic}\n\n" + "\n\n".join(paras) + "\n"


# ------------------------------------------------------------ S1: single call

S1_SYSTEM = ("You write the related-work section of a scientific paper, in neutral academic "
             "prose, using only the abstracts you are given.")

S1_PROMPT = """Write the related-work section of a paper on: {topic}

Use only these cited papers.

{papers}

Requirements:
- {n_sent} to {n_sent_max} sentences in 2-4 paragraphs.
- Cite papers inline as [{id_example}].
- Group papers by theme rather than listing them one by one.
- Discuss every paper at least once.
- Neutral academic prose. No bullet points, no headings.

Return the section text only."""


def single_call_review(topic: str, cited: Sequence[dict], n_sent: int = 10,
                       n_sent_max: int = 18, max_chars: int = 14_000) -> Review:
    """System S1: one well-prompted LLM call, same inputs, no agents or verification."""
    lines, used = [], 0
    for c in cited:
        pid = c.get("paper_id") or c.get("cite_key") or c.get("openalex_id") or "?"
        chunk = (f"\n[{pid}] {(c.get('title') or 'untitled')[:160]}"
                 f" ({c.get('year') or 'n.d.'})\n{(c.get('abstract') or '')[:1200]}")
        if used + len(chunk) > max_chars:
            break
        lines.append(chunk)
        used += len(chunk)
    prompt = S1_PROMPT.format(topic=topic, papers="\n".join(lines), n_sent=n_sent,
                              n_sent_max=n_sent_max,
                              id_example=(cited[0].get("cite_key") if cited else "paper_id"))
    with tagged("s1_single_call"):
        text = complete(prompt, system=S1_SYSTEM, est_tokens=used // 3 + 1200)
    return Review(topic=topic, sentences=[ReviewSentence(text=s, cited_paper_ids=cited_ids(s))
                                          for s in split_sentences(text)],
                  markdown=text.strip() + "\n")


CITE_IN_TEXT = re.compile(r"\[([^\]\s][^\]]{0,80})\]")


def cited_ids(text: str) -> list[str]:
    out = []
    for m in CITE_IN_TEXT.finditer(text or ""):
        for part in m.group(1).split(","):
            p = part.strip()
            if p and p.lower() not in ("unverified", "cite"):
                out.append(p)
    return out


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"(e\.g|i\.e|et al|cf|vs|Fig|Sec|Eq)\.", lambda m: m.group(1) + "<D>", text or "")
    return [s.replace("<D>", ".").strip() for s in _SENT.split(text)
            if len(s.strip()) > 15 and not s.strip().startswith("#")]
