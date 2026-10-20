"""The citation-relation taxonomy, its definitions, and the label-mapping tables.

The definitions live here rather than in a prompt string so the LLM labeller, the
human labelling guide (``docs/labelling_guide.md``) and the SciBERT training code
all use exactly the same wording.
"""
from __future__ import annotations

from ..schemas import RELATION_LABELS, Relation

DEFINITIONS: dict[str, str] = {
    "background": (
        "The cited work is mentioned to set context, define a problem, or point at a body "
        "of literature. The citing paper neither reuses anything specific from it nor "
        "takes a position on it."
    ),
    "uses-method": (
        "The citing paper adopts, implements or builds on a model, algorithm, architecture, "
        "loss, metric or tool introduced by the cited work."
    ),
    "uses-data": (
        "The citing paper trains or evaluates on a dataset, corpus, benchmark or annotation "
        "scheme introduced or distributed by the cited work."
    ),
    "extends": (
        "The citing paper modifies, generalises or improves the cited work's own approach, "
        "and says so. Stronger than uses-method: the cited approach is the thing being changed."
    ),
    "compares-against": (
        "The cited work is a baseline or competing system that the citing paper measures "
        "itself against, without asserting who is right about a scientific question."
    ),
    "supports": (
        "The citing paper states that its own findings, or another work's, agree with or "
        "confirm the cited work's finding."
    ),
    "contrasts": (
        "The citing paper states a disagreement with the cited work: a contradictory result, "
        "a criticised assumption, or a limitation it identifies in the cited work."
    ),
}

#: Applied in order; the first rule that fires wins. This resolves the common
#: overlaps (a baseline you also reimplement, a dataset paper you also criticise).
DECISION_RULES: list[str] = [
    "If the sentence expresses disagreement, a contradictory finding, or criticism of the "
    "cited work, label contrasts -- even if the cited work is also a baseline.",
    "If the sentence states agreement or confirmation between findings, label supports.",
    "If the cited work's own approach is the thing being modified or improved, label extends.",
    "If the cited work is being measured against as a baseline or competitor, label compares-against.",
    "If a dataset, corpus or benchmark from the cited work is used, label uses-data.",
    "If a method, model, architecture, loss, metric or tool from the cited work is used, "
    "label uses-method.",
    "Otherwise label background.",
]

#: SciCite (Cohan et al., NAACL 2019) has three coarse classes. Only `background`
#: maps cleanly; `method` and `result` are each ambiguous across two of our labels,
#: so they seed the model with a coarse signal and are never used as gold.
SCICITE_MAP: dict[str, str | None] = {
    "background": "background",
    "method": "uses-method",   # conflates uses-method / extends
    "result": "compares-against",  # conflates compares-against / supports / contrasts
}
SCICITE_LIMITS = (
    "SciCite's `method` class does not separate adopting a method from extending it, and its "
    "`result` class covers comparison, agreement and disagreement alike. Mapped rows are used "
    "only as extra training signal, never as evaluation gold."
)

#: ACL-ARC (Jurgens et al., TACL 2018) has six classes that are closer to ours.
ACL_ARC_MAP: dict[str, str | None] = {
    "Background": "background",
    "Uses": "uses-method",          # Uses covers both method and data
    "CompareOrContrast": "compares-against",  # conflates compares-against / contrasts
    "Extends": "extends",
    "Motivation": "background",
    "Future": None,                 # no counterpart: dropped
}
ACL_ARC_LIMITS = (
    "ACL-ARC's `Uses` does not distinguish method from data, and `CompareOrContrast` merges "
    "neutral comparison with explicit disagreement. `Future` has no counterpart and is dropped."
)

LABELS = RELATION_LABELS
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}


def normalise_label(raw: str | None) -> str | None:
    """Accept the small variations an LLM produces ('uses_method', 'Compares Against')."""
    if not raw:
        return None
    s = str(raw).strip().lower().replace("_", "-").replace(" ", "-")
    s = s.strip("`'\".")
    if s in LABEL_TO_ID:
        return s
    aliases = {
        "uses-methods": "uses-method", "method": "uses-method", "uses": "uses-method",
        "uses-dataset": "uses-data", "data": "uses-data", "dataset": "uses-data",
        "compare": "compares-against", "comparison": "compares-against",
        "compares": "compares-against", "compare-against": "compares-against",
        "compares-with": "compares-against", "baseline": "compares-against",
        "contrast": "contrasts", "contradicts": "contrasts", "disagrees": "contrasts",
        "support": "supports", "confirms": "supports", "agrees": "supports",
        "extend": "extends", "extension": "extends",
        "context": "background", "backgrounds": "background", "motivation": "background",
    }
    return aliases.get(s)


def taxonomy_prompt() -> str:
    """The shared definition block injected into every labelling prompt."""
    defs = "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
    rules = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(DECISION_RULES))
    return (
        "Citation-relation labels:\n" + defs +
        "\n\nWhen two labels seem to apply, use these tie-break rules in order:\n" + rules
    )


assert set(DEFINITIONS) == set(LABELS), "taxonomy definitions out of sync with the schema"
assert all(v is None or v in LABELS for v in {**SCICITE_MAP, **ACL_ARC_MAP}.values())
