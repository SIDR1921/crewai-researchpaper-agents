"""Pydantic models that every stage reads and writes.

These are the contract between the Extractor, the Verifier, the synthesis step
and the Writer, and they are what the distilled model in Stage 6 must emit.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ClaimType(str, Enum):
    finding = "finding"
    method = "method"
    limitation = "limitation"
    comparison = "comparison"


class Relation(str, Enum):
    """Citation-relation taxonomy (Stage 2)."""
    background = "background"
    uses_method = "uses-method"
    uses_data = "uses-data"
    extends = "extends"
    compares_against = "compares-against"
    supports = "supports"
    contrasts = "contrasts"


RELATION_LABELS: list[str] = [r.value for r in Relation]


class Evidence(BaseModel):
    section: str = Field(description="Where the passage came from, e.g. 'abstract' or 'results'.")
    passage: str = Field(description="Verbatim quote from the source text. Never paraphrase.")


class Claim(BaseModel):
    claim_id: str = Field(description="Short unique id within the paper, e.g. 'c1'.")
    text: str = Field(description="One sentence stating the claim.")
    type: ClaimType
    evidence: Evidence


class Metric(BaseModel):
    name: str = Field(description="Metric name as written, e.g. 'ROUGE-L' or 'accuracy'.")
    value: float | None = Field(default=None, description="Numeric value if one is reported.")
    setting: str | None = Field(default=None, description="Dataset/split/configuration it applies to.")
    higher_is_better: bool = True


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    year: int | None = None
    task: str | None = Field(default=None, description="The problem the paper addresses.")
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("methods", "datasets", "limitations", mode="before")
    @classmethod
    def _listify(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v

    @field_validator("year", mode="before")
    @classmethod
    def _year(cls, v):
        if v in (None, "", "n/a", "unknown"):
            return None
        try:
            y = int(str(v)[:4])
        except (TypeError, ValueError):
            return None
        return y if 1950 <= y <= 2100 else None


class ClaimVerdict(BaseModel):
    """Per-claim output of the Verifier (Stage 3)."""
    claim_id: str
    paper_id: str
    quote_found: bool = Field(description="Did the evidence passage really occur in the source?")
    quote_score: float = Field(description="Best fuzzy partial-ratio of the passage against the source, 0-100.")
    entail_prob: float = Field(description="P(entailment) of claim given evidence, from the NLI model.")
    contradiction_prob: float = 0.0
    verdict: Literal["supported", "partially", "unsupported"]


class ReviewSentence(BaseModel):
    text: str
    supporting_claim_ids: list[str] = Field(default_factory=list)
    cited_paper_ids: list[str] = Field(default_factory=list)


class SentenceVerdict(BaseModel):
    text: str
    cited_paper_ids: list[str] = Field(default_factory=list)
    entail_prob: float = 0.0
    verdict: Literal["supported", "partially", "unsupported"] = "unsupported"
    best_evidence: str | None = None


class Review(BaseModel):
    topic: str
    sentences: list[ReviewSentence] = Field(default_factory=list)
    markdown: str = ""


class RelationEdge(BaseModel):
    citing_id: str
    cited_id: str
    relation: Relation
    prob: float = 0.0
    sentence: str = ""


class Contradiction(BaseModel):
    task: str | None = None
    dataset: str | None = None
    metric: str | None = None
    papers: list[str] = Field(default_factory=list)
    values: list[dict] = Field(default_factory=list)
    kind: Literal["value-disagreement", "ranking-disagreement", "classifier-contrasts"]
    spread: float | None = None
    adjudication: str | None = None


class Gap(BaseModel):
    kind: Literal["empty-cell", "sparse-cell", "untested-comparison"]
    method_family: str | None = None
    dataset: str | None = None
    n_papers: int = 0
    candidate: str = Field(description="One-line statement of the gap.")
    score: float = 0.0
    argument_for: str | None = None
    argument_against: str | None = None
    verdict: Literal["real-gap", "not-a-gap", "uncertain"] | None = None
