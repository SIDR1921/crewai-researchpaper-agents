"""The agents and tasks (Stage 5's cast).

Roles are kept narrow on purpose: the Extractor never searches, the Writer never
verifies, and only the Verifier can call the entailment tool. The orchestration
that strings them together lives in :mod:`litreview.flow`.

Each agent is built from the one LLM configuration in :mod:`litreview.llm`, so a
provider switch is a one-line change and every call is still logged.
"""
from __future__ import annotations

from functools import lru_cache

from .llm import get_llm
from .schemas import PaperRecord
from .tools import LIBRARIAN_TOOLS, VERIFIER_TOOLS

AGENT_SPECS: dict[str, dict] = {
    "librarian": {
        "role": "Research Librarian",
        "goal": ("Assemble a corpus of real, relevant papers on the given topic and confirm "
                 "that every one of them exists."),
        "backstory": ("You are a scientific librarian. You are judged on precision: a corpus "
                      "with one fabricated or irrelevant paper in it is a failed corpus. You "
                      "always verify a title before you pass it on."),
        "tools": LIBRARIAN_TOOLS,
    },
    "extractor": {
        "role": "Claim Extractor",
        "goal": ("Turn a paper's text into a structured record whose every claim quotes the "
                 "passage it came from, word for word."),
        "backstory": ("You read papers for a systematic-review team. Your extractions are "
                      "audited against the source text, so you copy quotes exactly and leave "
                      "fields empty rather than guessing."),
        "tools": [],
    },
    "verifier": {
        "role": "Evidence Verifier",
        "goal": ("Decide, for each claim and each review sentence, whether the evidence really "
                 "supports it."),
        "backstory": ("You are an adversarial checker. You do not judge plausibility; you call "
                      "the entailment tool and report what it says, including when that "
                      "contradicts what the writer intended."),
        "tools": VERIFIER_TOOLS,
    },
    "synthesist": {
        "role": "Synthesist and Adjudicator",
        "goal": ("Build the consensus picture across papers and explain each apparent "
                 "contradiction using only the evidence."),
        "backstory": ("You have reviewed enough papers to know that most contradictions are "
                      "different experimental settings wearing a disguise, and that some are "
                      "real. You distinguish the two and say when you cannot."),
        "tools": [],
    },
    "gap_analyst": {
        "role": "Research Gap Analyst",
        "goal": ("Judge which candidate gaps in the literature are real and worth pursuing, "
                 "arguing both sides."),
        "backstory": ("You have seen many 'gaps' that were really dead ends or artefacts of a "
                      "small sample. You argue against every candidate before you endorse it."),
        "tools": [],
    },
    "writer": {
        "role": "Review Writer",
        "goal": ("Write a related-work section in which every sentence is traceable to a "
                 "verified claim."),
        "backstory": ("You write survey sections for journals. You attach the claim ids behind "
                      "every sentence, because a sentence you cannot source will be cut by the "
                      "verifier."),
        "tools": [],
    },
}


@lru_cache(maxsize=None)
def agent(name: str, size: str = "large"):
    """Build (and cache) one configured agent."""
    from crewai import Agent
    spec = AGENT_SPECS[name]
    return Agent(
        role=spec["role"], goal=spec["goal"], backstory=spec["backstory"],
        tools=list(spec["tools"]), llm=get_llm(size=size),
        allow_delegation=False, verbose=False, max_iter=6,
    )


def all_agents() -> dict:
    return {n: agent(n) for n in AGENT_SPECS}


# ---------------------------------------------------------------------- tasks

def extractor_task(paper: dict, text: str) -> PaperRecord:
    """The Extractor as a CrewAI task with ``output_pydantic=PaperRecord``."""
    from crewai import Crew, Process, Task
    from .extract import SYSTEM, build_prompt

    task = Task(
        description=f"{SYSTEM}\n\n{build_prompt(paper, text)}",
        expected_output="A PaperRecord whose every claim quotes its evidence verbatim.",
        agent=agent("extractor"),
        output_pydantic=PaperRecord,
    )
    crew = Crew(agents=[agent("extractor")], tasks=[task], process=Process.sequential,
                verbose=False)
    out = crew.kickoff()
    if getattr(out, "pydantic", None) is not None:
        return out.pydantic
    from .extract import coerce_record
    from .llm import parse_json
    return coerce_record(parse_json(str(out)), paper)


def librarian_task(topic: str, n_papers: int, seeds: list[str] | None = None) -> str:
    """Corpus assembly driven by the Librarian agent (used by the Flow's first step)."""
    from crewai import Crew, Process, Task
    seed_line = (f"Start from these seed papers and snowball outward: {', '.join(seeds)}.\n"
                 if seeds else "")
    task = Task(
        description=(
            f"Assemble a corpus of {n_papers} papers on: {topic}\n{seed_line}"
            "Use search_literature, then snowball_references from the most-cited hits. "
            "Verify every title with verify_reference_exists and drop any that is not found. "
            "Rank by relevance to the topic first, then by citations per year since "
            "publication.\n"
            'Return ONLY JSON: {"papers": [{"title": "...", "year": 2024, '
            '"openalex_id": "...", "why_relevant": "<=15 words"}]}'),
        expected_output="A JSON object with a 'papers' array of verified, ranked papers.",
        agent=agent("librarian"),
    )
    crew = Crew(agents=[agent("librarian")], tasks=[task], process=Process.sequential,
                verbose=False)
    return str(crew.kickoff())
