#!/usr/bin/env python
"""Generate docs/labelling_guide.md from the taxonomy module plus real examples.

The definitions and tie-break rules are read from ``litreview.relations.taxonomy``
rather than retyped, so the guide a human labels by and the prompt the LLM labels
by cannot drift apart. The examples come from ``docs/labelling_examples.json``,
which holds real mined contexts chosen by hand.

    python scripts/build_labelling_guide.py
"""
from __future__ import annotations

import json
from pathlib import Path

from litreview.relations.taxonomy import (ACL_ARC_LIMITS, ACL_ARC_MAP, DECISION_RULES,
                                          DEFINITIONS, LABELS, SCICITE_LIMITS, SCICITE_MAP)

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "docs" / "labelling_examples.json"
OUT = REPO / "docs" / "labelling_guide.md"


def main() -> int:
    ex = json.loads(EXAMPLES.read_text()) if EXAMPLES.exists() else {}
    lines = [
        "# Citation-relation labelling guide",
        "",
        "One page. Read the whole thing before labelling anything.",
        "",
        "## The task",
        "",
        "You are shown a **citation context**: one to three sentences from a paper's",
        "related-work section, with the citation under question rewritten as `[[CITED]]`",
        "and every other citation in the same sentence rewritten as `[OTHER]`. You are also",
        "shown the cited paper's title. Assign exactly one of seven labels describing what",
        "the citing paper asserts about the cited paper.",
        "",
        "Label what the **text says**, not what you know about the papers. If the context",
        "does not support a specific relation, the answer is `background`, not a guess.",
        "",
        "## The labels",
        "",
    ]
    for label in LABELS:
        lines += [f"### `{label}`", "", DEFINITIONS[label], ""]
        items = (ex.get(label) or [])[:2]
        if items:
            for it in items:
                lines += [
                    f"> {it['context'].strip()}",
                    "",
                    f"*Cited paper:* {it.get('cited_title') or '(unknown)'}  ",
                    f"*Why:* {it['why']}",
                    "",
                ]
        else:
            lines += ["_(No example recorded yet. Add one to docs/labelling_examples.json "
                      "and re-run scripts/build_labelling_guide.py.)_", ""]

    lines += [
        "## When two labels seem to apply",
        "",
        "Apply these in order; the first that fires wins.",
        "",
    ]
    lines += [f"{i}. {r}" for i, r in enumerate(DECISION_RULES, 1)]
    lines += [
        "",
        "The hard boundary in practice is `compares-against` against `contrasts`. Running a",
        "baseline and reporting that you beat it is `compares-against`: that is a measurement,",
        "not a disagreement. Saying that the cited work's conclusion does not hold, that its",
        "assumption is wrong, or that your result contradicts theirs is `contrasts`. The",
        "downstream contradiction map depends on this distinction, so when a sentence does both",
        "(\"unlike X, which assumes a single reference, we ... and outperform it by 3 ROUGE\"),",
        "rule 1 applies and the label is `contrasts`.",
        "",
        "The second-hardest is `uses-method` against `extends`. If the cited approach is the",
        "thing being modified, it is `extends`; if it is simply being used as a component, it is",
        "`uses-method`.",
        "",
        "## Procedure",
        "",
        "1. Read the whole context, not only the sentence with `[[CITED]]`.",
        "2. Decide what is asserted about `[[CITED]]` specifically, ignoring `[OTHER]` citations.",
        "3. If more than one label fits, apply the tie-break rules above in order.",
        "4. If the context is truncated, garbled by LaTeX conversion, or about a different",
        "   citation entirely, label it `background` and note it; do not guess.",
        "",
        "## Relationship to published corpora",
        "",
        "Two existing corpora are mapped onto this taxonomy to provide extra training signal.",
        "The mappings are lossy, and mapped rows are therefore **never** used as evaluation gold.",
        "",
        "| Source label | Mapped to | ",
        "|---|---|",
    ]
    for src, dst in SCICITE_MAP.items():
        lines.append(f"| SciCite `{src}` | `{dst}` |")
    for src, dst in ACL_ARC_MAP.items():
        lines.append(f"| ACL-ARC `{src}` | {'`' + dst + '`' if dst else '_dropped_'} |")
    lines += [
        "",
        f"**SciCite limits.** {SCICITE_LIMITS}",
        "",
        f"**ACL-ARC limits.** {ACL_ARC_LIMITS}",
        "",
        "## Gold set",
        "",
        "`scripts/sample_gold.py` writes `data/gold_blind.jsonl`, which contains the contexts",
        "with **no** LLM label attached; that is what makes the human labels independent. Fill",
        "in `human_label` for each row and save the file as `data/gold_labelled.jsonl`, then run",
        "`scripts/score_gold.py`.",
        "",
        "The sample is stratified over the LLM's predicted label so that the rare relations",
        "appear at all. Agreement measured on it is therefore **not** an estimate of accuracy on",
        "a uniformly random citation context: it deliberately over-weights the classes the",
        "labeller finds hardest, and should be read as a lower bound. The report says so where",
        "the number appears.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    missing = [l for l in LABELS if len(ex.get(l) or []) < 2]
    print(f"wrote {OUT}")
    if missing:
        print("labels still needing two real examples:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
