# Citation-relation labelling guide

One page. Read the whole thing before labelling anything.

## The task

You are shown a **citation context**: one to three sentences from a paper's
related-work section, with the citation under question rewritten as `[[CITED]]`
and every other citation in the same sentence rewritten as `[OTHER]`. You are also
shown the cited paper's title. Assign exactly one of seven labels describing what
the citing paper asserts about the cited paper.

Label what the **text says**, not what you know about the papers. If the context
does not support a specific relation, the answer is `background`, not a guess.

## The labels

### `background`

The cited work is mentioned to set context, define a problem, or point at a body of literature. The citing paper neither reuses anything specific from it nor takes a position on it.

_(No example recorded yet. Add one to docs/labelling_examples.json and re-run scripts/build_labelling_guide.py.)_

### `uses-method`

The citing paper adopts, implements or builds on a model, algorithm, architecture, loss, metric or tool introduced by the cited work.

_(No example recorded yet. Add one to docs/labelling_examples.json and re-run scripts/build_labelling_guide.py.)_

### `uses-data`

The citing paper trains or evaluates on a dataset, corpus, benchmark or annotation scheme introduced or distributed by the cited work.

_(No example recorded yet. Add one to docs/labelling_examples.json and re-run scripts/build_labelling_guide.py.)_

### `extends`

The citing paper modifies, generalises or improves the cited work's own approach, and says so. Stronger than uses-method: the cited approach is the thing being changed.

_(No example recorded yet. Add one to docs/labelling_examples.json and re-run scripts/build_labelling_guide.py.)_

### `compares-against`

The cited work is a baseline or competing system that the citing paper measures itself against, without asserting who is right about a scientific question.

_(No example recorded yet. Add one to docs/labelling_examples.json and re-run scripts/build_labelling_guide.py.)_

### `supports`

The citing paper states that its own findings, or another work's, agree with or confirm the cited work's finding.

_(No example recorded yet. Add one to docs/labelling_examples.json and re-run scripts/build_labelling_guide.py.)_

### `contrasts`

The citing paper states a disagreement with the cited work: a contradictory result, a criticised assumption, or a limitation it identifies in the cited work.

_(No example recorded yet. Add one to docs/labelling_examples.json and re-run scripts/build_labelling_guide.py.)_

## When two labels seem to apply

Apply these in order; the first that fires wins.

1. If the sentence expresses disagreement, a contradictory finding, or criticism of the cited work, label contrasts -- even if the cited work is also a baseline.
2. If the sentence states agreement or confirmation between findings, label supports.
3. If the cited work's own approach is the thing being modified or improved, label extends.
4. If the cited work is being measured against as a baseline or competitor, label compares-against.
5. If a dataset, corpus or benchmark from the cited work is used, label uses-data.
6. If a method, model, architecture, loss, metric or tool from the cited work is used, label uses-method.
7. Otherwise label background.

The hard boundary in practice is `compares-against` against `contrasts`. Running a
baseline and reporting that you beat it is `compares-against`: that is a measurement,
not a disagreement. Saying that the cited work's conclusion does not hold, that its
assumption is wrong, or that your result contradicts theirs is `contrasts`. The
downstream contradiction map depends on this distinction, so when a sentence does both
("unlike X, which assumes a single reference, we ... and outperform it by 3 ROUGE"),
rule 1 applies and the label is `contrasts`.

The second-hardest is `uses-method` against `extends`. If the cited approach is the
thing being modified, it is `extends`; if it is simply being used as a component, it is
`uses-method`.

## Procedure

1. Read the whole context, not only the sentence with `[[CITED]]`.
2. Decide what is asserted about `[[CITED]]` specifically, ignoring `[OTHER]` citations.
3. If more than one label fits, apply the tie-break rules above in order.
4. If the context is truncated, garbled by LaTeX conversion, or about a different
   citation entirely, label it `background` and note it; do not guess.

## Relationship to published corpora

Two existing corpora are mapped onto this taxonomy to provide extra training signal.
The mappings are lossy, and mapped rows are therefore **never** used as evaluation gold.

| Source label | Mapped to | 
|---|---|
| SciCite `background` | `background` |
| SciCite `method` | `uses-method` |
| SciCite `result` | `compares-against` |
| ACL-ARC `Background` | `background` |
| ACL-ARC `Uses` | `uses-method` |
| ACL-ARC `CompareOrContrast` | `compares-against` |
| ACL-ARC `Extends` | `extends` |
| ACL-ARC `Motivation` | `background` |
| ACL-ARC `Future` | _dropped_ |

**SciCite limits.** SciCite's `method` class does not separate adopting a method from extending it, and its `result` class covers comparison, agreement and disagreement alike. Mapped rows are used only as extra training signal, never as evaluation gold.

**ACL-ARC limits.** ACL-ARC's `Uses` does not distinguish method from data, and `CompareOrContrast` merges neutral comparison with explicit disagreement. `Future` has no counterpart and is dropped.

## Gold set

`scripts/sample_gold.py` writes `data/gold_blind.jsonl`, which contains the contexts
with **no** LLM label attached; that is what makes the human labels independent. Fill
in `human_label` for each row and save the file as `data/gold_labelled.jsonl`, then run
`scripts/score_gold.py`.

The sample is stratified over the LLM's predicted label so that the rare relations
appear at all. Agreement measured on it is therefore **not** an estimate of accuracy on
a uniformly random citation context: it deliberately over-weights the classes the
labeller finds hardest, and should be read as a lower bound. The report says so where
the number appears.

