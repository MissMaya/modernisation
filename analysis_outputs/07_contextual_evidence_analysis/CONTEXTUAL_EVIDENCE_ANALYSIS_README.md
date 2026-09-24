# 07 · Contextual evidence for prompt revision

## Purpose

This section moves from isolated annotated forms to evidence in context. It
asks which recurring reviewer-labelled behaviours are sufficiently broad and
consistent to merit human consideration when the modernisation prompt is
revised.

## Complete CSV outputs

- `contextual_annotation_evidence.csv` contains every distinct included
  annotation--rule assignment, its model and review metadata, modernised
  context, and approximately aligned source context where alignment succeeds.
- `form_rule_pattern_summary.csv` contains every form--category--sub-rule
  pattern. Overall rows and model-specific rows are distinguished by
  `analysis_scope`; this long format is intended for later computation.
- `context_extraction_issues.csv` records unavailable text, invalid offsets and
  source-alignment problems. These issues do not stop the stage.

## Candidate and figure selection

A form--rule pattern enters the candidate pool when it occurs in at least
2 documents and was recorded by at least
2 reviewers. These are the minimum conditions needed
for recurrence beyond one document and one reviewer's practice.

There is no minimum annotation count, archive count or fixed agreement
percentage. Archive breadth may reveal either a general pattern or a legitimate
archive-specific problem, so it is evidence rather than a gate. Raw rule
agreement is accompanied by its 95% Wilson lower bound, which discounts
apparently perfect agreement when it rests on very few observations.

Candidates are ordered lexicographically by affected documents, affected
reviewers, the 95% Wilson lower bound for rule consistency, affected archives,
and distinct annotations. The figure shows the first 6
candidates in that ordering. This is solely a presentation limit. Every
candidate and every other pattern remains in the CSV.

## Context rules

- Modernised context is extracted from the annotation's recorded `start` and
  `end` offsets; the marked span appears inside `[[double brackets]]`.
- The same offsets are never applied to the source transcription.
- Source context is located through character-sequence alignment between the
  source and modernised documents and is labelled approximate.
- The alignment ratio and extraction status remain in the evidence CSV so weak
  contextual matches can be filtered or inspected.

## Prompt use

The figure communicates candidate behaviours. The CSVs are the evidence base
for prompt construction: inspect several examples, decide whether the reviewer
label is correct, identify the desired transformation, and only then formulate
a rule or few-shot example. Recurrence alone does not prove model failure.
