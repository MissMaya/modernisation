# 05 · Actionable failure patterns

## What this section is trying to show

This section identifies exact forms that repeatedly received the same error
category and sub-rule. Its purpose is to produce evidence that can be checked
and converted into clearer prompt instructions or examples.

This is still reviewer feedback, not a list of confirmed model errors. Human
inspection is required before any candidate becomes a prompt change.

## What counts as one pattern

A pattern combines:

- model;
- error category;
- sub-rule; and
- normalised annotated text.

The annotated text is converted to lower case, surrounding space is removed and
repeated whitespace is collapsed. Accents and punctuation are retained.

## How prompt candidates are identified

For a particular model, a pattern is marked as a candidate when it appears in
at least:

- 2 documents;
- 2 reviewers; and
- 2 archives.

These are transparent breadth checks, not significance tests. They stop one
document, reviewer or archive from determining a proposed prompt change.

## Inputs

- `outputs/document_analysis.csv`: supplies model, archive, reviewer and token
  exposure for each reviewed document.
- `outputs/annotation_analysis.csv`: supplies the retained annotation text,
  categories and sub-rules.

## Tables produced by this script

- `failure_pattern_summary.csv`: every observed model-specific pattern, with
  annotation, document, reviewer and archive coverage.
- `prompt_candidates_for_review.csv`: the underlying annotations for candidate
  patterns, retained as examples for human inspection.

## Figure produced by this script

- `prompt_candidate_rule_concentration`: the number of distinct recurring
  annotated forms within each category and sub-rule, compared by model. The
  detailed forms themselves remain in the two CSVs. The figure is omitted if
  no pattern meets the breadth checks.

The figure is saved as both PNG and SVG.

## Interpretation

Patterns supported across documents, reviewers and archives are stronger
candidates for prompt clarification than high counts concentrated in one
context. They still require adjudication: the reviewer may be identifying a
genuine model failure, an ambiguous instruction or an inconsistent review
decision.
