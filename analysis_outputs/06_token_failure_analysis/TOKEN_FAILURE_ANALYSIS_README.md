# 06 · Annotated forms that could inform the prompt

## Purpose

This section combines the token or phrase marked by a reviewer with its error
category and sub-rule. It therefore shows both *what* was marked and *why* the
reviewer believed it was wrong.

## Figure selection

An annotated form is a cross-context prompt candidate for one model when it occurs
in at least 2 documents, 2 reviewer packets and
2 archives. Candidates are ranked by distinct annotation count
and retained until at least 80% of qualifying annotations
is represented. Ties at the cutoff are included. Bar-end labels show
`annotations · documents`.

Selection is based on frequency, but the selected forms are displayed
alphabetically so individual forms are easy to find. Beneath each bold italic
form, the figure shows the category–sub-rule combination that reviewers
assigned to that form most frequently across both model groups. This is a
description of the reviewer labels, not a claim that the label is correct.

## Outputs

- `annotated_form_summary.csv` contains every annotated form and its annotation,
  document, archive and reviewer coverage by model. It also records the most
  commonly assigned rule and all other rules assigned to that form.
- `token_examples_for_review.csv` contains individual examples with modernised
  context and document metadata for adjudication.
- Three numbered `prompt_candidate_forms_and_rules` figures divide the
  alphabetically ordered forms into approximately equal thirds.

Each figure uses an x-axis range suited to the values in that alphabetical
section. This keeps shorter bars readable; use the printed annotation counts,
rather than apparent bar length across different figures, for comparisons.

## Caution

These are reviewer-assigned apparent errors, not automatically genuine model
errors. Inspect the example table alongside the source transcription before
turning a pattern into a prompt instruction or example. Annotation offsets
belong to the modernised text and cannot safely slice the source text at the
same positions.
