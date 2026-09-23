# 00 · Data and allocation audit

## What this section is trying to show

This section establishes whether the analysis data are complete and describes
the realised allocation of models, archives and reviewers.

The documents were allocated randomly, but randomisation does not guarantee
perfect balance in one realised sample. The model groups can contain different
numbers or lengths of documents, reviewers can receive different mixtures of
model outputs, and archives can be concentrated under particular models or
reviewers. These patterns must be considered in the adjusted comparison.

A completed review with no annotations is kept distinct from a document whose
annotation data are unavailable. Missing annotation data are not counted as
zero errors.

## Inputs

- `outputs/document_analysis.csv`: one row per sample document.
- `outputs/annotation_analysis.csv`: one row per category-field assignment.

## Tables produced by this script

- `audit_summary.csv`: headline counts for data completeness, reviewed texts
  and included or excluded annotation assignments.
- `allocation_summary.csv`: model totals, model allocation within reviewer
  packets, model allocation within archives, and archive concentration within
  reviewer packets.
- `documents_requiring_attention.csv`: documents with unavailable data or
  excluded annotation assignments.

## Figures produced by this script

- `model_sample_and_token_exposure`: reviewed document and token exposure for
  each model.
- `model_allocation_by_reviewer_packet`: the model mixture received by each
  reviewer.

Each figure is saved as both PNG and SVG.

## Interpretation

These outputs describe allocation and completeness, not model quality. The
substantive comparison begins in `01_model_comparison`, where archive,
reviewer and document length will be considered alongside model identity.
Because each reviewer corresponds to one packet, reviewer and packet are
treated as one allocation effect rather than two independent effects.
