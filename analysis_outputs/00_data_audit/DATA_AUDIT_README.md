# 00 · Data and allocation audit

## Aim of analysis

Checks whether the analysis data are complete and extracts the split of documents
and tokens between models and reviewers. 

The documents were allocated randomly, but randomisation does not guarantee
perfect balance. The model groups can contain different
numbers or lengths of documents, reviewers can receive different mixtures of
model outputs, and archives can be concentrated under particular models or
reviewers. These patterns are considered when interpreting any descriptive
analyses and in any later statistical modelling.

Note that a completed review with no annotations is distinct from a document whose
annotation data are unavailable. Missing annotation data are not counted as
zero errors.

## Inputs

- `outputs/document_analysis.csv`: one row per sample document.
- `outputs/annotation_analysis.csv`: one row per category-field assignment.

## Tables produced by this script

- `audit_summary.csv`: headline counts for data completeness, distinct
  annotations, and included or excluded error-label assignment rows. A single
  annotation can have more than one assignment row.
- `allocation_summary.csv`: model totals, model allocation within reviewer
  packets, model allocation within archives, and archive concentration within
  reviewer packets.
- `documents_requiring_attention.csv`: documents with unavailable data or
  excluded annotation assignments.

## Figures produced by this script

- `model_sample_and_token_exposure`: reviewed document and token exposure for
  each model.
- `model_allocation_by_reviewer_packet`: the model split received by each
  reviewer.
