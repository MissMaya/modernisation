# 06 · Reviewer-associated variation

## What this section is trying to show

This section checks whether the reviewer evidence used to identify possible
prompt improvements is distributed across reviewer packets or concentrated in
particular packets.

It does not measure reviewer bias. Each document was reviewed once. Differences
between reviewers may therefore reflect the documents, archives and models
assigned to them as well as differences in annotation practice.

## Measures

Document annotation rates are distinct included annotations per 1,000
modernised tokens. Documents without annotation data are excluded rather than
treated as having no annotations.

The recurrent-span analysis uses the spans identified in Stage 05. One
annotation is counted once for a span even when several error labels were
attached to it. Reviewer counts are reported separately by model in the CSV,
but the heatmap combines models because it is showing the distribution of the
review evidence rather than comparing model performance.

## Inputs

- `outputs/document_analysis.csv`: document, reviewer, model, archive and token
  information.
- `outputs/annotation_analysis.csv`: included reviewer annotations.
- `analysis_outputs/05_recurrent_annotated_text/tables/recurrent_annotated_text.csv`:
  recurrent spans and their Stage 05 ranks.

## Tables produced by this script

- `reviewer_summary.csv`: document and token exposure, annotation totals,
  pooled annotation rates, model allocation and archive coverage by reviewer.
- `reviewer_document_rates.csv`: the document-level data plotted in the first
  figure.
- `reviewer_error_category_profile.csv`: annotation and document counts for
  each reviewer and error category.
- `recurrent_span_reviewer_distribution.csv`: annotation counts for every
  recurrent span by reviewer and model.

## Figures produced by this script

- `reviewer_annotation_rates`: document annotation rates by reviewer and model.
- `recurrent_span_reviewer_distribution`: reviewer distribution of the ten
  most frequently annotated recurrent spans from Stage 05.

Both figures are saved as PNG and SVG.
