# 01 · Descriptive model overview

## What this section is trying to show

This section shows what reviewers recorded for documents produced by each
modernisation model.

The main measure is **distinct included annotations per 1,000 modernised
tokens**. Each reviewer-marked annotation is counted once even when it has more
than one category or field assignment.

Documents with unavailable annotation data are excluded from the comparison;
they are not treated as documents with zero errors.

## Input

- `outputs/document_analysis.csv`: one row per sample document.

## Table produced by this script

- `model_observed_summary.csv`: reviewed documents, text length, annotation
  totals, annotation rates and zero-annotation documents for each model.

## Figures produced by this script

- `annotation_rate_by_model`: document-level annotation rates for each model.

Each figure is saved as both PNG and SVG.

## Interpretation

This is a descriptive comparison of human-review outcomes. A higher annotation
rate means that reviewers marked more apparent problems; it does not yet prove
that the model was worse. Later analysis will examine error types, reviewer
behaviour, archive composition and annotation decisions before statistical
modelling is undertaken.
