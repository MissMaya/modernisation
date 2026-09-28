# 01 · Descriptive model overview

## Aim of analysis

How did annotation rates very by model when measured per 1,000 tokens?

The main measure is **distinct included annotations per 1,000 modernised
tokens**. Each distinct annotation is counted once, including when the reviewer
attached more than one error-label assignment to it. An error-label assignment
is one category-sub-rule combination.

Documents with unavailable annotation data are excluded from the comparison;
they are not treated as documents with zero annotations.

## Input

- `outputs/document_analysis.csv`: one row per sample document.

## Table produced by this script

- `model_observed_summary.csv`: sample size, reviewed text length, distinct
  annotation totals, error-label assignment totals, annotation rates and
  zero-annotation documents for each model.

## PNG and SVG figures produced by this script

- `annotation_rate_by_model`: document-level annotation rates for each model.
