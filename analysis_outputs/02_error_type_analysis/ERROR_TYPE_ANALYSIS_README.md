# 02 · Error-type analysis

## What this section is trying to show

This section identifies which types of modernisation error reviewers assigned
most frequently and compares category-specific rates between the two models.

Only assignments retained for analysis are used. Category totals count each
annotation once within a category, even when it has multiple fields in that
category. Sub-rule totals count the individual category-field assignments.

## Inputs

- `outputs/document_analysis.csv`: supplies reviewed token totals for each model.
- `outputs/annotation_analysis.csv`: supplies the included error-category and
  sub-rule assignments.

## Tables produced by this script

- `error_type_summary.csv`: category and sub-rule counts and rates, overall and
  by model.
- `subrule_priority_summary.csv`: one ranked row per sub-rule, including the
  overall burden, documents affected and the observed rate for each model.

## Figures produced by this script

- `error_category_rates_by_model`: category-specific annotation rates for the
  two models.
- `priority_subrule_rates_by_model`: model-specific rates for the sub-rules that
  together account for at least 80% of included assignments. Sub-rules tied at
  the cutoff are retained.

Each figure is saved as both PNG and SVG.

## Interpretation

The figures describe categories and sub-rules assigned by reviewers. They do not show
how many opportunities each model had to apply each individual modernisation
rule, and they do not yet adjust for archive or reviewer effects.
