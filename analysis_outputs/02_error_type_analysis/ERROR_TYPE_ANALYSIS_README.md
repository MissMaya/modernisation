# 02 · Error-type analysis

## Aim of analysis

To identify which error labels reviewers assigned most frequently and to show how
the rates of annotation differed between models.

Category totals count each annotation once per category. Category-sub-rule totals count each
distinct label attached to an annotation. For example, if one annotation has two different 
Abreviaturas sub-rule labels, it contributes one count to the Abreviaturas category total but 
one count to each of the two sub-rule totals. Therefore, sub-rule totals may exceed the 
corresponding category total.

## Inputs

- `outputs/document_analysis.csv`: supplies reviewed token totals for each model.
- `outputs/annotation_analysis.csv`: supplies the included error-category and
  sub-rule assignments.

## Tables produced by this script

- `error_type_summary.csv`: counts and rates for categories and category-sub-rule
  labels, overall and by model.
- `subrule_frequency_summary.csv`: category-sub-rule labels ranked by frequency,
  with document counts and model-specific rates.

## Figures produced by this script

- `error_category_rates_by_model`: category-specific annotation rates by model.
- `frequent_error_label_rates_by_model`: model-specific rates for the smallest
  set of category-sub-rule labels accounting for at least 80% of included
  assignments. The 80% cutoff is used only to keep the figure readable; the
  CSV contains every label. Labels tied at the cutoff are retained.
