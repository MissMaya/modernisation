# 03 · Document diagnostics

## Aim of analysis

To examine annotation count in relation to document length and identify
documents with unusually high annotation counts or annotation rates.

It anlayses both documents with many annotations and documents with a
high annotation rate. This matters because a short document can have a high
rate per 1,000 tokens even when its absolute annotation count is moderate.

## Upper outlier thresholds

`upper_outlier_documents.csv` contains documents exceeding the upper outlier
threshold for either:

- distinct included annotation count; or
- distinct included annotations per 1,000 modernised tokens.

The upper outlier threshold is Q3 + 1.5 x IQR, where Q3 is the 75th percentile
and IQR is the range between the 25th and 75th percentiles. It is calculated
separately for each measure and model.

These thresholds identify documents that are unusual relative to other
documents produced by the same model. 

The numbered documents are ordered to try to flag the most useful to look at for 
prompt refinement. Documents exceeding both count and rate thresholds come first, 
followed by count-only and then rate-only outliers.
Within each group, documents are ordered by annotation count and then annotation
rate, both descending. This puts evidence supported by both measures first and
places rate-only cases, which can be affected by short documents, last. The
numbers indicate review order, not model performance.

`document_diagnostics.csv` contains one row for every document plotted in the
scatter plot. It reports the model, document length, annotation count,
annotation rate, model-specific thresholds and whether each document exceeds
the count threshold, the rate threshold, both or neither.

## Inputs

- `outputs/document_analysis.csv`: one row per sample document.
- `outputs/annotation_analysis.csv`: included category and sub-rule assignments.

## Tables produced by this script

- `document_diagnostics.csv`: all reviewed documents, with annotation burden,
  document context and the most frequently assigned error category and
  category-sub-rule. Repeated-document family fields are retained for later
  analysis.
- `upper_outlier_documents.csv`: documents exceeding either upper outlier
  threshold, with the criterion recorded.

## Figures produced by this script

- `document_length_and_annotation_burden`: annotation counts against document
  length, with numbered upper outliers identified in an adjacent key.
- `upper_outlier_document_error_profiles`: category composition for the same
  numbered documents.
