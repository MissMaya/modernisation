# 03 · Document diagnostics

## What this section is trying to show

This section identifies the individual documents behind the model- and
error-type patterns reported in Sections 1 and 2.

It distinguishes between documents with many annotations and documents with a
high annotation rate. This matters because a short document can have a high
rate per 1,000 tokens even when its absolute annotation count is moderate.

## How documents are selected for closer review

`documents_for_review.csv` contains documents in the highest 10% for either:

- distinct included annotation count; or
- distinct included annotations per 1,000 modernised tokens.

The table shows which criterion selected each document and retains its
modernised token count. Document length should be considered when interpreting
high rates because a moderate annotation count divided by a small number of
tokens can produce a high rate per 1,000 tokens. No separate short-document
cutoff is imposed.

## Document-family rule

The family identifier is created by removing only a final suffix of the form
`_duplicated_<number>` from `filename_stem`. All other stems remain unchanged.
The table states whether a document belongs to a family represented more than
once in the reviewed sample.

## Inputs

- `outputs/document_analysis.csv`: one row per sample document.
- `outputs/annotation_analysis.csv`: included category and sub-rule assignments.

## Tables produced by this script

- `document_diagnostics.csv`: all reviewed documents, with annotation burden,
  document context and dominant error types.
- `documents_for_review.csv`: the shorter list selected by the two highest-10%
  criteria described above.

## Figures produced by this script

- `document_length_and_annotation_burden`: annotation counts against document
  length, with numbered flagged documents identified in an adjacent key.
- `flagged_document_error_profiles`: category composition for the documents
  flagged for closer review.

Each figure is saved as both PNG and SVG.

## Interpretation

These are diagnostic outputs, not model rankings. Archive, reviewer, error
type, length and repeated-family membership are retained so that apparent
model failures can be checked for recurring textual patterns or human-review
effects in the following analyses.
