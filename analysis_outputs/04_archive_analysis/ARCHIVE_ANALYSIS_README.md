# 04 · Archive analysis

## Aim of analysis

To show whether the high-burden documents identified in Stage 3 are
concentrated in particular archives.

## Archive analysis

The figure retains one marker per reviewed document. Separate model panels use
the Stage 3 model-specific upper outlier classifications. Archives are ordered
alphabetically.

The sample contains no more than five documents from any archive. Archive
patterns are therefore descriptive and may reflect document characteristics,
model allocation and reviewer behaviour. Reviewer effects are not adjusted in
this stage.

`archive_context_summary.csv` uses one row per archive-and-model combination.
It reports document, token and annotation totals; the resulting model-specific
annotation rate; reviewer coverage; and counts of the three Stage 3 outlier
types. It also contains an archive-prefix summary using the text before the
first underscore in the archive name. Prefix groups must be verified before
being treated as institutions or collections.

## Input

- `analysis_outputs/03_document_diagnostics/tables/document_diagnostics.csv`:
  one row for every reviewed document in the Stage 3 scatter plot.

## Tables produced by this script

- `archive_context_summary.csv`: archive-and-model and archive-prefix-and-model
  exposure, annotation rates, reviewer coverage and Stage 3 outlier counts.

## Figure produced by this script

- `high_burden_documents_by_archive`: all reviewed documents by archive and
  model, with the Stage 3 upper outliers identified by type.
