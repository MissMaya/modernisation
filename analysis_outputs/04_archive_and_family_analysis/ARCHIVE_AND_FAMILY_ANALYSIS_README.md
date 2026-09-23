# 04 · Archive and repeated-family analysis

## What this section is trying to show

This section asks whether the documents flagged in Stage 3 form recurring
patterns within particular archives or repeated versions of the same source
document.

## Archive analysis

Archive rates are calculated by summing distinct included annotations and
modernised tokens within each archive, then reporting annotations per 1,000
tokens. The figure shows every represented archive. Most archives contain only
three or four sampled documents because the sample was distributed as evenly
as possible across the 47 archives. Their rates should therefore be treated as
descriptive signals for closer examination, not as reliable archive rankings.

The table also provides an automatically derived archive-prefix group. This is
the part of the archive name before its first underscore, converted to upper
case: for example, `bnf_267` and `bnf_032` both become `BNF`. It is included so
that apparent concentrations such as BnF can be checked against their reviewed
document and token exposure. These prefix groups should be verified before
they are described as institutions or collections in final reporting.

## Repeated-document families

The family identifier comes from Stage 3. Only a final suffix matching
`_duplicated_<number>` is removed. All other filename stems remain unchanged.

Every family is retained in the table and marked as either repeated or a
singleton. No family figure is produced because only a very small number of
repeated families is available. Plotting them would risk presenting a sparse
descriptive check as a general result.

## Input

- `analysis_outputs/03_document_diagnostics/tables/document_diagnostics.csv`:
  reviewed documents, annotation burden, archive, model, reviewer and family.

## Tables produced by this script

- `archive_context_summary.csv`: archive-level and archive-prefix-group
  exposure, annotation rates, model mix, reviewer coverage and flagged-document
  counts.
- `document_family_summary.csv`: every family, including its members, exposure,
  annotation rate, model coverage and whether it is genuinely repeated.

## Figures produced by this script

- `archive_annotation_rates`: all represented archives, ranked by pooled
  annotation rate across two landscape panels.
The archive figure is saved as both PNG and SVG.

## Interpretation

These results describe reviewer feedback rather than confirmed model errors.
An archive or family pattern may reflect textual characteristics, model
allocation, reviewer behaviour or some combination of these. The table retains
model and reviewer coverage so those explanations can be investigated next.
