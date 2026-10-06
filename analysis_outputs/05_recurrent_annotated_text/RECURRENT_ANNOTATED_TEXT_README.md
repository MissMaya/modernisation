# 05 · Recurrently annotated words and phrases

## Purpose

Identify the words and phrases repeatedly marked by reviewers and the associated 
error labels. 

## What counts as a recurrent text span?

A text span is **recurrent** when the same conservatively normalised text occurs
in at least 2 reviewed documents and at least 2
reviewer packets. We assess recurrence separately across both models because models.

Normalisation uses Unicode NFC, collapses whitespace, trims and case-folds. It
does not remove accents or punctuation, stem or lemmatise, split phrases, or
merge spelling variants. A single annotation is identified by document, annotation
ID and normalised text, so multiple labels attached to one annotation do not
inflate the annotation count. Missing or empty annotated text is excluded.


## Label consistency

An error label is the category-sub-rule combination assigned by a reviewer.
The most frequently assigned label is the one attached to the largest number
of distinct annotations of that span. If labels are tied in frequency, they are all
reported. 

Observed coverage is the proportion of the span's annotations carrying a most frequent label. 
The 95% Wilson lower bound is the lower end of a confidence interval for that proportion: 
it reduces when the evidence is sparse and rises when the same label is repeatedly assigned.

If one annotation has several different labels, each is retained and reported separately. 
These measures describe reviewer-label consistency. They do not establish  whether
reviewers annotated correctly or whether a model genuinely failed.

## Outputs

- `annotated_text_summary.csv`: every non-empty annotated word or phrase,
  recurrence status, evidence breadth, label measures and model split.
- `recurrent_annotated_text.csv`: every recurrent span in order of annotation
  frequency.
- `annotated_text_evidence.csv`: every individual annotation and error-label
  assignment, including the recurrence flag. 
- `recurrent_annotated_text_top_10`: the ten recurrent spans with the largest
  numbers of distinct annotations.
- `recurrent_annotated_text_index_*`: every recurrent span in alphabetical 
  order. Displayed as up to 15 rows for readability. 
  Each filename records the first and last span on that page.

## Ranking

The summary and recurrent-span CSV are ordered by distinct annotations, then
affected documents, reviewer packets, archives and the Wilson lower bound for
the most frequently assigned label. The first ten rows appear in the summary 
figure (ten is just a number chosen for readability on a single page.)
