"""Identify recurring reviewer feedback that could inform prompt revision.

Earlier stages establish which models, error categories, documents and archives
received the most reviewer feedback. This stage moves from description towards
prompt improvement by asking which exact annotated forms recur with the same
category and sub-rule.

The unit summarised here is a distinct annotation pattern:

    model + error category + sub-rule + normalised annotated text

Text is normalised conservatively: Unicode is standardised, leading and
trailing space is removed, repeated whitespace is collapsed and letters are
converted to lower case. Accents and punctuation are preserved because they
may be the feature that caused the annotation.

A pattern is marked as a prompt candidate for a model when it occurs in at
least two documents, is assigned by at least two reviewers and appears in at
least two archives. These are breadth checks, not statistical significance
tests. They prevent a single document, reviewer or archive from automatically
becoming a prompt recommendation. The thresholds are named constants below so
they can be changed deliberately if the review team adopts another rule.

Every observed pattern is retained in ``failure_pattern_summary.csv``.
Candidate patterns and their document-level examples are written separately so
they can be inspected before any prompt is changed. One figure summarises how
many distinct recurring forms fall under each category and sub-rule by model.

The script creates:

    analysis_outputs/05_actionable_failure_patterns/
        ACTIONABLE_FAILURE_PATTERNS_README.md
        tables/failure_pattern_summary.csv
        tables/prompt_candidates_for_review.csv
        figures/en/prompt_candidate_rule_concentration.png and .svg
            (only when at least one candidate pattern is found)
"""

import os
from pathlib import Path
import re
import shutil
from textwrap import fill
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd

from analysis_utils import (
    COLOURS,
    OUTPUT_LANGUAGES,
    add_chart_header,
    add_figure_note,
    apply_plot_style,
    check_required_files,
    coerce_boolean,
    create_output_folders,
    model_colour_map,
    save_figure,
    save_table,
)


# ---------------------------------------------------------------------------
# Set up the input and output paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(
    os.environ.get("MODERNISATION_PROJECT_DIR", "/workspaces/modernisation")
)
DOCUMENT_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "document_analysis.csv"
ANNOTATION_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "annotation_analysis.csv"

ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "05_actionable_failure_patterns"
SECTION_README_PATH = (
    SECTION_OUTPUT_DIR / "ACTIONABLE_FAILURE_PATTERNS_README.md"
)

# A candidate must recur beyond one document, reviewer and archive. These
# thresholds are intentionally about breadth of evidence rather than raw count.
MIN_DOCUMENTS_FOR_CANDIDATE = 2
MIN_REVIEWERS_FOR_CANDIDATE = 2
MIN_ARCHIVES_FOR_CANDIDATE = 2

NO_SUBRULE_LABEL = "No sub-rule assigned"
MISSING_TEXT_LABEL = "[missing annotated text]"


# ---------------------------------------------------------------------------
# Define the visible chart wording
# ---------------------------------------------------------------------------

CHART_TEXT = {
    "en": {
        "candidates": {
            "title": "Where are recurring prompt candidates concentrated?",
            "description": (
                "These patterns recur beyond one document, reviewer and archive "
                "for at least one model."
            ),
            "measure": (
                "Distinct recurring annotated forms within each category and "
                "sub-rule, compared by model."
            ),
            "x_label": "Recurring annotated forms",
        }
    }
}


README_TEXT = f"""# 05 · Actionable failure patterns

## What this section is trying to show

This section identifies exact forms that repeatedly received the same error
category and sub-rule. Its purpose is to produce evidence that can be checked
and converted into clearer prompt instructions or examples.

This is still reviewer feedback, not a list of confirmed model errors. Human
inspection is required before any candidate becomes a prompt change.

## What counts as one pattern

A pattern combines:

- model;
- error category;
- sub-rule; and
- normalised annotated text.

The annotated text is converted to lower case, surrounding space is removed and
repeated whitespace is collapsed. Accents and punctuation are retained.

## How prompt candidates are identified

For a particular model, a pattern is marked as a candidate when it appears in
at least:

- {MIN_DOCUMENTS_FOR_CANDIDATE} documents;
- {MIN_REVIEWERS_FOR_CANDIDATE} reviewers; and
- {MIN_ARCHIVES_FOR_CANDIDATE} archives.

These are transparent breadth checks, not significance tests. They stop one
document, reviewer or archive from determining a proposed prompt change.

## Inputs

- `outputs/document_analysis.csv`: supplies model, archive, reviewer and token
  exposure for each reviewed document.
- `outputs/annotation_analysis.csv`: supplies the retained annotation text,
  categories and sub-rules.

## Tables produced by this script

- `failure_pattern_summary.csv`: every observed model-specific pattern, with
  annotation, document, reviewer and archive coverage.
- `prompt_candidates_for_review.csv`: the underlying annotations for candidate
  patterns, retained as examples for human inspection.

## Figure produced by this script

- `prompt_candidate_rule_concentration`: the number of distinct recurring
  annotated forms within each category and sub-rule, compared by model. The
  detailed forms themselves remain in the two CSVs. The figure is omitted if
  no pattern meets the breadth checks.

The figure is saved as both PNG and SVG.

## Interpretation

Patterns supported across documents, reviewers and archives are stronger
candidates for prompt clarification than high counts concentrated in one
context. They still require adjudication: the reviewer may be identifying a
genuine model failure, an ambiguous instruction or an inconsistent review
decision.
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalise_annotated_text(value):
    """Return a conservative comparison form without losing accents or marks."""

    if pd.isna(value):
        return MISSING_TEXT_LABEL

    text = unicodedata.normalize("NFC", str(value))
    text = re.sub(r"\s+", " ", text.strip())
    return text.casefold() if text else MISSING_TEXT_LABEL


def join_examples(values, limit=5):
    """Join up to five distinct values for an inspectable summary cell."""

    examples = sorted({str(value) for value in values if pd.notna(value)})
    return " | ".join(examples[:limit])


# ---------------------------------------------------------------------------
# Check and load the two assembled analysis tables
# ---------------------------------------------------------------------------

check_required_files(
    [DOCUMENT_ANALYSIS_PATH, ANNOTATION_ANALYSIS_PATH],
    preceding_command="python scripts/construct_tables.py",
)

documents_df = pd.read_csv(
    DOCUMENT_ANALYSIS_PATH,
    dtype={"filename_stem": "string"},
)
annotations_df = pd.read_csv(
    ANNOTATION_ANALYSIS_PATH,
    dtype={
        "filename_stem": "string",
        "annotated_text": "string",
        "effective_error_category": "string",
        "effective_subrule": "string",
    },
)

required_document_columns = {
    "filename_stem",
    "model",
    "archive",
    "reviewer_name",
    "annotation_json_found",
    "n_modernised_tokens",
}
required_annotation_columns = {
    "filename_stem",
    "annotation_id",
    "annotated_text",
    "include_in_analysis",
    "effective_error_category",
    "effective_subrule",
}

missing_document_columns = required_document_columns - set(documents_df.columns)
missing_annotation_columns = required_annotation_columns - set(
    annotations_df.columns
)

if missing_document_columns:
    raise ValueError(
        "document_analysis.csv is missing required columns: "
        f"{sorted(missing_document_columns)}."
    )
if missing_annotation_columns:
    raise ValueError(
        "annotation_analysis.csv is missing required columns: "
        f"{sorted(missing_annotation_columns)}."
    )
if documents_df["filename_stem"].duplicated().any():
    raise ValueError(
        "document_analysis.csv must contain one row per sample document."
    )

documents_df["annotation_json_found"] = coerce_boolean(
    documents_df["annotation_json_found"]
)
annotations_df["include_in_analysis"] = coerce_boolean(
    annotations_df["include_in_analysis"]
)
documents_df["n_modernised_tokens"] = pd.to_numeric(
    documents_df["n_modernised_tokens"],
    errors="coerce",
)


# ---------------------------------------------------------------------------
# Select reviewed documents and retained annotation assignments
# ---------------------------------------------------------------------------

reviewed_documents_df = documents_df.loc[
    documents_df["annotation_json_found"].fillna(False)
    & documents_df["n_modernised_tokens"].notna()
    & documents_df["n_modernised_tokens"].gt(0)
].copy()

if reviewed_documents_df.empty:
    raise ValueError(
        "No documents have both annotation data and a positive modernised-token count."
    )

included_annotations_df = annotations_df.loc[
    annotations_df["include_in_analysis"].fillna(False)
].copy()

if included_annotations_df.empty:
    raise ValueError("No included annotations are available for Stage 05.")

# Document metadata are merged here rather than trusting repeated values in the
# annotation table. This keeps one authoritative source for model allocation,
# archive and reviewer.
annotation_metadata_columns = [
    "filename_stem",
    "model",
    "archive",
    "reviewer_name",
]
included_annotations_df = included_annotations_df.drop(
    columns=[
        column
        for column in ("model", "archive", "reviewer_name")
        if column in included_annotations_df.columns
    ]
).merge(
    reviewed_documents_df[annotation_metadata_columns],
    on="filename_stem",
    how="inner",
    validate="many_to_one",
)

included_annotations_df["effective_subrule"] = included_annotations_df[
    "effective_subrule"
].fillna(NO_SUBRULE_LABEL)
included_annotations_df["normalised_annotated_text"] = included_annotations_df[
    "annotated_text"
].map(normalise_annotated_text)

# One annotation can produce repeated flattened rows. Deduplicate only exact
# category/sub-rule assignments so genuinely different assigned fields remain.
pattern_key = [
    "model",
    "effective_error_category",
    "effective_subrule",
    "normalised_annotated_text",
]
annotation_identity = [
    "filename_stem",
    "annotation_id",
    *pattern_key,
]
distinct_pattern_annotations_df = included_annotations_df.drop_duplicates(
    subset=annotation_identity
).copy()


# ---------------------------------------------------------------------------
# Build the complete model-specific failure-pattern summary
# ---------------------------------------------------------------------------

model_token_totals = (
    reviewed_documents_df.groupby("model", dropna=False)["n_modernised_tokens"]
    .sum()
    .to_dict()
)

summary_rows = []
for pattern_values, pattern_df in distinct_pattern_annotations_df.groupby(
    pattern_key,
    dropna=False,
    sort=True,
):
    model, category, subrule, normalised_text = pattern_values
    total_model_tokens = model_token_totals.get(model, 0)
    distinct_annotations = len(pattern_df)
    affected_documents = pattern_df["filename_stem"].nunique()
    affected_reviewers = pattern_df["reviewer_name"].nunique(dropna=True)
    affected_archives = pattern_df["archive"].nunique(dropna=True)

    is_prompt_candidate = (
        affected_documents >= MIN_DOCUMENTS_FOR_CANDIDATE
        and affected_reviewers >= MIN_REVIEWERS_FOR_CANDIDATE
        and affected_archives >= MIN_ARCHIVES_FOR_CANDIDATE
    )

    summary_rows.append(
        {
            "model": model,
            "error_category": category,
            "subrule": subrule,
            "normalised_annotated_text": normalised_text,
            "distinct_annotations": distinct_annotations,
            "affected_documents": affected_documents,
            "affected_reviewers": affected_reviewers,
            "affected_archives": affected_archives,
            "annotations_per_1000_model_tokens": (
                distinct_annotations / total_model_tokens * 1000
                if total_model_tokens > 0
                else pd.NA
            ),
            "prompt_candidate": is_prompt_candidate,
            "example_surface_forms": join_examples(pattern_df["annotated_text"]),
            "example_documents": join_examples(pattern_df["filename_stem"]),
            "reviewers": join_examples(pattern_df["reviewer_name"]),
            "archives": join_examples(pattern_df["archive"]),
        }
    )

failure_pattern_summary_df = pd.DataFrame(summary_rows).sort_values(
    [
        "prompt_candidate",
        "affected_documents",
        "affected_reviewers",
        "affected_archives",
        "distinct_annotations",
        "model",
        "error_category",
        "subrule",
        "normalised_annotated_text",
    ],
    ascending=[False, False, False, False, False, True, True, True, True],
).reset_index(drop=True)

failure_pattern_summary_df.insert(
    0,
    "pattern_rank",
    range(1, len(failure_pattern_summary_df) + 1),
)


# ---------------------------------------------------------------------------
# Retain annotation-level examples for patterns selected as candidates
# ---------------------------------------------------------------------------

candidate_keys_df = failure_pattern_summary_df.loc[
    failure_pattern_summary_df["prompt_candidate"],
    [
        "model",
        "error_category",
        "subrule",
        "normalised_annotated_text",
    ],
].rename(
    columns={
        "error_category": "effective_error_category",
        "subrule": "effective_subrule",
    }
)

if candidate_keys_df.empty:
    prompt_candidates_for_review_df = distinct_pattern_annotations_df.iloc[0:0].copy()
else:
    prompt_candidates_for_review_df = distinct_pattern_annotations_df.merge(
        candidate_keys_df,
        on=pattern_key,
        how="inner",
        validate="many_to_one",
    )

candidate_review_columns = [
    "model",
    "effective_error_category",
    "effective_subrule",
    "normalised_annotated_text",
    "annotated_text",
    "filename_stem",
    "annotation_id",
    "reviewer_name",
    "archive",
]
prompt_candidates_for_review_df = (
    prompt_candidates_for_review_df[candidate_review_columns]
    .rename(
        columns={
            "effective_error_category": "error_category",
            "effective_subrule": "subrule",
        }
    )
    .sort_values(
        [
            "error_category",
            "subrule",
            "normalised_annotated_text",
            "model",
            "filename_stem",
            "annotation_id",
        ]
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------------
# Replace this section's earlier outputs and save the two tables
# ---------------------------------------------------------------------------

if (
    SECTION_OUTPUT_DIR.name != "05_actionable_failure_patterns"
    or SECTION_OUTPUT_DIR.parent != ANALYSIS_OUTPUT_DIR
):
    raise RuntimeError(
        f"Unsafe to replace analysis output directory: {SECTION_OUTPUT_DIR}"
    )

if SECTION_OUTPUT_DIR.exists():
    shutil.rmtree(SECTION_OUTPUT_DIR)

tables_directory, figures_directory = create_output_folders(SECTION_OUTPUT_DIR)
SECTION_README_PATH.write_text(README_TEXT, encoding="utf-8")

save_table(
    failure_pattern_summary_df,
    tables_directory / "failure_pattern_summary.csv",
)
save_table(
    prompt_candidates_for_review_df,
    tables_directory / "prompt_candidates_for_review.csv",
)


# ---------------------------------------------------------------------------
# Create one concise comparison of candidate concentration by rule and model
# ---------------------------------------------------------------------------

candidate_summary_df = failure_pattern_summary_df.loc[
    failure_pattern_summary_df["prompt_candidate"]
].copy()

apply_plot_style()
models = sorted(str(model) for model in reviewed_documents_df["model"].dropna().unique())
model_colours = model_colour_map(models)

if not candidate_summary_df.empty:
    candidate_summary_df["rule_label"] = candidate_summary_df.apply(
        lambda row: (
            f"{row['error_category']} — {row['subrule']}"
        ),
        axis=1,
    )

    # The detailed text patterns remain in the CSVs. Aggregating them to their
    # category and sub-rule keeps the presentation chart readable even when
    # many individual words or phrases meet the breadth checks.
    chart_df = (
        candidate_summary_df
        .groupby(["rule_label", "model"], as_index=False)
        .agg(recurring_forms=("normalised_annotated_text", "nunique"))
        .pivot_table(
            index="rule_label",
            columns="model",
            values="recurring_forms",
            aggfunc="sum",
            fill_value=0,
        )
    )

    for model in models:
        if model not in chart_df.columns:
            chart_df[model] = 0
    chart_df = chart_df[models]
    chart_df["total_recurring_forms"] = chart_df.sum(axis=1)
    chart_df = chart_df.sort_values(
        "total_recurring_forms",
        ascending=False,
    ).drop(columns="total_recurring_forms")

    for language in OUTPUT_LANGUAGES:
        if language not in CHART_TEXT:
            raise ValueError(
                f"No actionable-pattern chart wording has been supplied for: {language}."
            )

        language_directory = figures_directory / language
        text = CHART_TEXT[language]["candidates"]
        # A fixed landscape canvas prevents a large number of individual text
        # patterns from creating an extremely tall image. If more than sixteen
        # rule rows remain, the ranked list is continued in a second panel.
        panel_count = 2 if len(chart_df) > 16 else 1
        panel_size = (len(chart_df) + panel_count - 1) // panel_count
        chart_panels = [
            chart_df.iloc[start : start + panel_size]
            for start in range(0, len(chart_df), panel_size)
        ]

        fig, axes = plt.subplots(
            1,
            panel_count,
            figsize=(16, 10),
            sharex=True,
            squeeze=False,
        )
        fig.subplots_adjust(
            top=0.70,
            bottom=0.18,
            left=0.31 if panel_count == 1 else 0.19,
            right=0.96,
            wspace=0.58 if panel_count == 2 else 0.20,
        )

        legend_handles = None
        legend_labels = None
        for panel_number, (ax, panel_df) in enumerate(
            zip(axes[0], chart_panels),
            start=1,
        ):
            panel_df.plot.barh(
                ax=ax,
                color=[model_colours[model] for model in models],
                width=0.76,
                edgecolor="none",
                legend=False,
            )
            ax.invert_yaxis()
            ax.set_yticklabels(
                [fill(label, width=42) for label in panel_df.index],
                fontsize=8,
            )
            ax.set_ylabel("")
            ax.set_xlabel(text["x_label"])
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            if panel_count == 2:
                ax.set_title(
                    f"Ranked rule areas: panel {panel_number}",
                    fontsize=10,
                    fontweight="bold",
                    loc="left",
                    pad=10,
                )
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.065),
            ncol=2,
        )

        add_chart_header(
            fig,
            **{key: text[key] for key in ("title", "description", "measure")},
        )
        add_figure_note(
            fig,
            "Each count is a distinct annotated form recurring in at least two documents, two reviewers and two archives for that model. Detailed forms remain in the CSVs.",
        )
        save_figure(
            fig,
            language_directory,
            "prompt_candidate_rule_concentration",
        )


# ---------------------------------------------------------------------------
# Print a brief completion report
# ---------------------------------------------------------------------------

print(f"\nDistinct model-specific patterns: {len(failure_pattern_summary_df)}")
print(
    "Patterns meeting the prompt-candidate breadth checks: "
    f"{len(candidate_summary_df)}"
)
print(
    "Annotation examples retained for candidate review: "
    f"{len(prompt_candidates_for_review_df)}"
)
if candidate_summary_df.empty:
    print("Candidate figure not created: no pattern met all three breadth checks.")

print(f"\nTables saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
