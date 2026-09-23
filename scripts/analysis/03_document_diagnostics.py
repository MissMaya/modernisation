"""Identify the individual documents that most need closer examination.

Stages 1 and 2 show that annotation rates differ between the models and that
some error types contribute much more than others. This section locates the
documents behind those overall patterns.

For every reviewed document, the script records its length, annotation count,
annotation rate, archive, reviewer, dominant error category, dominant sub-rule
and repeated-document family. It then creates a shorter review list containing
documents in the highest 10% for either:

* the number of distinct included annotations; or
* distinct included annotations per 1,000 modernised tokens.

The two criteria are deliberately kept separate. A long document can contain
many annotations without having an exceptional rate, while a short document
can have an exceptional rate based on relatively few annotations. Document
length is therefore retained in the table and shown directly on the scatter
plot so that high rates from small token denominators can be examined without
introducing an arbitrary short-document threshold.

A repeated-document family is derived conservatively by removing only a final
suffix matching ``_duplicated_<number>``. All other filename stems remain
unchanged.

The script produces two tables and two figures:

    analysis_outputs/03_document_diagnostics/
        DOCUMENT_DIAGNOSTICS_README.md
        tables/document_diagnostics.csv
        tables/documents_for_review.csv
        figures/en/document_length_and_annotation_burden.png and .svg
        figures/en/flagged_document_error_profiles.png and .svg
"""

import os
from pathlib import Path
import re
import shutil
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd

from analysis_utils import (
    COLOURS,
    ERROR_CATEGORY_COLOURS,
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
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "03_document_diagnostics"
SECTION_README_PATH = SECTION_OUTPUT_DIR / "DOCUMENT_DIAGNOSTICS_README.md"

FAMILY_SUFFIX_PATTERN = re.compile(r"_duplicated_\d+$", flags=re.IGNORECASE)
PRIORITY_PERCENTILE = 0.90


# ---------------------------------------------------------------------------
# Define the visible chart wording
# ---------------------------------------------------------------------------

CHART_TEXT = {
    "en": {
        "burden": {
            "title": "Which documents received the heaviest reviewer feedback?",
            "description": (
                "This comparison separates documents with many annotations "
                "from high rates that may partly reflect short texts."
            ),
            "measure": (
                "Each point is one reviewed document; numbered points are in "
                "the highest 10% by annotation count or annotation rate and "
                "are identified in the adjacent key."
            ),
            "x_label": "Modernised tokens",
            "y_label": "Distinct included annotations",
        },
        "profiles": {
            "title": (
                "What types of error were recorded in documents flagged "
                "for closer review?"
            ),
            "description": (
                "This comparison shows the category composition of reviewer "
                "feedback in documents flagged for closer examination."
            ),
            "measure": (
                "Distinct annotations within each category; one annotation "
                "can contribute to more than one category."
            ),
            "x_label": "Category-specific annotations",
        },
    }
}


README_TEXT = """# 03 · Document diagnostics

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
"""


# ---------------------------------------------------------------------------
# Check and load the two analysis tables
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
        "effective_error_category": "string",
        "effective_subrule": "string",
    },
)

required_document_columns = {
    "filename_stem",
    "archive",
    "model",
    "reviewer_packet",
    "reviewer_name",
    "annotation_json_found",
    "n_modernised_tokens",
    "n_included_annotations",
    "n_included_assignments",
    "included_annotations_per_1000_tokens",
}
required_annotation_columns = {
    "filename_stem",
    "annotation_id",
    "include_in_analysis",
    "effective_error_category",
    "effective_subrule",
}

missing_document_columns = required_document_columns - set(documents_df.columns)
missing_annotation_columns = required_annotation_columns - set(annotations_df.columns)

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

for column in (
    "n_modernised_tokens",
    "n_included_annotations",
    "n_included_assignments",
    "included_annotations_per_1000_tokens",
):
    documents_df[column] = pd.to_numeric(documents_df[column], errors="coerce")


# ---------------------------------------------------------------------------
# Select reviewed documents with a usable token denominator
# ---------------------------------------------------------------------------

analysis_df = documents_df.loc[
    documents_df["annotation_json_found"].fillna(False)
    & documents_df["n_modernised_tokens"].notna()
    & documents_df["n_modernised_tokens"].gt(0)
].copy()

if analysis_df.empty:
    raise ValueError(
        "No documents have both available annotation data and a positive "
        "modernised token count."
    )

included_annotations_df = annotations_df.loc[
    annotations_df["include_in_analysis"].fillna(False)
].copy()

# Keep only assignments belonging to documents included in this analysis.
included_annotations_df = included_annotations_df.merge(
    analysis_df[["filename_stem"]],
    on="filename_stem",
    how="inner",
    validate="many_to_one",
)

if included_annotations_df["effective_error_category"].isna().any():
    raise ValueError(
        "At least one included annotation assignment has no effective error "
        "category. Check the Stage 6 resolution output."
    )


# ---------------------------------------------------------------------------
# Derive conservative repeated-document family information
# ---------------------------------------------------------------------------

def derive_document_family(filename_stem):
    """Remove a final _duplicated_<number> suffix from a filename stem."""

    return FAMILY_SUFFIX_PATTERN.sub("", str(filename_stem))


analysis_df["document_family"] = analysis_df["filename_stem"].map(
    derive_document_family
)
family_sizes = analysis_df.groupby("document_family")["filename_stem"].transform(
    "size"
)
analysis_df["reviewed_documents_in_family"] = family_sizes
analysis_df["repeated_document_family"] = family_sizes.gt(1)


# ---------------------------------------------------------------------------
# Summarise the dominant category and sub-rule within each document
# ---------------------------------------------------------------------------

def join_joint_modes(values):
    """Return all equally frequent non-missing labels in alphabetical order."""

    counts = pd.Series(values).dropna().astype(str).value_counts()
    if counts.empty:
        return pd.NA
    modes = sorted(counts.index[counts.eq(counts.max())])
    return " | ".join(modes)


# Count an annotation once within a category, even when several field rows for
# that category were created during flattening.
category_annotations_df = included_annotations_df.drop_duplicates(
    subset=["filename_stem", "annotation_id", "effective_error_category"]
)
dominant_category_df = (
    category_annotations_df.groupby("filename_stem")["effective_error_category"]
    .agg(join_joint_modes)
    .rename("dominant_error_category")
    .reset_index()
)

# Preserve category-only assignments under an explicit description so they
# remain visible in the document diagnostic table.
included_annotations_df["subrule_for_summary"] = (
    included_annotations_df["effective_subrule"].fillna("No sub-rule assigned")
)
included_annotations_df["category_and_subrule"] = (
    included_annotations_df["effective_error_category"].astype(str)
    + " — "
    + included_annotations_df["subrule_for_summary"].astype(str)
)
subrule_annotations_df = included_annotations_df.drop_duplicates(
    subset=[
        "filename_stem",
        "annotation_id",
        "effective_error_category",
        "subrule_for_summary",
    ]
)
dominant_subrule_df = (
    subrule_annotations_df.groupby("filename_stem")["category_and_subrule"]
    .agg(join_joint_modes)
    .rename("dominant_subrule")
    .reset_index()
)

analysis_df = analysis_df.merge(
    dominant_category_df,
    on="filename_stem",
    how="left",
    validate="one_to_one",
)
analysis_df = analysis_df.merge(
    dominant_subrule_df,
    on="filename_stem",
    how="left",
    validate="one_to_one",
)

zero_annotation_document = analysis_df["n_included_annotations"].eq(0)
analysis_df.loc[zero_annotation_document, "dominant_error_category"] = (
    "No included annotations"
)
analysis_df.loc[zero_annotation_document, "dominant_subrule"] = (
    "No included annotations"
)


# ---------------------------------------------------------------------------
# Select documents requiring closer examination
# ---------------------------------------------------------------------------

# ``interpolation='higher'`` ensures that the threshold is an observed value.
# Ties at either boundary are retained, so the selected share can exceed 10%.
count_threshold = analysis_df["n_included_annotations"].quantile(
    PRIORITY_PERCENTILE,
    interpolation="higher",
)
rate_threshold = analysis_df["included_annotations_per_1000_tokens"].quantile(
    PRIORITY_PERCENTILE,
    interpolation="higher",
)
analysis_df["high_annotation_count"] = analysis_df[
    "n_included_annotations"
].ge(count_threshold)
analysis_df["high_annotation_rate"] = analysis_df[
    "included_annotations_per_1000_tokens"
].ge(rate_threshold)
analysis_df["selected_for_review"] = (
    analysis_df["high_annotation_count"] | analysis_df["high_annotation_rate"]
)


def describe_selection_reason(row):
    """Describe which transparent rule placed a document in the review list."""

    if row["high_annotation_count"] and row["high_annotation_rate"]:
        return "highest 10% by count and rate"
    if row["high_annotation_count"]:
        return "highest 10% by count"
    if row["high_annotation_rate"]:
        return "highest 10% by rate"
    return "not selected"


analysis_df["selection_reason"] = analysis_df.apply(
    describe_selection_reason,
    axis=1,
)


# ---------------------------------------------------------------------------
# Replace this section's earlier outputs
# ---------------------------------------------------------------------------

if (
    SECTION_OUTPUT_DIR.name != "03_document_diagnostics"
    or SECTION_OUTPUT_DIR.parent != ANALYSIS_OUTPUT_DIR
):
    raise RuntimeError(
        f"Unsafe to replace analysis output directory: {SECTION_OUTPUT_DIR}"
    )

if SECTION_OUTPUT_DIR.exists():
    shutil.rmtree(SECTION_OUTPUT_DIR)

tables_directory, figures_directory = create_output_folders(SECTION_OUTPUT_DIR)
SECTION_README_PATH.write_text(README_TEXT, encoding="utf-8")


# ---------------------------------------------------------------------------
# Save the complete diagnostic table and shorter review list
# ---------------------------------------------------------------------------

diagnostic_columns = [
    "filename_stem",
    "model",
    "archive",
    "reviewer_packet",
    "reviewer_name",
    "n_modernised_tokens",
    "n_included_annotations",
    "n_included_assignments",
    "included_annotations_per_1000_tokens",
    "dominant_error_category",
    "dominant_subrule",
    "document_family",
    "reviewed_documents_in_family",
    "repeated_document_family",
    "high_annotation_count",
    "high_annotation_rate",
    "selected_for_review",
    "selection_reason",
]

document_diagnostics_df = (
    analysis_df[diagnostic_columns]
    .sort_values(
        ["included_annotations_per_1000_tokens", "n_included_annotations"],
        ascending=[False, False],
    )
    .reset_index(drop=True)
)
document_diagnostics_df.insert(
    0,
    "annotation_rate_rank",
    range(1, len(document_diagnostics_df) + 1),
)

documents_for_review_df = (
    document_diagnostics_df.loc[
        document_diagnostics_df["selected_for_review"]
    ]
    .copy()
    .reset_index(drop=True)
)

# Give every flagged document a short number for this run. The same number
# appears on the scatter plot, in its adjacent document key, in the bar-chart
# labels and in documents_for_review.csv. This is more readable than placing
# long filenames directly beside tightly clustered scatter points.
documents_for_review_df.insert(
    0,
    "review_flag_number",
    range(1, len(documents_for_review_df) + 1),
)

save_table(
    document_diagnostics_df,
    tables_directory / "document_diagnostics.csv",
)
save_table(
    documents_for_review_df,
    tables_directory / "documents_for_review.csv",
)


# ---------------------------------------------------------------------------
# Prepare category counts for the flagged-document profile figure
# ---------------------------------------------------------------------------

flagged_stems = set(documents_for_review_df["filename_stem"])
flagged_categories_df = category_annotations_df.loc[
    category_annotations_df["filename_stem"].isin(flagged_stems)
]
category_counts_df = (
    flagged_categories_df.groupby(
        ["filename_stem", "effective_error_category"]
    )
    .size()
    .unstack(fill_value=0)
)

category_order = (
    category_counts_df.sum(axis=0).sort_values(ascending=False).index.tolist()
)
flagged_plot_order = documents_for_review_df.sort_values(
    ["included_annotations_per_1000_tokens", "n_included_annotations"],
    ascending=[True, True],
)["filename_stem"].tolist()
category_counts_df = category_counts_df.reindex(
    index=flagged_plot_order,
    columns=category_order,
    fill_value=0,
)


# ---------------------------------------------------------------------------
# Generate the two document-diagnostic figures
# ---------------------------------------------------------------------------

apply_plot_style()
model_colours = model_colour_map(analysis_df["model"])

for language in OUTPUT_LANGUAGES:
    if language not in CHART_TEXT:
        raise ValueError(
            f"No document-diagnostic chart wording has been supplied for: {language}."
        )

    language_directory = figures_directory / language
    text = CHART_TEXT[language]

    # Figure 1: show annotation burden against document length. Model colour
    # supports comparison. Flagged documents use numbered markers connected to
    # a complete key beside the plot, avoiding overlapping filename labels.
    scatter_figure_height = max(
        8.5,
        4.8 + 0.16 * len(documents_for_review_df),
    )
    fig, ax = plt.subplots(figsize=(15, scatter_figure_height))
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.09, right=0.69)

    for model, model_df in analysis_df.groupby("model", sort=True):
        ax.scatter(
            model_df["n_modernised_tokens"],
            model_df["n_included_annotations"],
            label=str(model),
            color=model_colours[str(model)],
            s=42,
            alpha=0.70,
            edgecolor=COLOURS["panel"],
            linewidth=0.5,
            zorder=2,
        )

    flagged_plot_df = documents_for_review_df.copy()
    ax.scatter(
        flagged_plot_df["n_modernised_tokens"],
        flagged_plot_df["n_included_annotations"],
        facecolors=[
            model_colours[str(model)] for model in flagged_plot_df["model"]
        ],
        edgecolors=COLOURS["text"],
        s=150,
        linewidth=1.0,
        zorder=3,
    )

    # Numbers remain legible even where several flagged documents are close
    # together. Full filenames are retained in the key and output table.
    for _, row in flagged_plot_df.iterrows():
        ax.text(
            row["n_modernised_tokens"],
            row["n_included_annotations"],
            str(row["review_flag_number"]),
            fontsize=6.2,
            fontweight="bold",
            color=COLOURS["panel"],
            ha="center",
            va="center",
            zorder=4,
        )

    ax.set_xlabel(text["burden"]["x_label"])
    ax.set_ylabel(text["burden"]["y_label"])
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    # Place the complete numbered key in its own axes so no filename can fall
    # outside the plot or obscure another document. Long names are wrapped
    # within the fixed key width rather than allowed to cross the figure edge.
    key_ax = fig.add_axes([0.715, 0.14, 0.27, 0.58])
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)
    key_ax.set_axis_off()
    key_ax.text(
        0.045,
        1.02,
        "Documents flagged for closer review",
        fontsize=9,
        fontweight="bold",
        color=COLOURS["text"],
        va="bottom",
        transform=key_ax.transAxes,
    )
    key_line_height = 0.98 / max(len(flagged_plot_df), 1)
    for key_position, (_, row) in enumerate(flagged_plot_df.iterrows()):
        key_y_position = 0.98 - key_position * key_line_height

        # Repeat the model colour beside each filename so the key can be read
        # directly without tracing the numbered point back to the scatter.
        key_ax.text(
            0.015,
            key_y_position,
            "●",
            fontsize=7,
            color=model_colours[str(row["model"])],
            ha="center",
            va="top",
            transform=key_ax.transAxes,
        )
        key_ax.text(
            0.045,
            key_y_position,
            fill(
                f"{row['review_flag_number']}. {row['filename_stem']}",
                width=43,
            ),
            fontsize=6.4,
            color=COLOURS["muted_text"],
            va="top",
            linespacing=0.95,
            transform=key_ax.transAxes,
        )
    add_chart_header(
        fig,
        **{
            key: text["burden"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "Document length is shown because a moderate annotation count can create a high per-1,000-token rate when the token denominator is small.",
    )
    save_figure(fig, language_directory, "document_length_and_annotation_burden")

    # Figure 2: show which error categories make up the feedback received by
    # each flagged document. The document, reviewer and archive are kept on one
    # line so possible clustering is easier to scan during human inspection.
    figure_height = max(8.0, 4.8 + 0.32 * len(category_counts_df))
    fig, ax = plt.subplots(figsize=(16, figure_height))

    # The wide figure provides enough room for one-line metadata labels without
    # allowing the label column to consume nearly half of the canvas.
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.36, right=0.95)

    left_values = pd.Series(0, index=category_counts_df.index, dtype=float)
    for category_position, category in enumerate(category_order):
        values = category_counts_df[category]
        ax.barh(
            category_counts_df.index,
            values,
            left=left_values,
            label=category,
            color=ERROR_CATEGORY_COLOURS[
                category_position % len(ERROR_CATEGORY_COLOURS)
            ],
            edgecolor="none",
        )
        left_values = left_values + values

    flagged_metadata = documents_for_review_df.set_index("filename_stem")
    y_labels = []
    for filename_stem in category_counts_df.index:
        row = flagged_metadata.loc[filename_stem]
        y_labels.append(
            f"{row['review_flag_number']}. {filename_stem} · "
            f"{row['reviewer_name']} · {row['archive']}"
        )

    ax.set_yticks(range(len(category_counts_df.index)))
    ax.set_yticklabels(y_labels, fontsize=7.5)

    # Explain the three pieces of metadata concatenated in each y-axis label.
    # This heading sits above the label column rather than inside the data area.
    fig.text(
        0.355,
        0.735,
        "Document · reviewer · archive",
        fontsize=8.5,
        fontweight="bold",
        color=COLOURS["muted_text"],
        ha="right",
        va="bottom",
    )
    ax.set_xlabel(text["profiles"]["x_label"])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        bbox_transform=fig.transFigure,
        ncol=3,
        fontsize=8,
    )
    add_chart_header(
        fig,
        **{
            key: text["profiles"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "Documents are flagged using count and rate thresholds, not by error category. Each label shows the document, reviewer and archive.",
    )
    save_figure(fig, language_directory, "flagged_document_error_profiles")


# ---------------------------------------------------------------------------
# Print a brief completion report
# ---------------------------------------------------------------------------

print(f"\nReviewed documents included: {len(analysis_df)}")
print(
    "Documents selected for closer review: "
    f"{len(documents_for_review_df)}"
)
print(
    "Highest-10% annotation-count threshold: "
    f"{count_threshold:.0f} annotations"
)
print(
    "Highest-10% annotation-rate threshold: "
    f"{rate_threshold:.2f} annotations per 1,000 tokens"
)
print(f"\nTables saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
