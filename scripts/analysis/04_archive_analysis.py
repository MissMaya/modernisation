"""
Examine the archive distribution of the documents flagged in the scatterplot.

The previous step highlighted documents with annotation counts and/or annotation rates
above the model-specific upper outlier threshold. This script analyses how the
reviewed documents and those outlier classifications are distributed across
archives.

The figure retains one marker per document. It shows whether the documents
identified in Stage 3 are concentrated in particular archives without treating
the small archive samples as stable archive estimates.

The archive table uses one row per archive and model. It retains the counts and
token totals from which each model-specific rate is calculated.

Outputs:

    analysis_outputs/04_archive_analysis/
        ARCHIVE_ANALYSIS_README.md
        tables/archive_context_summary.csv
        figures/en/high_burden_documents_by_archive.png and .svg
"""

# ---------------------------------------------------------------------------
# Import the required modules
# ---------------------------------------------------------------------------

import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    save_figure,
    save_table,
)


# ---------------------------------------------------------------------------
# Set up the input and output paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(
    os.environ.get("MODERNISATION_PROJECT_DIR", "/workspaces/modernisation")
)
DOCUMENT_DIAGNOSTICS_PATH = (
    PROJECT_DIR
    / "analysis_outputs"
    / "03_document_diagnostics"
    / "tables"
    / "document_diagnostics.csv"
)

ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "04_archive_analysis"
SECTION_README_PATH = (
    SECTION_OUTPUT_DIR / "ARCHIVE_ANALYSIS_README.md"
)


# ---------------------------------------------------------------------------
# Specify wording for charts
# ---------------------------------------------------------------------------

CHART_TEXT = {
    "en": {
        "archives": {
            "title": "Where did high-burden documents occur?",
            "description": (
                "The chart shows whether the documents identified in Stage 3 "
                "were concentrated in particular archives."
            ),
            "measure": (
                "Each marker is one reviewed document. Highlighted markers "
                "exceed the model-specific upper outlier threshold for "
                "annotation count, annotations per 1,000 tokens, or both."
            ),
            "x_label": "Reviewed documents within archive",
        },
    }
}


README_TEXT = """# 04 · Archive analysis

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
"""


# ---------------------------------------------------------------------------
# Check and load the Stage 3 diagnostic table
# ---------------------------------------------------------------------------

check_required_files(
    [DOCUMENT_DIAGNOSTICS_PATH],
    preceding_command = "python scripts/analysis/03_document_diagnostics.py",
)

documents_df = pd.read_csv(
    DOCUMENT_DIAGNOSTICS_PATH,
    dtype = {"filename_stem": "string"},
)

required_columns = {
    "filename_stem",
    "archive",
    "model",
    "reviewer_name",
    "n_modernised_tokens",
    "n_included_annotations",
    "n_included_assignments",
    "included_annotations_per_1000_tokens",
    "unusually_high_annotation_count",
    "unusually_high_annotation_rate",
    "above_either_upper_outlier_threshold",
    "upper_outlier_reason",
}

missing_columns = required_columns - set(documents_df.columns)
if missing_columns:
    raise ValueError(
        "document_diagnostics.csv is missing required columns: "
        f"{sorted(missing_columns)}."
    )

if documents_df["filename_stem"].duplicated().any():
    raise ValueError(
        "document_diagnostics.csv must contain one row per reviewed document."
    )

for column in (
    "unusually_high_annotation_count",
    "unusually_high_annotation_rate",
    "above_either_upper_outlier_threshold",
):
    documents_df[column] = coerce_boolean(documents_df[column])

for column in (
    "n_modernised_tokens",
    "n_included_annotations",
    "n_included_assignments",
    "included_annotations_per_1000_tokens",
):
    documents_df[column] = pd.to_numeric(
        documents_df[column],
        errors = "coerce",
    )

invalid_denominator = (
    documents_df["n_modernised_tokens"].isna()
    | documents_df["n_modernised_tokens"].le(0)
)
if invalid_denominator.any():
    raise ValueError(
        "Stage 3 contains a document without a positive modernised-token "
        "count. Rebuild Stage 3 before running this analysis."
    )

valid_outlier_reasons = {"count and rate", "count", "rate", "neither"}
observed_outlier_reasons = set(
    documents_df["upper_outlier_reason"].dropna().astype(str)
)
unexpected_outlier_reasons = observed_outlier_reasons - valid_outlier_reasons
if unexpected_outlier_reasons:
    raise ValueError(
        "Stage 3 contains unexpected upper-outlier reasons: "
        f"{sorted(unexpected_outlier_reasons)}."
    )

reason_indicates_outlier = documents_df["upper_outlier_reason"].fillna(
    "neither"
).ne("neither")
recorded_outlier_flag = documents_df[
    "above_either_upper_outlier_threshold"
].fillna(False).astype(bool)
outlier_flag_mismatch = reason_indicates_outlier.ne(recorded_outlier_flag)
if outlier_flag_mismatch.any():
    mismatch_documents = documents_df.loc[
        outlier_flag_mismatch,
        "filename_stem",
    ].astype(str).tolist()
    raise ValueError(
        "The Stage 3 upper-outlier flag does not match its recorded reason "
        f"for: {mismatch_documents}."
    )

missing_archive = (
    documents_df["archive"].isna()
    | documents_df["archive"].astype(str).str.strip().eq("")
)
if missing_archive.any():
    raise ValueError("Every reviewed document must have an archive.")

models = sorted(str(model) for model in documents_df["model"].dropna().unique())
if len(models) != 2:
    raise ValueError("The archive comparison expects exactly two models.")

# ---------------------------------------------------------------------------
# Derive a simple archive-prefix grouping
# ---------------------------------------------------------------------------

def derive_archive_prefix_group(archive):
    """Return the upper-case text before the first archive-name underscore."""

    archive_text = str(archive).strip()
    if not archive_text:
        return "MISSING"
    return archive_text.split("_", maxsplit = 1)[0].upper()


documents_df["archive_prefix_group"] = documents_df["archive"].map(
    derive_archive_prefix_group
)


# ---------------------------------------------------------------------------
# Replace this section's earlier outputs
# ---------------------------------------------------------------------------

if (
    SECTION_OUTPUT_DIR.name != "04_archive_analysis"
    or SECTION_OUTPUT_DIR.parent != ANALYSIS_OUTPUT_DIR
):
    raise RuntimeError(
        f"Unsafe to replace analysis output directory: {SECTION_OUTPUT_DIR}"
    )

if SECTION_OUTPUT_DIR.exists():
    shutil.rmtree(SECTION_OUTPUT_DIR)

tables_directory, figures_directory = create_output_folders(SECTION_OUTPUT_DIR)
SECTION_README_PATH.write_text(README_TEXT, encoding = "utf-8")


# ---------------------------------------------------------------------------
# Summarise one group for one model
# ---------------------------------------------------------------------------

def summarise_model_group(group_df):
    """Return exposure, feedback and Stage 3 outlier counts for one model."""

    total_tokens = group_df["n_modernised_tokens"].sum()
    total_annotations = group_df["n_included_annotations"].sum()
    outlier_reasons = group_df["upper_outlier_reason"].astype(str)

    return pd.Series(
        {
            "reviewed_documents": len(group_df),
            "modernised_tokens": int(total_tokens),
            "included_annotations": int(total_annotations),
            "included_assignments": int(
                group_df["n_included_assignments"].sum()
            ),
            "annotations_per_1000_tokens": (
                total_annotations / total_tokens * 1000
            ),
            "distinct_reviewers": group_df["reviewer_name"].nunique(
                dropna = True
            ),
            "upper_outlier_documents": int(
                group_df["above_either_upper_outlier_threshold"].sum()
            ),
            "count_and_rate_outlier_documents": int(
                outlier_reasons.eq("count and rate").sum()
            ),
            "count_only_outlier_documents": int(
                outlier_reasons.eq("count").sum()
            ),
            "rate_only_outlier_documents": int(
                outlier_reasons.eq("rate").sum()
            ),
        }
    )


# ---------------------------------------------------------------------------
# Build the archive-and-model summary table
# ---------------------------------------------------------------------------

archive_rows = []

for summary_level, grouping_column in (
    ("archive_prefix_group", "archive_prefix_group"),
    ("archive", "archive"),
):
    grouped = documents_df.groupby(
        [grouping_column, "model"],
        dropna = False,
        sort = True,
    )
    for (group_name, model), group_df in grouped:
        summary = summarise_model_group(group_df).to_dict()
        summary["summary_level"] = summary_level
        summary["group_name"] = group_name
        summary["model"] = model
        archive_rows.append(summary)

archive_context_summary_df = pd.DataFrame(archive_rows)
archive_context_summary_df = archive_context_summary_df[
    [
        "summary_level",
        "group_name",
        "model",
        "reviewed_documents",
        "modernised_tokens",
        "included_annotations",
        "included_assignments",
        "annotations_per_1000_tokens",
        "distinct_reviewers",
        "upper_outlier_documents",
        "count_and_rate_outlier_documents",
        "count_only_outlier_documents",
        "rate_only_outlier_documents",
    ]
].sort_values(
    ["summary_level", "group_name", "model"],
    ascending = [True, True, True],
)

save_table(
    archive_context_summary_df,
    tables_directory / "archive_context_summary.csv",
)


# ---------------------------------------------------------------------------
# Generate the document-status archive figure
# ---------------------------------------------------------------------------

apply_plot_style()

archive_names = sorted(
    documents_df["archive"].dropna().astype(str).unique(),
    key = str.casefold,
)
outlier_styles = {
    "neither": {
        "label": "Below both thresholds",
        "colour": "#C9C1B7",
        "marker": "o",
        "size": 24,
    },
    "count": {
        "label": "Count only",
        "colour": "#8A6073",
        "marker": "s",
        "size": 42,
    },
    "rate": {
        "label": "Rate only",
        "colour": "#64758A",
        "marker": "^",
        "size": 48,
    },
    "count and rate": {
        "label": "Count and rate",
        "colour": "#4F5947",
        "marker": "D",
        "size": 46,
    },
}
outlier_order = {
    "count and rate": 0,
    "count": 1,
    "rate": 2,
    "neither": 3,
}

for language in OUTPUT_LANGUAGES:
    if language not in CHART_TEXT:
        raise ValueError(
            "No archive chart wording has been supplied for: "
            f"{language}."
        )

    language_directory = figures_directory / language
    text = CHART_TEXT[language]

    fig, axes = plt.subplots(
        1,
        2,
        figsize = (16, 11),
        sharex = True,
        sharey = True,
    )
    fig.subplots_adjust(
        top = 0.72,
        bottom = 0.16,
        left = 0.15,
        right = 0.97,
        wspace = 0.10,
    )

    y_positions = {
        archive: position
        for position, archive in enumerate(archive_names)
    }

    for ax, model in zip(axes, models):
        for archive in archive_names:
            archive_df = documents_df.loc[
                documents_df["archive"].astype(str).eq(archive)
                & documents_df["model"].astype(str).eq(model)
            ].copy()
            archive_df["display_order"] = archive_df[
                "upper_outlier_reason"
            ].map(outlier_order)
            archive_df = archive_df.sort_values(
                [
                    "display_order",
                    "n_included_annotations",
                    "included_annotations_per_1000_tokens",
                    "filename_stem",
                ],
                ascending = [True, False, False, True],
            )

            for document_position, (_, row) in enumerate(
                archive_df.iterrows(),
                start = 1,
            ):
                reason = str(row["upper_outlier_reason"])
                style = outlier_styles[reason]
                ax.scatter(
                    document_position,
                    y_positions[archive],
                    color = style["colour"],
                    marker = style["marker"],
                    s = style["size"],
                    alpha = 0.95,
                    edgecolor = COLOURS["panel"],
                    linewidth = 0.5,
                    zorder = 3,
                )

        ax.set_yticks(range(len(archive_names)))
        ax.set_yticklabels(archive_names, fontsize = 6.8)
        ax.set_ylim(len(archive_names) - 0.5, -0.5)
        ax.set_xlim(0.5, 5.5)
        ax.set_xticks(range(1, 6))
        ax.set_title(
            model,
            fontsize = 10,
            fontweight = "bold",
            loc = "center",
            pad = 10,
        )
        ax.set_xlabel(text["archives"]["x_label"])
        ax.grid(axis = "both", alpha = 0.35)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis = "y", length = 0)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker = style["marker"],
            linestyle = "none",
            markerfacecolor = style["colour"],
            markeredgecolor = COLOURS["panel"],
            markersize = 7,
            label = style["label"],
        )
        for style in outlier_styles.values()
    ]
    fig.legend(
        handles = legend_handles,
        frameon = False,
        loc = "lower center",
        bbox_to_anchor = (0.5, 0.055),
        ncol = len(outlier_styles),
        fontsize = 8.5,
    )

    add_chart_header(
        fig,
        **{
            key: text["archives"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "Archives are alphabetical. Horizontal position only separates "
        "documents within an archive; highlighted documents are placed first.",
    )
    save_figure(
        fig,
        language_directory,
        "high_burden_documents_by_archive",
    )


# ---------------------------------------------------------------------------
# Print a brief completion report
# ---------------------------------------------------------------------------

print(f"\nReviewed documents shown: {len(documents_df)}")
print(f"Distinct archives: {documents_df['archive'].nunique()}")
print(
    "Maximum reviewed documents per archive: "
    f"{documents_df.groupby('archive').size().max()}"
)
print(
    "Archive-and-model summary rows: "
    f"{len(archive_context_summary_df)}"
)

print(f"\nTables saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
