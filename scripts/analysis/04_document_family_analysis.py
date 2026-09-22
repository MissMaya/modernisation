"""Rank problematic documents and repeated document families.

This section identifies individual documents with high reviewer-annotation
rates and asks whether repeated versions of the same underlying document were
problematic as a group.

A document family is derived conservatively from filename_stem. Only a final
suffix matching ``_duplicated_<number>`` is removed. For example:

    utblac_gg_mss_g62_duplicated_03
    utblac_gg_mss_g62_duplicated_09

both become:

    utblac_gg_mss_g62

All other filename stems remain unchanged. Family analysis is restricted to
families containing at least two reviewed documents after this rule is applied.

The script produces two tables and two focused figures:

    analysis_outputs/04_document_family_analysis/
        DOCUMENT_FAMILY_ANALYSIS_README.md
        tables/document_ranking.csv
        tables/document_family_summary.csv
        figures/en/highest_document_annotation_rates.png and .svg
        figures/en/highest_family_annotation_rates.png and .svg
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
# Set paths, the family rule and small display limits
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(
    os.environ.get("MODERNISATION_PROJECT_DIR", "/workspaces/modernisation")
)
DOCUMENT_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "document_analysis.csv"

ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "04_document_family_analysis"
SECTION_README_PATH = SECTION_OUTPUT_DIR / "DOCUMENT_FAMILY_ANALYSIS_README.md"

FAMILY_SUFFIX_PATTERN = re.compile(r"_duplicated_\d+$", flags=re.IGNORECASE)
N_DOCUMENTS_IN_FIGURE = 15
N_FAMILIES_IN_FIGURE = 15
MIN_REVIEWED_DOCUMENTS_PER_FAMILY = 2


# ---------------------------------------------------------------------------
# Define visible chart wording
# ---------------------------------------------------------------------------

CHART_TEXT = {
    "en": {
        "documents": {
            "title": "Which documents had the highest annotation rates?",
            "description": (
                "This ranking identifies individual modernised documents "
                "with the highest reviewer-marked error burden relative to length."
            ),
            "measure": (
                "Distinct included annotations per 1,000 modernised tokens; "
                "the 15 highest-rate documents are shown."
            ),
            "x_label": "Annotations per 1,000 tokens",
        },
        "families": {
            "title": "Which repeated document families were most problematic?",
            "description": (
                "This ranking shows whether repeated versions of the same "
                "underlying document attracted consistently high annotation levels."
            ),
            "measure": (
                "Pooled distinct annotations per 1,000 modernised tokens for "
                "families containing at least two reviewed documents."
            ),
            "x_label": "Annotations per 1,000 tokens",
        },
    }
}


README_TEXT = """# 04 · Document and family analysis

## What this section is trying to show

This section identifies the individual documents with the highest annotation
rates and tests whether repeated versions of the same underlying document were
problematic as a group.

## Document-family rule

The family name is created by removing only a final suffix of the form
`_duplicated_<number>` from `filename_stem`. All other filename stems remain
unchanged. Family-level results include only families with at least two
reviewed documents after this rule is applied.

## Input

- `outputs/document_analysis.csv`: document metadata, annotation counts and
  modernised token counts.

## Tables produced by this script

- `document_ranking.csv`: all reviewed documents ranked by annotation rate.
- `document_family_summary.csv`: pooled results for repeated document families.

## Figures produced by this script

- `highest_document_annotation_rates`: the 15 highest-rate documents.
- `highest_family_annotation_rates`: up to 15 highest-rate repeated families.

Each figure is saved as both PNG and SVG.

## Interpretation

Very short documents can have high rates based on few annotations, so the
document table retains token and annotation counts. Family results depend on
the filename rule above and should be checked against knowledge of the source
collection before final reporting.
"""


# ---------------------------------------------------------------------------
# Load and validate the document-level table
# ---------------------------------------------------------------------------

check_required_files(
    [DOCUMENT_ANALYSIS_PATH],
    preceding_command="python scripts/construct_tables.py",
)

documents_df = pd.read_csv(
    DOCUMENT_ANALYSIS_PATH,
    dtype={"filename_stem": "string"},
)

required_columns = {
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

missing_columns = required_columns - set(documents_df.columns)
if missing_columns:
    raise ValueError(
        "document_analysis.csv is missing required columns: "
        f"{sorted(missing_columns)}."
    )

if documents_df["filename_stem"].duplicated().any():
    raise ValueError(
        "document_analysis.csv must contain one row per sample document."
    )

documents_df["annotation_json_found"] = coerce_boolean(
    documents_df["annotation_json_found"]
)

for column in (
    "n_modernised_tokens",
    "n_included_annotations",
    "n_included_assignments",
    "included_annotations_per_1000_tokens",
):
    documents_df[column] = pd.to_numeric(documents_df[column], errors="coerce")

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


# ---------------------------------------------------------------------------
# Derive the conservative document-family identifier
# ---------------------------------------------------------------------------

def derive_document_family(filename_stem):
    """Remove a final _duplicated_<number> suffix from a filename stem."""

    return FAMILY_SUFFIX_PATTERN.sub("", str(filename_stem))


analysis_df["document_family"] = analysis_df["filename_stem"].map(
    derive_document_family
)


# ---------------------------------------------------------------------------
# Replace this section's previous outputs
# ---------------------------------------------------------------------------

if (
    SECTION_OUTPUT_DIR.name != "04_document_family_analysis"
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
# Create the complete document ranking
# ---------------------------------------------------------------------------

document_ranking_columns = [
    "filename_stem",
    "document_family",
    "archive",
    "model",
    "reviewer_packet",
    "reviewer_name",
    "n_modernised_tokens",
    "n_included_annotations",
    "n_included_assignments",
    "included_annotations_per_1000_tokens",
]

document_ranking_df = (
    analysis_df[document_ranking_columns]
    .sort_values(
        [
            "included_annotations_per_1000_tokens",
            "n_included_annotations",
        ],
        ascending=[False, False],
    )
    .reset_index(drop=True)
)
document_ranking_df.insert(0, "rank", document_ranking_df.index + 1)

for column in (
    "rank",
    "reviewer_packet",
    "n_modernised_tokens",
    "n_included_annotations",
    "n_included_assignments",
):
    document_ranking_df[column] = document_ranking_df[column].astype("Int64")

save_table(
    document_ranking_df,
    tables_directory / "document_ranking.csv",
)


# ---------------------------------------------------------------------------
# Create the repeated-family summary
# ---------------------------------------------------------------------------

family_summary_df = (
    analysis_df.groupby("document_family", as_index=False)
    .agg(
        reviewed_documents=("filename_stem", "size"),
        distinct_archives=("archive", "nunique"),
        models_represented=("model", "nunique"),
        modernised_tokens=("n_modernised_tokens", "sum"),
        included_annotations=("n_included_annotations", "sum"),
        included_assignments=("n_included_assignments", "sum"),
        median_document_annotation_rate=(
            "included_annotations_per_1000_tokens",
            "median",
        ),
    )
)

family_summary_df = family_summary_df.loc[
    family_summary_df["reviewed_documents"].ge(
        MIN_REVIEWED_DOCUMENTS_PER_FAMILY
    )
].copy()

family_summary_df["pooled_annotations_per_1000_tokens"] = (
    family_summary_df["included_annotations"]
    / family_summary_df["modernised_tokens"]
    * 1000
)

family_summary_df = family_summary_df.sort_values(
    ["pooled_annotations_per_1000_tokens", "included_annotations"],
    ascending=[False, False],
).reset_index(drop=True)
family_summary_df.insert(0, "rank", family_summary_df.index + 1)

for column in (
    "rank",
    "reviewed_documents",
    "distinct_archives",
    "models_represented",
    "modernised_tokens",
    "included_annotations",
    "included_assignments",
):
    family_summary_df[column] = family_summary_df[column].astype("Int64")

save_table(
    family_summary_df,
    tables_directory / "document_family_summary.csv",
)


# ---------------------------------------------------------------------------
# Generate the two ranking figures
# ---------------------------------------------------------------------------

apply_plot_style()
model_colours = model_colour_map(analysis_df["model"])

document_plot_df = (
    document_ranking_df.head(N_DOCUMENTS_IN_FIGURE)
    .sort_values("included_annotations_per_1000_tokens", ascending=True)
)

family_plot_df = (
    family_summary_df.head(N_FAMILIES_IN_FIGURE)
    .sort_values("pooled_annotations_per_1000_tokens", ascending=True)
)

for language in OUTPUT_LANGUAGES:
    if language not in CHART_TEXT:
        raise ValueError(
            f"No document-family chart wording has been supplied for: {language}."
        )

    language_directory = figures_directory / language
    text = CHART_TEXT[language]

    # Figure 1: highest-rate individual documents. Model colour is useful here
    # because it reveals whether one model dominates the top of the ranking.
    fig, ax = plt.subplots(figsize=(11.5, 9.2))
    fig.subplots_adjust(top=0.72, bottom=0.17, left=0.37, right=0.92)
    bars = ax.barh(
        [fill(str(stem), 32) for stem in document_plot_df["filename_stem"]],
        document_plot_df["included_annotations_per_1000_tokens"],
        color=[
            model_colours[str(model)]
            for model in document_plot_df["model"]
        ],
        height=0.62,
    )
    ax.set_xlabel(text["documents"]["x_label"])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    labels = [
        f"{rate:.1f}"
        for rate in document_plot_df[
            "included_annotations_per_1000_tokens"
        ]
    ]
    ax.bar_label(bars, labels=labels, padding=5, fontsize=8.8)
    ax.set_xlim(
        0,
        max(
            float(
                document_plot_df[
                    "included_annotations_per_1000_tokens"
                ].max()
            )
            * 1.17,
            1,
        ),
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=9,
            markerfacecolor=colour,
            markeredgecolor="none",
            label=model,
        )
        for model, colour in model_colours.items()
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        bbox_transform=fig.transFigure,
        ncol=2,
    )
    add_chart_header(
        fig,
        **{
            key: text["documents"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "Rates can be high when a short document contains relatively few annotations; consult document_ranking.csv for counts and lengths.",
    )
    save_figure(fig, language_directory, "highest_document_annotation_rates")

    # Figure 2: highest-rate repeated families. Do not create an empty or
    # misleading chart if the filename rule finds no multi-document families.
    if not family_plot_df.empty:
        fig, ax = plt.subplots(figsize=(11.5, 8.6))
        fig.subplots_adjust(top=0.72, bottom=0.14, left=0.37, right=0.90)
        bars = ax.barh(
            [
                fill(str(family), 32)
                for family in family_plot_df["document_family"]
            ],
            family_plot_df["pooled_annotations_per_1000_tokens"],
            color=COLOURS["gold"],
            height=0.62,
        )
        ax.set_xlabel(text["families"]["x_label"])
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        labels = [
            f"{rate:.1f}  ·  n={int(n_documents)}"
            for rate, n_documents in zip(
                family_plot_df["pooled_annotations_per_1000_tokens"],
                family_plot_df["reviewed_documents"],
            )
        ]
        ax.bar_label(bars, labels=labels, padding=5, fontsize=8.8)
        ax.set_xlim(
            0,
            max(
                float(
                    family_plot_df[
                        "pooled_annotations_per_1000_tokens"
                    ].max()
                )
                * 1.28,
                1,
            ),
        )
        add_chart_header(
            fig,
            **{
                key: text["families"][key]
                for key in ("title", "description", "measure")
            },
        )
        add_figure_note(
            fig,
            "n is the number of reviewed documents in the family; family membership is derived only from the _duplicated_<number> suffix.",
        )
        save_figure(fig, language_directory, "highest_family_annotation_rates")
    else:
        print(
            "\nNo repeated document families contained at least two reviewed "
            "documents; the family figure was not produced."
        )


# ---------------------------------------------------------------------------
# Print a brief completion report
# ---------------------------------------------------------------------------

print(f"\nDocuments ranked: {len(document_ranking_df)}")
print(f"Repeated document families summarised: {len(family_summary_df)}")
print(f"\nTables saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
