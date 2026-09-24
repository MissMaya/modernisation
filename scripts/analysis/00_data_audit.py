"""
Script to audit data completeness and provide an overview

How were documents distributed between reviewers and models?

Script creates three tables and two figures under
``analysis_outputs/00_data_audit``. 
"""

# ---------------------------------------------------------------------------
# Import the required modules
# ---------------------------------------------------------------------------
import os
from pathlib import Path
import shutil
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

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
INPUT_DIR = PROJECT_DIR / "outputs"
ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "00_data_audit"
SECTION_README_PATH = SECTION_OUTPUT_DIR / "DATA_AUDIT_README.md"

DOCUMENT_ANALYSIS_PATH = INPUT_DIR / "document_analysis.csv"
ANNOTATION_ANALYSIS_PATH = INPUT_DIR / "annotation_analysis.csv"


# ---------------------------------------------------------------------------
# Specify wording for charts
# ---------------------------------------------------------------------------

# English figures are produced by default. If Spanish output is needed, add the 
# complete Spanish titles, descriptions, measure lines and axis labels here.
# then add "es" to OUTPUT_LANGUAGES in analysis_utils.py, 
CHART_TEXT = {
    "en": {
        "model_exposure": {
            "title": "How much of the modernised text came from each model?",
            "description": (
                "This chart shows the document and token-level split"
                "between the two models."
            ),
            "measure": (
                "Counts of modernised documents per model (left) and the constituent tokens "
                "reviewed per model (right)."
            ),
            "documents_label": "Percentage of reviewed documents",
            "tokens_label": "Percentage of reviewed tokens",
        },
        "reviewer_allocation": {
            "title": "How were model outputs split between reviewers?",
            "description": (
                "Chart showing the variability in the overall composition of each "
                "review packet in terms of the underlying model used for the modernisation."
            ),
            "measure": "Reviewed documents in each reviewer packet, split by model.",
            "x_label": "Reviewer packet",
            "y_label": "Percentage of reviewed documents within packet",
        },
    }
}


README_TEXT = """# 00 · Data and allocation audit

## Aim of analysis

Checks whether the analysis data are complete and extracts the split of documents
and tokens between models and reviewers. 

The documents were allocated randomly, but randomisation does not guarantee
perfect balance. The model groups can contain different
numbers or lengths of documents, reviewers can receive different mixtures of
model outputs, and archives can be concentrated under particular models or
reviewers. These patterns are considered when interpreting any descriptive
analyses and in any later statistical modelling.

Note that a completed review with no annotations is distinct from a document whose
annotation data are unavailable. Missing annotation data are not counted as
zero errors.

## Inputs

- `outputs/document_analysis.csv`: one row per sample document.
- `outputs/annotation_analysis.csv`: one row per category-field assignment.

## Tables produced by this script

- `audit_summary.csv`: headline counts for data completeness, distinct
  annotations, and included or excluded error-label assignment rows. A single
  annotation can have more than one assignment row.
- `allocation_summary.csv`: model totals, model allocation within reviewer
  packets, model allocation within archives, and archive concentration within
  reviewer packets.
- `documents_requiring_attention.csv`: documents with unavailable data or
  excluded annotation assignments.

## Figures produced by this script

- `model_sample_and_token_exposure`: reviewed document and token exposure for
  each model.
- `model_allocation_by_reviewer_packet`: the model split received by each
  reviewer.
"""


# ---------------------------------------------------------------------------
# First check, load and validate the analysis tables
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
    dtype={"filename_stem": "string"},
)

required_document_columns = {
    "filename_stem",
    "archive",
    "reviewer_packet",
    "reviewer_name",
    "model",
    "annotation_json_found",
    "modernised_text_available",
    "pre_modernisation_text_available",
    "review_data_status",
    "n_modernised_tokens",
    "n_excluded_assignments",
}
required_annotation_columns = {
    "filename_stem",
    "annotation_id",
    "include_in_analysis",
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
    duplicates = documents_df.loc[
        documents_df["filename_stem"].duplicated(keep=False),
        "filename_stem",
    ]
    raise ValueError(
        "document_analysis.csv must contain one row per sample document. "
        "Duplicate filename_stem values were found:\n"
        f"{duplicates.to_string(index=False)}"
    )

for column in (
    "annotation_json_found",
    "modernised_text_available",
    "pre_modernisation_text_available",
):
    documents_df[column] = coerce_boolean(documents_df[column])

annotations_df["include_in_analysis"] = coerce_boolean(
    annotations_df["include_in_analysis"]
)
documents_df["reviewer_packet"] = pd.to_numeric(
    documents_df["reviewer_packet"], errors="raise"
).astype("Int64")
documents_df["n_modernised_tokens"] = pd.to_numeric(
    documents_df["n_modernised_tokens"], errors="coerce"
)


# ---------------------------------------------------------------------------
# Define completeness and rate eligibility once
# ---------------------------------------------------------------------------

annotation_data_available = documents_df["annotation_json_found"].fillna(False)
modernised_text_available = documents_df[
    "modernised_text_available"
].fillna(False)
pre_modernisation_text_available = documents_df[
    "pre_modernisation_text_available"
].fillna(False)

# A document is included in the token-adjusted analyses only when its annotation
# data and modernised token count are both available.
usable_for_rate_analysis = (
    annotation_data_available
    & documents_df["n_modernised_tokens"].notna()
    & documents_df["n_modernised_tokens"].gt(0)
)

documents_df["annotation_data_available"] = annotation_data_available
documents_df["usable_for_rate_analysis"] = usable_for_rate_analysis

included_assignments = int(
    annotations_df["include_in_analysis"].fillna(False).sum()
)
excluded_assignments = int(
    (~annotations_df["include_in_analysis"].fillna(False)).sum()
)

# The annotation table is flattened: one original annotation can occupy
# several rows when the reviewer attached several category-sub-rule labels.
# Count original annotations separately using document + annotation ID.
# An annotation is included when at least one assignment row is included. 
# Only fully exclude an annotation if there is no error labelling attached.
annotation_status_df = (
    annotations_df.assign(
        assignment_included = annotations_df["include_in_analysis"].fillna(False)
    )
    .groupby(["filename_stem", "annotation_id"], dropna = False)[
        "assignment_included"
    ]
    .any()
)
recorded_distinct_annotations = len(annotation_status_df)
included_distinct_annotations = int(annotation_status_df.sum())
fully_excluded_distinct_annotations = int((~annotation_status_df).sum())


# ---------------------------------------------------------------------------
# On a re-run, replace only this section's previous outputs
# ---------------------------------------------------------------------------

if (
    SECTION_OUTPUT_DIR.name != "00_data_audit"
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
# Create the table of data completeness
# ---------------------------------------------------------------------------

audit_summary_df = pd.DataFrame(
    {
        "measure": [
            "sample_documents",
            "distinct_archives",
            "reviewer_packets",
            "reviewers",
            "models",
            "documents_with_annotation_data",
            "documents_without_annotation_data",
            "documents_usable_for_rate_analysis",
            "reviewed_documents_with_no_annotations",
            "modernised_texts_available",
            "pre_modernisation_texts_available",
            "recorded_distinct_annotations",
            "included_distinct_annotations",
            "fully_excluded_distinct_annotations",
            "recorded_assignment_rows",
            "included_assignment_rows",
            "excluded_assignment_rows",
        ],
        "value": [
            len(documents_df),
            documents_df["archive"].nunique(dropna=True),
            documents_df["reviewer_packet"].nunique(dropna=True),
            documents_df["reviewer_name"].nunique(dropna=True),
            documents_df["model"].nunique(dropna=True),
            int(annotation_data_available.sum()),
            int((~annotation_data_available).sum()),
            int(usable_for_rate_analysis.sum()),
            int(
                documents_df["review_data_status"]
                .eq("reviewed_no_annotations")
                .sum()
            ),
            int(modernised_text_available.sum()),
            int(pre_modernisation_text_available.sum()),
            recorded_distinct_annotations,
            included_distinct_annotations,
            fully_excluded_distinct_annotations,
            len(annotations_df),
            included_assignments,
            excluded_assignments,
        ],
    }
)
save_table(audit_summary_df, tables_directory / "audit_summary.csv")


# ---------------------------------------------------------------------------
# Create one filterable allocation table
# ---------------------------------------------------------------------------

def summarise_allocation(dataframe, group_columns):
    """Summarise document and token exposure for one allocation grouping."""

    rows = []
    for group_values, group_df in dataframe.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        usable_group_df = group_df.loc[group_df["usable_for_rate_analysis"]]
        row = dict(zip(group_columns, group_values))
        row.update(
            {
                "sample_documents": len(group_df),
                "documents_with_annotation_data": int(
                    group_df["annotation_data_available"].sum()
                ),
                "documents_usable_for_rate_analysis": len(usable_group_df),
                "reviewed_modernised_tokens": int(
                    usable_group_df["n_modernised_tokens"].sum()
                ),
                "median_modernised_tokens": usable_group_df[
                    "n_modernised_tokens"
                ].median(),
                "archives_represented": group_df["archive"].nunique(),
                "reviewers_represented": group_df["reviewer_name"].nunique(),
                "models_represented": group_df["model"].nunique(),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


model_allocation_df = summarise_allocation(documents_df, ["model"])
model_allocation_df.insert(0, "allocation_level", "model_total")

reviewer_model_df = summarise_allocation(
    documents_df,
    ["reviewer_packet", "reviewer_name", "model"],
)
reviewer_model_df.insert(0, "allocation_level", "reviewer_by_model")

archive_model_df = summarise_allocation(
    documents_df,
    ["archive", "model"],
)
archive_model_df.insert(0, "allocation_level", "archive_by_model")

reviewer_archive_df = summarise_allocation(
    documents_df,
    ["reviewer_packet", "reviewer_name", "archive"],
)
reviewer_archive_df.insert(0, "allocation_level", "reviewer_by_archive")

allocation_columns = [
    "allocation_level",
    "reviewer_packet",
    "reviewer_name",
    "archive",
    "model",
    "sample_documents",
    "documents_with_annotation_data",
    "documents_usable_for_rate_analysis",
    "reviewed_modernised_tokens",
    "median_modernised_tokens",
    "archives_represented",
    "reviewers_represented",
    "models_represented",
]

allocation_parts = []
for allocation_df in (
    model_allocation_df,
    reviewer_model_df,
    archive_model_df,
    reviewer_archive_df,
):
    allocation_df = allocation_df.copy()
    for column in allocation_columns:
        if column not in allocation_df.columns:
            allocation_df[column] = pd.NA
    allocation_parts.append(allocation_df[allocation_columns])

allocation_summary_df = pd.concat(allocation_parts, ignore_index=True)

for column in (
    "reviewer_packet",
    "sample_documents",
    "documents_with_annotation_data",
    "documents_usable_for_rate_analysis",
    "reviewed_modernised_tokens",
    "archives_represented",
    "reviewers_represented",
    "models_represented",
):
    allocation_summary_df[column] = allocation_summary_df[column].astype("Int64")

save_table(
    allocation_summary_df,
    tables_directory / "allocation_summary.csv",
)


# ---------------------------------------------------------------------------
# Create an exceptions table 
# ---------------------------------------------------------------------------

documents_requiring_attention_df = documents_df.loc[
    (~annotation_data_available)
    | (~modernised_text_available)
    | (~pre_modernisation_text_available)
    | documents_df["n_excluded_assignments"].fillna(0).gt(0),
    [
        "filename_stem",
        "archive",
        "reviewer_packet",
        "reviewer_name",
        "model",
        "review_data_status",
        "annotation_json_found",
        "modernised_text_available",
        "pre_modernisation_text_available",
        "n_excluded_assignments",
    ],
].copy()

save_table(
    documents_requiring_attention_df,
    tables_directory / "documents_requiring_attention.csv",
)


# ---------------------------------------------------------------------------
# Set up the two plots 
# ---------------------------------------------------------------------------

apply_plot_style()
model_colours = model_colour_map(documents_df["model"])

model_plot_df = model_allocation_df.set_index("model")
models = sorted(str(model) for model in model_plot_df.index)
model_plot_df = model_plot_df.reindex(models)

reviewer_model_plot_df = reviewer_model_df.loc[
    reviewer_model_df["documents_usable_for_rate_analysis"].gt(0)
]
reviewer_packet_pivot = (
    reviewer_model_plot_df.pivot(
        index="reviewer_packet",
        columns="model",
        values="documents_usable_for_rate_analysis",
    )
    .fillna(0)
    .sort_index()
)

# Add a short reviewer identifier beneath each packet number to make
# packet identification easier
reviewer_labels = (
    reviewer_model_plot_df[["reviewer_packet", "reviewer_name"]]
    .drop_duplicates()
    .set_index("reviewer_packet")["reviewer_name"]
)


def model_axis_label(model_name):
    """Place model capacity information on a centred second line."""

    return str(model_name).replace(" · ", "\n")

for language in OUTPUT_LANGUAGES:
    if language not in CHART_TEXT:
        raise ValueError(
            f"No audit chart wording has been supplied for language: {language}."
        )

    language_directory = figures_directory / language
    text = CHART_TEXT[language]

    # Figure 1: combine reviewed document and token exposure in one figure so
    # that unequal allocation is visible without producing repetitive charts.
    fig, axes = plt.subplots(1, 2, figsize = (12, 7.2))
    fig.subplots_adjust(
        top = 0.72,
        bottom = 0.22,
        left = 0.10,
        right = 0.94,
        wspace = 0.34,
    )

    document_counts = model_plot_df["documents_usable_for_rate_analysis"]
    token_counts = model_plot_df["reviewed_modernised_tokens"]
    document_values = document_counts / document_counts.sum()
    token_values = token_counts / token_counts.sum()
    display_labels = [model_axis_label(model) for model in models]
    colours = [model_colours[model] for model in models]

    document_bars = axes[0].bar(
        display_labels,
        document_values,
        color = colours,
        width = 0.58,
    )
    axes[0].set_ylabel(text["model_exposure"]["documents_label"])
    axes[0].bar_label(
        document_bars,
        labels = [f"{int(value):,}" for value in document_counts],
        padding = 5,
        fontsize = 10,
        fontweight = "bold",
    )

    token_bars = axes[1].bar(
        display_labels,
        token_values,
        color = colours,
        width = 0.58,
    )
    axes[1].set_ylabel(text["model_exposure"]["tokens_label"])
    axes[1].bar_label(
        token_bars,
        labels = [f"{int(value):,}" for value in token_counts],
        padding = 5,
        fontsize = 10,
        fontweight = "bold",
    )

    for ax in axes:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax = 1, decimals = 0))
        ax.grid(axis = "y")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation = 0)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")

    add_chart_header(
        fig,
        **{
            key: text["model_exposure"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "",
    )
    save_figure(fig, language_directory, "model_sample_and_token_exposure")

    # Figure 2 model allocation by reviewer
    fig, ax = plt.subplots(figsize = (11, 7.4))
    fig.subplots_adjust(top = 0.72, bottom = 0.20, left = 0.10, right = 0.92)
    packet_totals = reviewer_packet_pivot.sum(axis = 1)
    reviewer_packet_percentages = reviewer_packet_pivot.div(
        packet_totals.replace(0, pd.NA),
        axis = 0,
    ).fillna(0)
    bottom = pd.Series(0, index = reviewer_packet_percentages.index, dtype = float)

    for model in reviewer_packet_percentages.columns:
        values = reviewer_packet_percentages[model].astype(float)
        ax.bar(
            range(len(reviewer_packet_percentages.index)),
            values,
            bottom=bottom,
            label = str(model),
            color = model_colours[str(model)],
            width = 0.67,
        )
        bottom = bottom + values

    packet_axis_labels = []
    for packet in reviewer_packet_percentages.index:
        reviewer_name = str(reviewer_labels.get(packet, ""))
        short_name = reviewer_name[:4]
        packet_axis_labels.append(f"{packet}\n({short_name})")

    ax.set_xticks(range(len(packet_axis_labels)))
    ax.set_xticklabels(packet_axis_labels)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax = 1, decimals = 0))

    ax.set_xlabel(text["reviewer_allocation"]["x_label"])
    ax.set_ylabel(text["reviewer_allocation"]["y_label"])
    ax.grid(axis = "y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon = False,
        loc = "lower center",
        bbox_to_anchor = (0.5, 0.055),
        bbox_transform = fig.transFigure,
        ncol = 2,
    )
    add_chart_header(
        fig,
        **{
            key: text["reviewer_allocation"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "",
    )
    save_figure(
        fig,
        language_directory,
        "model_allocation_by_reviewer_packet",
    )


# ---------------------------------------------------------------------------
# Brief completion report for Stage 0 
# ---------------------------------------------------------------------------

print(f"\nSample documents audited: {len(documents_df)}")
print(f"Documents with annotation data: {int(annotation_data_available.sum())}")
print(f"Documents without annotation data: {int((~annotation_data_available).sum())}")
print(f"Documents usable for rate analysis: {int(usable_for_rate_analysis.sum())}")
print(f"Recorded distinct annotations: {recorded_distinct_annotations}")
print(f"Included distinct annotations: {included_distinct_annotations}")
print(f"Fully excluded distinct annotations: {fully_excluded_distinct_annotations}")
print(f"Included assignment rows: {included_assignments}")
print(f"Excluded assignment rows: {excluded_assignments}")
print(f"\nTables saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
