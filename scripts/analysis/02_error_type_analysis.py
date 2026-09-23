"""Summarise the types of errors identified by human reviewers.

This section answers two practical questions:

1. Which broad types of apparent error were assigned to each model?
2. Which specific sub-rules account for the greatest annotation burden?

Only annotation rows marked include_in_analysis=True are used. Category totals
count distinct document–annotation pairs within each category, preventing an
annotation with two fields in the same category from being counted twice.
Sub-rule totals count flattened category-field assignments because each field
is itself the sub-rule classification being summarised.

The script deliberately produces two CSVs and two figures:

    analysis_outputs/02_error_type_analysis/ERROR_TYPE_ANALYSIS_README.md
    analysis_outputs/02_error_type_analysis/tables/error_type_summary.csv
    analysis_outputs/02_error_type_analysis/tables/subrule_priority_summary.csv
    analysis_outputs/02_error_type_analysis/figures/en/
        error_category_rates_by_model.png and .svg
        priority_subrule_rates_by_model.png and .svg
"""

import os
from pathlib import Path
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
# Set up the input and output paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(
    os.environ.get("MODERNISATION_PROJECT_DIR", "/workspaces/modernisation")
)
DOCUMENT_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "document_analysis.csv"
ANNOTATION_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "annotation_analysis.csv"

ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "02_error_type_analysis"
SECTION_README_PATH = SECTION_OUTPUT_DIR / "ERROR_TYPE_ANALYSIS_README.md"


# ---------------------------------------------------------------------------
# Define the visible chart wording
# ---------------------------------------------------------------------------

CHART_TEXT = {
    "en": {
        "priority_subrules": {
            "title": "Which sub-rules drove reviewer feedback for each model?",
            "description": (
                "This comparison shows the model-specific rates for the "
                "sub-rules accounting for most included assignments."
            ),
            "measure": (
                "Assignments per 1,000 reviewed modernised tokens; sub-rules "
                "are retained until at least 80% of assignments is covered."
            ),
            "x_label": "Assignments per 1,000 tokens for each model",
        },
        "categories_by_model": {
            "title": "How did category-specific error rates differ by model?",
            "description": (
                "This comparison shows whether particular types of error "
                "were more commonly assigned to either model."
            ),
            "measure": (
                "Distinct annotations in each category per 1,000 modernised "
                "tokens reviewed for that model."
            ),
            "x_label": "Annotations per 1,000 tokens",
        },
    }
}


README_TEXT = """# 02 · Error-type analysis

## What this section is trying to show

This section identifies which types of modernisation error reviewers assigned
most frequently and compares category-specific rates between the two models.

Only assignments retained for analysis are used. Category totals count each
annotation once within a category, even when it has multiple fields in that
category. Sub-rule totals count the individual category-field assignments.

## Inputs

- `outputs/document_analysis.csv`: supplies reviewed token totals for each model.
- `outputs/annotation_analysis.csv`: supplies the included error-category and
  sub-rule assignments.

## Tables produced by this script

- `error_type_summary.csv`: category and sub-rule counts and rates, overall and
  by model.
- `subrule_priority_summary.csv`: one ranked row per sub-rule, including the
  overall burden, documents affected and the observed rate for each model.

## Figures produced by this script

- `error_category_rates_by_model`: category-specific annotation rates for the
  two models.
- `priority_subrule_rates_by_model`: model-specific rates for the sub-rules that
  together account for at least 80% of included assignments. Sub-rules tied at
  the cutoff are retained.

Each figure is saved as both PNG and SVG.

## Interpretation

The figures describe categories and sub-rules assigned by reviewers. They do not show
how many opportunities each model had to apply each individual modernisation
rule, and they do not yet adjust for archive or reviewer effects.
"""


# ---------------------------------------------------------------------------
# Check and load both analysis tables
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
    "model",
    "annotation_json_found",
    "n_modernised_tokens",
}

required_annotation_columns = {
    "filename_stem",
    "annotation_id",
    "model",
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
documents_df["n_modernised_tokens"] = pd.to_numeric(
    documents_df["n_modernised_tokens"], errors="coerce"
)


# ---------------------------------------------------------------------------
# Select the reviewed documents and included annotation assignments
# ---------------------------------------------------------------------------

reviewed_documents_df = documents_df.loc[
    documents_df["annotation_json_found"].fillna(False)
    & documents_df["n_modernised_tokens"].notna()
    & documents_df["n_modernised_tokens"].gt(0)
].copy()

if reviewed_documents_df.empty:
    raise ValueError(
        "No documents have both available annotation data and a positive "
        "modernised token count."
    )

included_annotations_df = annotations_df.loc[
    annotations_df["include_in_analysis"].fillna(False)
].copy()

# Retain only annotation rows belonging to documents whose token counts can be
# used as denominators. The inner join also prevents an unmatched annotation
# from contributing to an error-type rate.
included_annotations_df = included_annotations_df.merge(
    reviewed_documents_df[["filename_stem", "model"]],
    on=["filename_stem", "model"],
    how="inner",
    validate="many_to_one",
)

if included_annotations_df.empty:
    raise ValueError("No included annotation assignments are available to summarise.")

if included_annotations_df["effective_error_category"].isna().any():
    raise ValueError(
        "At least one included annotation assignment has no effective error "
        "category. Check the Stage 6 resolution output."
    )

models = sorted(
    str(model) for model in reviewed_documents_df["model"].dropna().unique()
)
if len(models) != 2:
    raise ValueError("The error-type comparison expects exactly two models.")

tokens_by_model = (
    reviewed_documents_df.groupby("model")["n_modernised_tokens"].sum()
)
total_reviewed_tokens = reviewed_documents_df["n_modernised_tokens"].sum()


# ---------------------------------------------------------------------------
# Replace this section's earlier outputs
# ---------------------------------------------------------------------------

if (
    SECTION_OUTPUT_DIR.name != "02_error_type_analysis"
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
# Build category-level summaries
# ---------------------------------------------------------------------------

# An annotation can contain several fields within one category. Drop repeated
# document–annotation–category combinations before calculating category totals.
category_annotations_df = included_annotations_df.drop_duplicates(
    subset=[
        "filename_stem",
        "annotation_id",
        "model",
        "effective_error_category",
    ]
)

category_by_model_df = (
    category_annotations_df.groupby(
        ["effective_error_category", "model"],
        as_index=False,
    )
    .agg(count=("annotation_id", "size"))
)
category_by_model_df["rate_per_1000_tokens"] = category_by_model_df.apply(
    lambda row: row["count"] / tokens_by_model.loc[row["model"]] * 1000,
    axis=1,
)
category_by_model_df = category_by_model_df.rename(
    columns={"effective_error_category": "error_category"}
)
category_by_model_df.insert(0, "summary_level", "category_by_model")
category_by_model_df.insert(2, "subrule", "All sub-rules")
category_by_model_df["count_definition"] = (
    "distinct_document_annotation_pairs_within_category"
)

category_overall_df = (
    category_annotations_df.groupby(
        "effective_error_category",
        as_index=False,
    )
    .agg(count=("annotation_id", "size"))
    .rename(columns={"effective_error_category": "error_category"})
)
category_overall_df["rate_per_1000_tokens"] = (
    category_overall_df["count"] / total_reviewed_tokens * 1000
)
category_overall_df.insert(0, "summary_level", "category_overall")
category_overall_df.insert(2, "subrule", "All sub-rules")
category_overall_df.insert(3, "model", "All models")
category_overall_df["count_definition"] = (
    "distinct_document_annotation_pairs_within_category"
)


# ---------------------------------------------------------------------------
# Build sub-rule summaries
# ---------------------------------------------------------------------------

# A missing field code represents a category-only annotation. Preserve it in
# the summary under an explicit label instead of silently dropping it.
included_annotations_df["subrule_for_summary"] = (
    included_annotations_df["effective_subrule"]
    .fillna("No sub-rule assigned")
)

# Count a category–sub-rule assignment only once within a source annotation.
# This protects the summary against accidental duplicate flattened rows while
# preserving genuinely different sub-rules attached to the same annotation.
subrule_annotations_df = included_annotations_df.drop_duplicates(
    subset=[
        "filename_stem",
        "annotation_id",
        "model",
        "effective_error_category",
        "subrule_for_summary",
    ]
)

subrule_by_model_df = (
    subrule_annotations_df.groupby(
        [
            "effective_error_category",
            "subrule_for_summary",
            "model",
        ],
        as_index=False,
    )
    .agg(count=("annotation_id", "size"))
    .rename(
        columns={
            "effective_error_category": "error_category",
            "subrule_for_summary": "subrule",
        }
    )
)
subrule_by_model_df["rate_per_1000_tokens"] = subrule_by_model_df.apply(
    lambda row: row["count"] / tokens_by_model.loc[row["model"]] * 1000,
    axis=1,
)
subrule_by_model_df.insert(0, "summary_level", "subrule_by_model")
subrule_by_model_df["count_definition"] = "included_category_field_assignments"

subrule_overall_df = (
    subrule_annotations_df.groupby(
        ["effective_error_category", "subrule_for_summary"],
        as_index=False,
    )
    .agg(count=("annotation_id", "size"))
    .rename(
        columns={
            "effective_error_category": "error_category",
            "subrule_for_summary": "subrule",
        }
    )
)
subrule_overall_df["rate_per_1000_tokens"] = (
    subrule_overall_df["count"] / total_reviewed_tokens * 1000
)
subrule_overall_df.insert(0, "summary_level", "subrule_overall")
subrule_overall_df.insert(3, "model", "All models")
subrule_overall_df["count_definition"] = "included_category_field_assignments"

output_columns = [
    "summary_level",
    "error_category",
    "subrule",
    "model",
    "count",
    "rate_per_1000_tokens",
    "count_definition",
]

error_type_summary_df = pd.concat(
    [
        category_overall_df[output_columns],
        category_by_model_df[output_columns],
        subrule_overall_df[output_columns],
        subrule_by_model_df[output_columns],
    ],
    ignore_index=True,
)

save_table(
    error_type_summary_df,
    tables_directory / "error_type_summary.csv",
)


# ---------------------------------------------------------------------------
# Create a ranked, human-readable sub-rule priority table
# ---------------------------------------------------------------------------

subrule_document_counts = (
    subrule_annotations_df.groupby(
        ["effective_error_category", "subrule_for_summary"]
    )["filename_stem"]
    .nunique()
    .rename("documents_affected")
    .reset_index()
    .rename(
        columns={
            "effective_error_category": "error_category",
            "subrule_for_summary": "subrule",
        }
    )
)

subrule_priority_df = subrule_overall_df[
    ["error_category", "subrule", "count", "rate_per_1000_tokens"]
].rename(columns={"count": "included_assignments"})
subrule_priority_df = subrule_priority_df.merge(
    subrule_document_counts,
    on=["error_category", "subrule"],
    how="left",
    validate="one_to_one",
)
subrule_priority_df["share_of_included_assignments_pct"] = (
    subrule_priority_df["included_assignments"]
    / len(subrule_annotations_df)
    * 100
)

# Add the observed count and rate for each model to the same ranked row. The
# model names are stored as values rather than embedded in column names, making
# the output stable if the display names change later.
for position, model in enumerate(models, start=1):
    model_values = (
        subrule_by_model_df.loc[
            subrule_by_model_df["model"].astype(str).eq(model),
            ["error_category", "subrule", "count", "rate_per_1000_tokens"],
        ]
        .rename(
            columns={
                "count": f"model_{position}_assignments",
                "rate_per_1000_tokens": f"model_{position}_rate_per_1000_tokens",
            }
        )
    )
    subrule_priority_df = subrule_priority_df.merge(
        model_values,
        on=["error_category", "subrule"],
        how="left",
        validate="one_to_one",
    )
    subrule_priority_df.insert(
        subrule_priority_df.columns.get_loc(f"model_{position}_assignments"),
        f"model_{position}",
        model,
    )

count_columns = [f"model_{position}_assignments" for position in (1, 2)]
rate_columns = [
    f"model_{position}_rate_per_1000_tokens" for position in (1, 2)
]
subrule_priority_df[count_columns + rate_columns] = (
    subrule_priority_df[count_columns + rate_columns].fillna(0)
)
subrule_priority_df["absolute_model_rate_difference"] = (
    subrule_priority_df[rate_columns[0]] - subrule_priority_df[rate_columns[1]]
).abs()
subrule_priority_df = subrule_priority_df.sort_values(
    ["rate_per_1000_tokens", "included_assignments"],
    ascending=False,
).reset_index(drop=True)
subrule_priority_df.insert(0, "priority_rank", range(1, len(subrule_priority_df) + 1))
subrule_priority_df["cumulative_assignment_share_pct"] = (
    subrule_priority_df["included_assignments"].cumsum()
    / subrule_priority_df["included_assignments"].sum()
    * 100
)

save_table(
    subrule_priority_df,
    tables_directory / "subrule_priority_summary.csv",
)


# ---------------------------------------------------------------------------
# Generate the two error-type figures
# ---------------------------------------------------------------------------

apply_plot_style()
model_colours = model_colour_map(models)

category_order = category_overall_df.sort_values(
    "count", ascending=False
)["error_category"].tolist()
# Select the smallest leading set whose cumulative assignment count reaches
# 80%. If the final included sub-rule is tied with others on assignment count,
# retain the full tie rather than separating equally frequent sub-rules.
cutoff_row_index = subrule_priority_df[
    "cumulative_assignment_share_pct"
].ge(80).idxmax()
cutoff_assignment_count = subrule_priority_df.loc[
    cutoff_row_index, "included_assignments"
]
top_subrules_plot_df = subrule_priority_df.loc[
    subrule_priority_df["included_assignments"].ge(cutoff_assignment_count)
].copy()
top_subrules_plot_df = top_subrules_plot_df.sort_values(
    ["included_assignments", "rate_per_1000_tokens"],
    ascending=True,
)
top_subrules_plot_df["display_label"] = top_subrules_plot_df.apply(
    lambda row: fill(f"{row['error_category']} — {row['subrule']}", 42),
    axis=1,
)

for language in OUTPUT_LANGUAGES:
    if language not in CHART_TEXT:
        raise ValueError(
            f"No error-type chart wording has been supplied for: {language}."
        )

    language_directory = figures_directory / language
    text = CHART_TEXT[language]

    # Figure 1: compare model-specific rates for the sub-rules responsible for
    # most of the reviewer feedback. The selection is based on overall counts,
    # while the displayed rates account for unequal model token exposure.
    figure_height = max(8.0, 4.8 + 0.46 * len(top_subrules_plot_df))
    fig, ax = plt.subplots(figsize=(12, figure_height))
    fig.subplots_adjust(top=0.72, bottom=0.15, left=0.43, right=0.92)
    y_positions = list(range(len(top_subrules_plot_df)))

    for y_position, (_, row) in zip(
        y_positions, top_subrules_plot_df.iterrows()
    ):
        ax.plot(
            [
                row["model_1_rate_per_1000_tokens"],
                row["model_2_rate_per_1000_tokens"],
            ],
            [y_position, y_position],
            color=COLOURS["grid"],
            linewidth=2,
            zorder=1,
        )

    for position, model in enumerate(models, start=1):
        ax.scatter(
            top_subrules_plot_df[
                f"model_{position}_rate_per_1000_tokens"
            ],
            y_positions,
            label=model,
            color=model_colours[model],
            s=68,
            edgecolor=COLOURS["panel"],
            linewidth=0.7,
            zorder=3,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_subrules_plot_df["display_label"])
    ax.set_xlabel(text["priority_subrules"]["x_label"])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        bbox_transform=fig.transFigure,
        ncol=2,
    )
    add_chart_header(
        fig,
        **{
            key: text["priority_subrules"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "Sub-rules are ranked by overall assignment count and retained until at least 80% is represented; ties at the cutoff are included.",
    )
    save_figure(fig, language_directory, "priority_subrule_rates_by_model")

    # Figure 2: category-specific rates for both models.
    comparison_df = (
        category_by_model_df.pivot(
            index="error_category",
            columns="model",
            values="rate_per_1000_tokens",
        )
        .reindex(category_order)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    fig.subplots_adjust(top=0.72, bottom=0.17, left=0.36, right=0.92)
    y_positions = list(range(len(comparison_df)))

    # Light connecting lines make the size and direction of each model
    # difference visible without adding another set of bars.
    for y_position, (_, row) in zip(y_positions, comparison_df.iterrows()):
        ax.plot(
            [row[models[0]], row[models[1]]],
            [y_position, y_position],
            color=COLOURS["grid"],
            linewidth=2,
            zorder=1,
        )

    for model in models:
        ax.scatter(
            comparison_df[model],
            y_positions,
            label=model,
            color=model_colours[model],
            s=72,
            edgecolor=COLOURS["panel"],
            linewidth=0.7,
            zorder=3,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([fill(category, 30) for category in comparison_df.index])
    ax.invert_yaxis()
    ax.set_xlabel(text["categories_by_model"]["x_label"])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        bbox_transform=fig.transFigure,
        ncol=2,
    )
    add_chart_header(
        fig,
        **{
            key: text["categories_by_model"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "Rates use the total modernised tokens in reviewed documents for each model; they are not rule-opportunity rates.",
    )
    save_figure(fig, language_directory, "error_category_rates_by_model")


# ---------------------------------------------------------------------------
# Print a brief completion report
# ---------------------------------------------------------------------------

print(f"\nIncluded assignment rows summarised: {len(included_annotations_df)}")
print(f"Distinct error categories: {category_overall_df['error_category'].nunique()}")
print(
    "Sub-rules displayed in the 80% chart: "
    f"{len(top_subrules_plot_df)} of {len(subrule_priority_df)}"
)
print(f"\nSummary table saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
