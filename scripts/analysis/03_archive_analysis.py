"""Identify archives associated with higher modernisation error rates.

This section asks whether documents from some archives attracted more reviewer
annotations than others and whether the two models showed the same pattern in
the best-represented archives.

The primary archive measure is the pooled number of distinct included
annotations per 1,000 modernised tokens. Pooling means that annotations and
tokens are summed across the reviewed documents in an archive before the rate
is calculated. The output also retains document-level medians and sample sizes
so that rates based on very small groups can be recognised.

The script produces only one table and two focused figures:

    analysis_outputs/03_archive_analysis/ARCHIVE_ANALYSIS_README.md
    analysis_outputs/03_archive_analysis/tables/archive_performance_summary.csv
    analysis_outputs/03_archive_analysis/figures/en/
        highest_archive_annotation_rates.png and .svg
        model_rates_in_best_represented_archives.png and .svg
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
# Set paths and deliberately small display limits
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(
    os.environ.get("MODERNISATION_PROJECT_DIR", "/workspaces/modernisation")
)
DOCUMENT_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "document_analysis.csv"

ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "03_archive_analysis"
SECTION_README_PATH = SECTION_OUTPUT_DIR / "ARCHIVE_ANALYSIS_README.md"

# The table retains every archive. Limiting the figures prevents a 48-archive
# chart from becoming too dense to communicate a useful message.
N_ARCHIVES_IN_RANKING_FIGURE = 15
N_ARCHIVES_IN_MODEL_FIGURE = 15


# ---------------------------------------------------------------------------
# Define visible chart wording
# ---------------------------------------------------------------------------

CHART_TEXT = {
    "en": {
        "archive_ranking": {
            "title": "Which archives had the highest annotation rates?",
            "description": (
                "This ranking identifies archives whose reviewed documents "
                "attracted the most reviewer-marked errors relative to length."
            ),
            "measure": (
                "Pooled distinct included annotations per 1,000 modernised "
                "tokens; the 15 highest-rate archives are shown."
            ),
            "x_label": "Annotations per 1,000 tokens",
        },
        "models_by_archive": {
            "title": "How did the models compare in the best-represented archives?",
            "description": (
                "This comparison shows whether the model difference was "
                "similar across archives with observations for both models."
            ),
            "measure": (
                "Pooled distinct annotations per 1,000 modernised tokens in "
                "up to 15 archives with the most reviewed documents."
            ),
            "x_label": "Annotations per 1,000 tokens",
        },
    }
}


README_TEXT = """# 03 · Archive analysis

## What this section is trying to show

This section identifies archives whose documents received comparatively high
annotation rates and examines whether the two models showed similar patterns
within the best-represented archives.

Rates are calculated by summing distinct included annotations and modernised
tokens within each group, then reporting annotations per 1,000 tokens.

## Input

- `outputs/document_analysis.csv`: document metadata, annotation counts and
  modernised token counts.

## Table produced by this script

- `archive_performance_summary.csv`: overall and model-specific document
  counts, annotation totals and annotation rates for every archive.

## Figures produced by this script

- `highest_archive_annotation_rates`: the 15 archives with the highest overall
  annotation rates.
- `model_rates_in_best_represented_archives`: model-specific rates in up to 15
  archives with the largest reviewed samples and observations for both models.

Each figure is saved as both PNG and SVG.

## Interpretation

Archive rates based on few documents are unstable. Sample sizes are retained
in the table and shown beside the overall archive ranking. These descriptive
results do not yet adjust jointly for reviewer, model and archive effects.
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
    "annotation_json_found",
    "n_modernised_tokens",
    "n_included_annotations",
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

models = sorted(str(model) for model in analysis_df["model"].dropna().unique())
if len(models) != 2:
    raise ValueError("The archive comparison expects exactly two models.")


# ---------------------------------------------------------------------------
# Replace this section's previous outputs
# ---------------------------------------------------------------------------

if (
    SECTION_OUTPUT_DIR.name != "03_archive_analysis"
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
# Summarise one archive group
# ---------------------------------------------------------------------------

def summarise_archive_group(group_df):
    """Return the concise performance measures used for one archive group."""

    total_tokens = group_df["n_modernised_tokens"].sum()
    total_annotations = group_df["n_included_annotations"].sum()
    zero_documents = group_df["n_included_annotations"].eq(0).sum()

    return pd.Series(
        {
            "documents_in_comparison": len(group_df),
            "modernised_tokens": int(total_tokens),
            "included_annotations": int(total_annotations),
            "pooled_annotations_per_1000_tokens": (
                total_annotations / total_tokens * 1000
            ),
            "median_document_annotation_rate": group_df[
                "included_annotations_per_1000_tokens"
            ].median(),
            "documents_with_zero_annotations": int(zero_documents),
            "documents_with_zero_annotations_pct": (
                zero_documents / len(group_df) * 100
            ),
        }
    )


# ---------------------------------------------------------------------------
# Build and save the complete archive summary
# ---------------------------------------------------------------------------

archive_overall_df = (
    analysis_df.groupby("archive", dropna=False)
    .apply(summarise_archive_group, include_groups=False)
    .reset_index()
)
archive_overall_df.insert(0, "summary_level", "archive_overall")
archive_overall_df.insert(2, "model", "All models")

archive_by_model_df = (
    analysis_df.groupby(["archive", "model"], dropna=False)
    .apply(summarise_archive_group, include_groups=False)
    .reset_index()
)
archive_by_model_df.insert(0, "summary_level", "archive_by_model")

# Record how many sample documents were originally assigned to each group,
# including any documents omitted because review data were unavailable.
sample_overall_df = (
    documents_df.groupby("archive", dropna=False)
    .size()
    .rename("sample_documents")
    .reset_index()
)
sample_by_model_df = (
    documents_df.groupby(["archive", "model"], dropna=False)
    .size()
    .rename("sample_documents")
    .reset_index()
)

archive_overall_df = archive_overall_df.merge(
    sample_overall_df,
    on="archive",
    how="left",
    validate="one_to_one",
)
archive_by_model_df = archive_by_model_df.merge(
    sample_by_model_df,
    on=["archive", "model"],
    how="left",
    validate="one_to_one",
)

output_columns = [
    "summary_level",
    "archive",
    "model",
    "sample_documents",
    "documents_in_comparison",
    "modernised_tokens",
    "included_annotations",
    "pooled_annotations_per_1000_tokens",
    "median_document_annotation_rate",
    "documents_with_zero_annotations",
    "documents_with_zero_annotations_pct",
]

archive_summary_df = pd.concat(
    [
        archive_overall_df[output_columns],
        archive_by_model_df[output_columns],
    ],
    ignore_index=True,
)

for column in (
    "sample_documents",
    "documents_in_comparison",
    "modernised_tokens",
    "included_annotations",
    "documents_with_zero_annotations",
):
    archive_summary_df[column] = archive_summary_df[column].astype("Int64")

archive_summary_df["_summary_order"] = archive_summary_df[
    "summary_level"
].map({"archive_overall": 0, "archive_by_model": 1})
archive_summary_df = archive_summary_df.sort_values(
    ["_summary_order", "pooled_annotations_per_1000_tokens"],
    ascending=[True, False],
).drop(columns="_summary_order")

save_table(
    archive_summary_df,
    tables_directory / "archive_performance_summary.csv",
)


# ---------------------------------------------------------------------------
# Prepare the limited plotting subsets
# ---------------------------------------------------------------------------

ranking_plot_df = (
    archive_overall_df.nlargest(
        N_ARCHIVES_IN_RANKING_FIGURE,
        "pooled_annotations_per_1000_tokens",
    )
    .sort_values("pooled_annotations_per_1000_tokens", ascending=True)
)

model_rate_pivot = archive_by_model_df.pivot(
    index="archive",
    columns="model",
    values="pooled_annotations_per_1000_tokens",
)
model_document_pivot = archive_by_model_df.pivot(
    index="archive",
    columns="model",
    values="documents_in_comparison",
)

# Compare only archives represented by both models. Select the archives with
# the most reviewed documents before sorting them by the size of the observed
# model difference.
comparable_archives = model_rate_pivot.dropna(subset=models).index
model_comparison_df = model_rate_pivot.loc[comparable_archives].copy()
model_comparison_df["reviewed_documents"] = (
    model_document_pivot.loc[comparable_archives, models].sum(axis=1)
)
model_comparison_df = model_comparison_df.nlargest(
    N_ARCHIVES_IN_MODEL_FIGURE,
    "reviewed_documents",
)
model_comparison_df["absolute_difference"] = (
    model_comparison_df[models[0]] - model_comparison_df[models[1]]
).abs()
model_comparison_df = model_comparison_df.sort_values(
    "absolute_difference",
    ascending=False,
)


# ---------------------------------------------------------------------------
# Generate the two archive figures
# ---------------------------------------------------------------------------

apply_plot_style()
model_colours = model_colour_map(models)

for language in OUTPUT_LANGUAGES:
    if language not in CHART_TEXT:
        raise ValueError(
            f"No archive chart wording has been supplied for: {language}."
        )

    language_directory = figures_directory / language
    text = CHART_TEXT[language]

    # Figure 1: archives with the highest overall annotation rates.
    fig, ax = plt.subplots(figsize=(11, 9.2))
    fig.subplots_adjust(top=0.72, bottom=0.13, left=0.34, right=0.90)
    bars = ax.barh(
        [fill(str(archive), 28) for archive in ranking_plot_df["archive"]],
        ranking_plot_df["pooled_annotations_per_1000_tokens"],
        color=COLOURS["gold"],
        height=0.62,
    )
    ax.set_xlabel(text["archive_ranking"]["x_label"])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    labels = [
        f"{rate:.1f}  ·  n={int(n_documents)}"
        for rate, n_documents in zip(
            ranking_plot_df["pooled_annotations_per_1000_tokens"],
            ranking_plot_df["documents_in_comparison"],
        )
    ]
    ax.bar_label(bars, labels=labels, padding=5, fontsize=8.8)
    ax.set_xlim(
        0,
        max(
            float(ranking_plot_df["pooled_annotations_per_1000_tokens"].max())
            * 1.30,
            1,
        ),
    )
    add_chart_header(
        fig,
        **{
            key: text["archive_ranking"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "n is the number of reviewed documents contributing to the archive rate; small groups should be interpreted cautiously.",
    )
    save_figure(fig, language_directory, "highest_archive_annotation_rates")

    # Figure 2: model rates in the best-represented comparable archives.
    if not model_comparison_df.empty:
        fig, ax = plt.subplots(figsize=(11.5, 9.2))
        fig.subplots_adjust(top=0.72, bottom=0.17, left=0.34, right=0.92)
        y_positions = list(range(len(model_comparison_df)))

        for y_position, (_, row) in zip(
            y_positions,
            model_comparison_df.iterrows(),
        ):
            ax.plot(
                [row[models[0]], row[models[1]]],
                [y_position, y_position],
                color=COLOURS["grid"],
                linewidth=2,
                zorder=1,
            )

        for model in models:
            ax.scatter(
                model_comparison_df[model],
                y_positions,
                label=model,
                color=model_colours[model],
                s=68,
                edgecolor=COLOURS["panel"],
                linewidth=0.7,
                zorder=3,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [fill(str(archive), 28) for archive in model_comparison_df.index]
        )
        ax.invert_yaxis()
        ax.set_xlabel(text["models_by_archive"]["x_label"])
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
                key: text["models_by_archive"][key]
                for key in ("title", "description", "measure")
            },
        )
        add_figure_note(
            fig,
            "This is an exploratory within-archive comparison. The CSV retains the contributing document counts for each model.",
        )
        save_figure(
            fig,
            language_directory,
            "model_rates_in_best_represented_archives",
        )
    else:
        print(
            "\nNo archive contained reviewed documents from both models; "
            "the within-archive model figure was not produced."
        )


# ---------------------------------------------------------------------------
# Print a brief completion report
# ---------------------------------------------------------------------------

print(f"\nArchives summarised: {archive_overall_df['archive'].nunique()}")
print(f"Documents included in archive rates: {len(analysis_df)}")
print(f"\nSummary table saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
