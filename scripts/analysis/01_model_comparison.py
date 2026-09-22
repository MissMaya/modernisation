"""Compare the two modernisation models using human-review annotations.

This section answers one central question: did reviewers identify fewer errors
in documents modernised by one model than in documents modernised by the other?

The primary measure is the number of distinct included annotations per 1,000
modernised tokens. A single annotation may have more than one category or
field assignment, but it is counted once here as one reviewer-marked error.

Only documents with available annotation data and a positive modernised token
count are used when calculating document-level rates. A missing annotation JSON
is not interpreted as a document with zero errors.

The script deliberately produces a small set of outputs:

    analysis_outputs/01_model_comparison/MODEL_COMPARISON_README.md
    analysis_outputs/01_model_comparison/tables/model_performance_summary.csv
    analysis_outputs/01_model_comparison/figures/en/
        annotation_rate_by_model.png and .svg
        zero_annotation_documents_by_model.png and .svg
"""

import os
from pathlib import Path
import shutil
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
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

ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "01_model_comparison"
SECTION_README_PATH = SECTION_OUTPUT_DIR / "MODEL_COMPARISON_README.md"


# ---------------------------------------------------------------------------
# Define the visible chart wording
# ---------------------------------------------------------------------------

CHART_TEXT = {
    "en": {
        "annotation_rate": {
            "title": "Which model received fewer reviewer annotations?",
            "description": (
                "This comparison shows the variation in reviewer-marked "
                "errors across documents modernised by each model."
            ),
            "measure": (
                "Distinct included annotations per 1,000 modernised tokens "
                "for each reviewed document."
            ),
            "y_label": "Annotations per 1,000 tokens",
        },
        "zero_annotations": {
            "title": "How often did reviewers record no errors?",
            "description": (
                "This comparison shows the share of reviewed documents in "
                "which no annotations were recorded."
            ),
            "measure": (
                "Percentage of reviewed documents with zero included "
                "annotations."
            ),
            "y_label": "Reviewed documents with no annotations (%)",
        },
    }
}


README_TEXT = """# 01 · Model comparison

## What this section is trying to show

This section compares the number of errors identified by human reviewers in
documents modernised by the two models.

The main measure is **distinct included annotations per 1,000 modernised
tokens**. Each reviewer-marked annotation is counted once even when it has more
than one category or field assignment.

Documents with unavailable annotation data are excluded from the comparison;
they are not treated as documents with zero errors.

## Input

- `outputs/document_analysis.csv`: one row per sample document.

## Table produced by this script

- `model_performance_summary.csv`: reviewed documents, text length, annotation
  totals, annotation rates and zero-annotation documents for each model.

## Figures produced by this script

- `annotation_rate_by_model`: document-level annotation rates for each model.
- `zero_annotation_documents_by_model`: percentage of reviewed documents with
  no included annotations.

Each figure is saved as both PNG and SVG.

## Interpretation

These are descriptive comparisons. They do not yet adjust for archive,
reviewer or other document-level differences, and they do not constitute the
final statistical model comparison.
"""


# ---------------------------------------------------------------------------
# Check and load the document-level analysis table
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
    "model",
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

numeric_columns = [
    "n_modernised_tokens",
    "n_included_annotations",
    "n_included_assignments",
    "included_annotations_per_1000_tokens",
]
for column in numeric_columns:
    documents_df[column] = pd.to_numeric(documents_df[column], errors="coerce")


# ---------------------------------------------------------------------------
# Select documents that have both review data and a valid rate denominator
# ---------------------------------------------------------------------------

usable_document = (
    documents_df["annotation_json_found"].fillna(False)
    & documents_df["n_modernised_tokens"].notna()
    & documents_df["n_modernised_tokens"].gt(0)
)

analysis_df = documents_df.loc[usable_document].copy()

if analysis_df.empty:
    raise ValueError(
        "No documents have both available annotation data and a positive "
        "modernised token count."
    )

if analysis_df["model"].nunique(dropna=True) != 2:
    raise ValueError(
        "The model comparison expects exactly two model values after documents "
        "with unavailable data are removed."
    )


# ---------------------------------------------------------------------------
# Replace this section's earlier outputs
# ---------------------------------------------------------------------------

if (
    SECTION_OUTPUT_DIR.name != "01_model_comparison"
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
# Create the single model-performance summary table
# ---------------------------------------------------------------------------

summary_rows = []

for model, model_df in analysis_df.groupby("model", sort=True):
    total_tokens = model_df["n_modernised_tokens"].sum()
    total_annotations = model_df["n_included_annotations"].sum()
    total_assignments = model_df["n_included_assignments"].sum()
    annotation_rates = model_df["included_annotations_per_1000_tokens"]
    zero_annotation_documents = model_df["n_included_annotations"].eq(0).sum()

    summary_rows.append(
        {
            "model": model,
            "sample_documents": int(documents_df["model"].eq(model).sum()),
            "documents_in_comparison": len(model_df),
            "modernised_tokens": int(total_tokens),
            "included_annotations": int(total_annotations),
            "included_assignments": int(total_assignments),
            "pooled_annotations_per_1000_tokens": (
                total_annotations / total_tokens * 1000
            ),
            "median_document_annotation_rate": annotation_rates.median(),
            "mean_document_annotation_rate": annotation_rates.mean(),
            "first_quartile_document_annotation_rate": annotation_rates.quantile(0.25),
            "third_quartile_document_annotation_rate": annotation_rates.quantile(0.75),
            "documents_with_zero_annotations": int(zero_annotation_documents),
            "documents_with_zero_annotations_pct": (
                zero_annotation_documents / len(model_df) * 100
            ),
        }
    )

model_summary_df = pd.DataFrame(summary_rows)
save_table(
    model_summary_df,
    tables_directory / "model_performance_summary.csv",
)


# ---------------------------------------------------------------------------
# Generate two focused model-comparison figures
# ---------------------------------------------------------------------------

apply_plot_style()
model_colours = model_colour_map(analysis_df["model"])
models = sorted(str(model) for model in analysis_df["model"].dropna().unique())

for language in OUTPUT_LANGUAGES:
    if language not in CHART_TEXT:
        raise ValueError(
            f"No model-comparison chart wording has been supplied for: {language}."
        )

    language_directory = figures_directory / language
    text = CHART_TEXT[language]

    # Figure 1: distribution of document-level annotation rates.
    rate_groups = [
        analysis_df.loc[
            analysis_df["model"].astype(str).eq(model),
            "included_annotations_per_1000_tokens",
        ].dropna()
        for model in models
    ]

    fig, ax = plt.subplots(figsize=(10, 7.4))
    fig.subplots_adjust(top=0.72, bottom=0.22, left=0.14, right=0.92)

    boxplot = ax.boxplot(
        rate_groups,
        tick_labels=[fill(model, 22) for model in models],
        patch_artist=True,
        widths=0.50,
        showfliers=False,
        medianprops={"color": COLOURS["text"], "linewidth": 2.0},
        whiskerprops={"color": COLOURS["muted_text"]},
        capprops={"color": COLOURS["muted_text"]},
    )

    for patch, model in zip(boxplot["boxes"], models):
        patch.set_facecolor(model_colours[model])
        patch.set_edgecolor("none")
        patch.set_alpha(0.72)

    # Add every document as a lightly jittered point so the box plot does not
    # conceal the distribution or the number of observations.
    random_generator = np.random.default_rng(42)
    for position, (model, rates) in enumerate(zip(models, rate_groups), start=1):
        jitter = random_generator.uniform(-0.12, 0.12, size=len(rates))
        ax.scatter(
            np.full(len(rates), position) + jitter,
            rates,
            color=model_colours[model],
            edgecolor=COLOURS["panel"],
            linewidth=0.35,
            s=24,
            alpha=0.72,
            zorder=3,
        )

    ax.set_ylabel(text["annotation_rate"]["y_label"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    add_chart_header(
        fig,
        **{
            key: text["annotation_rate"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "Each point represents one document. Documents with unavailable annotation data are omitted.",
    )
    save_figure(fig, language_directory, "annotation_rate_by_model")

    # Figure 2: percentage of reviewed documents with zero annotations.
    zero_summary = model_summary_df.set_index("model").reindex(models)
    zero_percentages = zero_summary["documents_with_zero_annotations_pct"]

    fig, ax = plt.subplots(figsize=(10, 7.0))
    fig.subplots_adjust(top=0.72, bottom=0.22, left=0.16, right=0.92)
    bars = ax.bar(
        [fill(model, 22) for model in models],
        zero_percentages,
        color=[model_colours[model] for model in models],
        width=0.55,
    )
    ax.set_ylabel(text["zero_annotations"]["y_label"])
    ax.set_ylim(0, max(float(zero_percentages.max()) * 1.25, 10))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(
        bars,
        labels=[f"{value:.1f}%" for value in zero_percentages],
        padding=5,
        fontsize=10,
        fontweight="bold",
    )
    add_chart_header(
        fig,
        **{
            key: text["zero_annotations"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "Only documents with available annotation data and a positive modernised token count are included.",
    )
    save_figure(
        fig,
        language_directory,
        "zero_annotation_documents_by_model",
    )


# ---------------------------------------------------------------------------
# Print a brief completion report
# ---------------------------------------------------------------------------

print(f"\nDocuments included in model comparison: {len(analysis_df)}")
print(f"Documents omitted because review data or token counts were unavailable: {len(documents_df) - len(analysis_df)}")
print(f"\nSummary table saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")

