"""Compare the two modernisation models using human-review annotations.

This section describes how frequently reviewers marked apparent problems in
documents modernised by each model. It does not attempt to prove that the model
itself caused the observed difference.

The primary measure is the number of distinct included annotations per 1,000
modernised tokens. A single annotation may have more than one category or
field assignment, but it is counted once here as one reviewer-marked error.

Only documents with available annotation data and a positive modernised token
count are used when calculating document-level rates. A missing annotation JSON
is not interpreted as a document with zero errors.

The script deliberately produces a small set of outputs:

    analysis_outputs/01_model_comparison/MODEL_COMPARISON_README.md
    analysis_outputs/01_model_comparison/tables/model_observed_summary.csv
    analysis_outputs/01_model_comparison/figures/en/
        annotation_rate_by_model.png and .svg

Statistical adjustment for archive and reviewer is deliberately postponed
until the annotation patterns and possible reviewer effects have been examined.
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
            "title": "How did observed annotation rates differ by model?",
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
    }
}


README_TEXT = """# 01 · Descriptive model overview

## What this section is trying to show

This section shows what reviewers recorded for documents produced by each
modernisation model.

The main measure is **distinct included annotations per 1,000 modernised
tokens**. Each reviewer-marked annotation is counted once even when it has more
than one category or field assignment.

Documents with unavailable annotation data are excluded from the comparison;
they are not treated as documents with zero errors.

## Input

- `outputs/document_analysis.csv`: one row per sample document.

## Table produced by this script

- `model_observed_summary.csv`: reviewed documents, text length, annotation
  totals, annotation rates and zero-annotation documents for each model.

## Figures produced by this script

- `annotation_rate_by_model`: document-level annotation rates for each model.

Each figure is saved as both PNG and SVG.

## Interpretation

This is a descriptive comparison of human-review outcomes. A higher annotation
rate means that reviewers marked more apparent problems; it does not yet prove
that the model was worse. Later analysis will examine error types, reviewer
behaviour, archive composition and annotation decisions before statistical
modelling is undertaken.
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
    tables_directory / "model_observed_summary.csv",
)


# ---------------------------------------------------------------------------
# Generate one focused descriptive figure
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
    axis_labels = [
        f"{fill(model, 24)}\n(n={len(rates)} documents)"
        for model, rates in zip(models, rate_groups)
    ]

    fig, ax = plt.subplots(figsize=(10, 7.4))
    fig.subplots_adjust(top=0.72, bottom=0.26, left=0.14, right=0.92)

    boxplot = ax.boxplot(
        rate_groups,
        tick_labels=axis_labels,
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
    ax.tick_params(axis="x", labelrotation=0)
    add_chart_header(
        fig,
        **{
            key: text["annotation_rate"][key]
            for key in ("title", "description", "measure")
        },
    )
    add_figure_note(
        fig,
        "Each point represents one document. Rates are not adjusted for archive or reviewer.",
    )
    save_figure(fig, language_directory, "annotation_rate_by_model")

# ---------------------------------------------------------------------------
# Print a brief completion report
# ---------------------------------------------------------------------------

print(f"\nDocuments included in model comparison: {len(analysis_df)}")
print(f"Documents omitted because review data or token counts were unavailable: {len(documents_df) - len(analysis_df)}")
print(f"\nSummary table saved to: {tables_directory}")
print(f"Figure saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
