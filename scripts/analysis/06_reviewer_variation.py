"""Describe reviewer exposure and the distribution of recurrent evidence.

This stage examines whether the reviewer feedback used for prompt review is
distributed across reviewer packets or concentrated in particular packets.
It does not estimate reviewer bias. Each document was reviewed once, so
reviewer differences may also reflect document, archive or model allocation.

The script produces four tables and two figures:

    analysis_outputs/06_reviewer_variation/
        REVIEWER_VARIATION_README.md
        tables/reviewer_summary.csv
        tables/reviewer_document_rates.csv
        tables/reviewer_error_category_profile.csv
        tables/recurrent_span_reviewer_distribution.csv
        figures/en/reviewer_annotation_rates.png and .svg
        figures/en/recurrent_span_reviewer_distribution.png and .svg
"""

import os
from pathlib import Path
import shutil
import unicodedata

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
ANNOTATION_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "annotation_analysis.csv"
RECURRENT_SPAN_PATH = (
    PROJECT_DIR
    / "analysis_outputs"
    / "05_recurrent_annotated_text"
    / "tables"
    / "recurrent_annotated_text.csv"
)
ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "06_reviewer_variation"
SECTION_README_PATH = SECTION_OUTPUT_DIR / "REVIEWER_VARIATION_README.md"

DISPLAY_SPAN_LIMIT = 10


# ---------------------------------------------------------------------------
# Define the visible chart wording
# ---------------------------------------------------------------------------

CHART_TEXT = {
    "en": {
        "rates": {
            "title": "How did document annotation rates vary across reviewers?",
            "description": (
                "The chart shows the reviewed documents assigned to each reviewer."
            ),
            "measure": (
                "Each point is one document; colour identifies the model. "
                "Differences may reflect document allocation as well as reviewer practice."
            ),
            "x_label": "Distinct annotations per 1,000 modernised tokens",
        },
        "spans": {
            "title": "Which reviewers recorded the recurrent spans highlighted in Stage 05?",
            "description": (
                "The chart shows how the ten most frequently annotated recurrent "
                "spans were distributed across reviewer packets."
            ),
            "measure": (
                "Cells contain distinct annotation counts. Blank cells indicate that "
                "the reviewer did not annotate that span."
            ),
        },
    }
}


README_TEXT = """# 06 · Reviewer-associated variation

## What this section is trying to show

This section checks whether the reviewer evidence used to identify possible
prompt improvements is distributed across reviewer packets or concentrated in
particular packets.

It does not measure reviewer bias. Each document was reviewed once. Differences
between reviewers may therefore reflect the documents, archives and models
assigned to them as well as differences in annotation practice.

## Measures

Document annotation rates are distinct included annotations per 1,000
modernised tokens. Documents without annotation data are excluded rather than
treated as having no annotations.

The recurrent-span analysis uses the spans identified in Stage 05. One
annotation is counted once for a span even when several error labels were
attached to it. Reviewer counts are reported separately by model in the CSV,
but the heatmap combines models because it is showing the distribution of the
review evidence rather than comparing model performance.

## Inputs

- `outputs/document_analysis.csv`: document, reviewer, model, archive and token
  information.
- `outputs/annotation_analysis.csv`: included reviewer annotations.
- `analysis_outputs/05_recurrent_annotated_text/tables/recurrent_annotated_text.csv`:
  recurrent spans and their Stage 05 ranks.

## Tables produced by this script

- `reviewer_summary.csv`: document and token exposure, annotation totals,
  pooled annotation rates, model allocation and archive coverage by reviewer.
- `reviewer_document_rates.csv`: the document-level data plotted in the first
  figure.
- `reviewer_error_category_profile.csv`: annotation and document counts for
  each reviewer and error category.
- `recurrent_span_reviewer_distribution.csv`: annotation counts for every
  recurrent span by reviewer and model.

## Figures produced by this script

- `reviewer_annotation_rates`: document annotation rates by reviewer and model.
- `recurrent_span_reviewer_distribution`: reviewer distribution of the ten
  most frequently annotated recurrent spans from Stage 05.

Both figures are saved as PNG and SVG.
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalise_annotated_text(value):
    """Apply the conservative normalisation used in Stage 05."""

    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFC", str(value))
    text = " ".join(text.split()).strip().casefold()
    return text if text else pd.NA


def join_unique(values):
    """Join unique non-empty values in their first observed order."""

    observed = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in observed:
            observed.append(text)
    return " | ".join(observed)


def reviewer_display_name(row):
    """Return a concise reviewer label with its packet identifier."""

    packet = row["reviewer_packet"]
    packet_text = str(int(packet)) if pd.notna(packet) and float(packet).is_integer() else str(packet)
    return f"{row['reviewer_name']} · packet {packet_text}"


# ---------------------------------------------------------------------------
# Check and load the analysis tables
# ---------------------------------------------------------------------------

check_required_files(
    [DOCUMENT_ANALYSIS_PATH, ANNOTATION_ANALYSIS_PATH, RECURRENT_SPAN_PATH],
    preceding_command = (
        "python scripts/analysis/05_recurrent_annotated_text.py"
    ),
)

documents_df = pd.read_csv(DOCUMENT_ANALYSIS_PATH)
annotations_df = pd.read_csv(ANNOTATION_ANALYSIS_PATH)
recurrent_df = pd.read_csv(RECURRENT_SPAN_PATH)

required_document_columns = {
    "filename_stem", "archive", "reviewer_packet", "reviewer_name", "model",
    "annotation_json_found", "n_modernised_tokens",
}
required_annotation_columns = {
    "filename_stem", "annotation_id", "annotated_text", "include_in_analysis",
    "effective_error_category",
}
required_recurrent_columns = {
    "normalised_annotated_text", "span_rank", "distinct_annotations",
}

for label, required, frame in (
    ("document_analysis.csv", required_document_columns, documents_df),
    ("annotation_analysis.csv", required_annotation_columns, annotations_df),
    ("recurrent_annotated_text.csv", required_recurrent_columns, recurrent_df),
):
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}.")

if documents_df["filename_stem"].duplicated().any():
    raise ValueError("document_analysis.csv must contain one row per document.")

documents_df["annotation_json_found"] = coerce_boolean(
    documents_df["annotation_json_found"]
)
annotations_df["include_in_analysis"] = coerce_boolean(
    annotations_df["include_in_analysis"]
)
documents_df["n_modernised_tokens"] = pd.to_numeric(
    documents_df["n_modernised_tokens"], errors = "coerce"
)
documents_df["reviewer_packet"] = pd.to_numeric(
    documents_df["reviewer_packet"], errors = "coerce"
)

reviewed_documents_df = documents_df.loc[
    documents_df["annotation_json_found"]
].copy()
invalid_tokens = (
    reviewed_documents_df["n_modernised_tokens"].isna()
    | reviewed_documents_df["n_modernised_tokens"].le(0)
)
if invalid_tokens.any():
    raise ValueError(
        "A reviewed document lacks a positive modernised-token count."
    )

reviewer_packet_counts = (
    reviewed_documents_df.groupby("reviewer_name")["reviewer_packet"].nunique()
)
if reviewer_packet_counts.gt(1).any():
    raise ValueError(
        "A reviewer is associated with more than one reviewer packet. Review "
        "the assignment table before interpreting reviewer variation."
    )

models = sorted(str(value) for value in reviewed_documents_df["model"].dropna().unique())
model_colours = model_colour_map(models)


# ---------------------------------------------------------------------------
# Construct document-level annotation rates
# ---------------------------------------------------------------------------

included_df = annotations_df.loc[annotations_df["include_in_analysis"]].copy()
annotation_identity = ["filename_stem", "annotation_id"]
distinct_annotations_df = included_df.drop_duplicates(annotation_identity)
document_annotation_counts = (
    distinct_annotations_df.groupby("filename_stem")["annotation_id"]
    .nunique()
    .rename("n_included_annotations")
)

reviewer_documents_df = reviewed_documents_df.merge(
    document_annotation_counts,
    left_on = "filename_stem",
    right_index = True,
    how = "left",
    validate = "one_to_one",
)
reviewer_documents_df["n_included_annotations"] = (
    reviewer_documents_df["n_included_annotations"].fillna(0).astype(int)
)
reviewer_documents_df["annotations_per_1000_tokens"] = (
    1000
    * reviewer_documents_df["n_included_annotations"]
    / reviewer_documents_df["n_modernised_tokens"]
)

reviewer_order_df = (
    reviewer_documents_df[["reviewer_name", "reviewer_packet"]]
    .drop_duplicates()
    .sort_values(["reviewer_packet", "reviewer_name"])
    .reset_index(drop = True)
)
reviewer_order_df["reviewer_display"] = reviewer_order_df.apply(
    reviewer_display_name, axis = 1
)
reviewer_order = reviewer_order_df["reviewer_name"].tolist()
reviewer_display = dict(zip(
    reviewer_order_df["reviewer_name"], reviewer_order_df["reviewer_display"]
))


# ---------------------------------------------------------------------------
# Summarise reviewer exposure and category use
# ---------------------------------------------------------------------------

reviewer_summary_df = (
    reviewer_documents_df.groupby(
        ["reviewer_name", "reviewer_packet"], dropna = False
    )
    .agg(
        reviewed_documents = ("filename_stem", "nunique"),
        modernised_tokens = ("n_modernised_tokens", "sum"),
        distinct_annotations = ("n_included_annotations", "sum"),
        documents_without_annotations = (
            "n_included_annotations", lambda values: int(pd.Series(values).eq(0).sum())
        ),
        represented_archives = ("archive", "nunique"),
        represented_models = ("model", "nunique"),
        archives = ("archive", join_unique),
        models = ("model", join_unique),
    )
    .reset_index()
)
reviewer_summary_df["pooled_annotations_per_1000_tokens"] = (
    1000
    * reviewer_summary_df["distinct_annotations"]
    / reviewer_summary_df["modernised_tokens"]
)

model_exposure_df = (
    reviewer_documents_df.groupby(
        ["reviewer_name", "reviewer_packet", "model"], dropna = False
    )
    .agg(
        model_documents = ("filename_stem", "nunique"),
        model_tokens = ("n_modernised_tokens", "sum"),
        model_annotations = ("n_included_annotations", "sum"),
        model_archives = ("archive", "nunique"),
    )
    .reset_index()
)
model_exposure_df["model_pooled_annotations_per_1000_tokens"] = (
    1000 * model_exposure_df["model_annotations"] / model_exposure_df["model_tokens"]
)
for measure in (
    "model_documents", "model_tokens", "model_annotations", "model_archives",
    "model_pooled_annotations_per_1000_tokens",
):
    wide = model_exposure_df.pivot(
        index = ["reviewer_name", "reviewer_packet"],
        columns = "model",
        values = measure,
    )
    wide.columns = [f"{measure}__{column}" for column in wide.columns]
    reviewer_summary_df = reviewer_summary_df.merge(
        wide.reset_index(),
        on = ["reviewer_name", "reviewer_packet"],
        how = "left",
        validate = "one_to_one",
    )

category_identity = annotation_identity + ["effective_error_category"]
category_annotations_df = included_df.drop_duplicates(category_identity).merge(
    documents_df[["filename_stem", "reviewer_name", "reviewer_packet", "model"]],
    on = "filename_stem",
    how = "left",
    validate = "many_to_one",
)
reviewer_category_df = (
    category_annotations_df.groupby(
        ["reviewer_name", "reviewer_packet", "model", "effective_error_category"],
        dropna = False,
    )
    .agg(
        distinct_annotations = ("annotation_id", "nunique"),
        affected_documents = ("filename_stem", "nunique"),
    )
    .reset_index()
)
reviewer_category_df["reviewer_model_category_share"] = (
    reviewer_category_df["distinct_annotations"]
    / reviewer_category_df.groupby(
        ["reviewer_name", "model"], dropna = False
    )["distinct_annotations"].transform("sum")
)


# ---------------------------------------------------------------------------
# Count recurrent spans by reviewer and model
# ---------------------------------------------------------------------------

included_df = included_df.merge(
    documents_df[[
        "filename_stem", "reviewer_name", "reviewer_packet", "model", "archive"
    ]],
    on = "filename_stem",
    how = "left",
    validate = "many_to_one",
)
included_df["normalised_annotated_text"] = included_df["annotated_text"].map(
    normalise_annotated_text
)
span_identity = annotation_identity + ["normalised_annotated_text"]
span_annotations_df = included_df.drop_duplicates(span_identity)
span_reviewer_df = (
    span_annotations_df.merge(
        recurrent_df[[
            "normalised_annotated_text", "span_rank", "distinct_annotations"
        ]],
        on = "normalised_annotated_text",
        how = "inner",
        validate = "many_to_one",
    )
    .groupby(
        [
            "span_rank", "normalised_annotated_text", "reviewer_name",
            "reviewer_packet", "model",
        ],
        dropna = False,
    )
    .agg(
        distinct_annotations = ("annotation_id", "nunique"),
        affected_documents = ("filename_stem", "nunique"),
        affected_archives = ("archive", "nunique"),
    )
    .reset_index()
    .sort_values(["span_rank", "reviewer_packet", "model"])
)


# ---------------------------------------------------------------------------
# Replace this stage's outputs and save the tables
# ---------------------------------------------------------------------------

if (SECTION_OUTPUT_DIR.name != "06_reviewer_variation"
        or SECTION_OUTPUT_DIR.parent != ANALYSIS_OUTPUT_DIR):
    raise RuntimeError(f"Unsafe output directory: {SECTION_OUTPUT_DIR}")
if SECTION_OUTPUT_DIR.exists():
    shutil.rmtree(SECTION_OUTPUT_DIR)
tables_directory, figures_directory = create_output_folders(SECTION_OUTPUT_DIR)

save_table(reviewer_summary_df, tables_directory / "reviewer_summary.csv")
save_table(
    reviewer_documents_df.sort_values(["reviewer_packet", "filename_stem"]),
    tables_directory / "reviewer_document_rates.csv",
)
save_table(
    reviewer_category_df,
    tables_directory / "reviewer_error_category_profile.csv",
)
save_table(
    span_reviewer_df,
    tables_directory / "recurrent_span_reviewer_distribution.csv",
)


# ---------------------------------------------------------------------------
# Plot document rates by reviewer
# ---------------------------------------------------------------------------

apply_plot_style()
for language in OUTPUT_LANGUAGES:
    text = CHART_TEXT.get(language, CHART_TEXT["en"])
    language_directory = figures_directory / language
    language_directory.mkdir(parents = True, exist_ok = True)

    fig, ax = plt.subplots(figsize = (17.5, 10.5))
    y_lookup = {name: position for position, name in enumerate(reviewer_order)}
    rng = np.random.default_rng(42)
    for model_index, model in enumerate(models):
        subset = reviewer_documents_df.loc[
            reviewer_documents_df["model"].astype(str).eq(model)
        ].copy()
        y_values = subset["reviewer_name"].map(y_lookup).astype(float).to_numpy()
        y_values += rng.uniform(-0.12, 0.12, size = len(subset))
        ax.scatter(
            subset["annotations_per_1000_tokens"],
            y_values,
            s = 48,
            color = model_colours[model],
            alpha = 0.82,
            edgecolor = "white",
            linewidth = 0.5,
            label = model,
            zorder = 3,
        )
    ax.set_yticks(range(len(reviewer_order)))
    ax.set_yticklabels([reviewer_display[name] for name in reviewer_order])
    ax.invert_yaxis()
    ax.set_xlabel(text["rates"]["x_label"])
    ax.grid(axis = "x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon = False, loc = "lower center", bbox_to_anchor = (0.5, -0.16),
              ncol = max(1, len(models)))
    add_chart_header(
        fig,
        title = text["rates"]["title"],
        description = text["rates"]["description"],
        measure = text["rates"]["measure"],
    )
    add_figure_note(
        fig,
        "Documents were not reviewed by multiple reviewers; differences cannot "
        "be attributed to reviewer practice alone.",
    )
    fig.subplots_adjust(left = 0.20, right = 0.96, bottom = 0.19, top = 0.71)
    save_figure(fig, language_directory, "reviewer_annotation_rates")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot reviewer coverage for the ten Stage 05 spans
# ---------------------------------------------------------------------------

    top_spans = (
        recurrent_df.sort_values("span_rank").head(DISPLAY_SPAN_LIMIT)[
            ["normalised_annotated_text", "span_rank"]
        ]
    )
    top_span_names = top_spans["normalised_annotated_text"].tolist()
    heatmap_counts = (
        span_reviewer_df.loc[
            span_reviewer_df["normalised_annotated_text"].isin(top_span_names)
        ]
        .groupby(["normalised_annotated_text", "reviewer_name"])[
            "distinct_annotations"
        ]
        .sum()
        .unstack(fill_value = 0)
        .reindex(index = top_span_names, columns = reviewer_order, fill_value = 0)
    )

    fig, ax = plt.subplots(figsize = (17.5, 10.5))
    heatmap_colour = LinearSegmentedColormap.from_list(
        "reviewer_evidence",
        [COLOURS["panel"], COLOURS["gold"], COLOURS["model_terracotta"]],
    )
    image = ax.imshow(heatmap_counts.to_numpy(), aspect = "auto", cmap = heatmap_colour)
    ax.set_xticks(range(len(reviewer_order)))
    ax.set_xticklabels(
        [reviewer_display[name] for name in reviewer_order],
        rotation = 38,
        ha = "right",
    )
    ax.set_yticks(range(len(top_span_names)))
    ax.set_yticklabels(top_span_names, fontstyle = "italic", fontweight = "bold")
    for row_index in range(heatmap_counts.shape[0]):
        for column_index in range(heatmap_counts.shape[1]):
            value = int(heatmap_counts.iat[row_index, column_index])
            if value:
                ax.text(
                    column_index,
                    row_index,
                    str(value),
                    ha = "center",
                    va = "center",
                    fontsize = 8,
                    color = COLOURS["text"],
                )
    ax.tick_params(axis = "both", length = 0)
    ax.spines[:].set_visible(False)
    colourbar = fig.colorbar(image, ax = ax, fraction = 0.025, pad = 0.02)
    colourbar.set_label("Distinct reviewer annotations")
    add_chart_header(
        fig,
        title = text["spans"]["title"],
        description = text["spans"]["description"],
        measure = text["spans"]["measure"],
    )
    add_figure_note(
        fig,
        "Models are combined in this display. Model-specific counts remain in the CSV.",
    )
    fig.subplots_adjust(left = 0.12, right = 0.94, bottom = 0.23, top = 0.71)
    save_figure(fig, language_directory, "recurrent_span_reviewer_distribution")
    plt.close(fig)


SECTION_README_PATH.write_text(README_TEXT, encoding = "utf-8")

print(f"Reviewed documents: {len(reviewer_documents_df):,}")
print(f"Reviewers: {reviewer_documents_df['reviewer_name'].nunique():,}")
print(f"Recurrent spans represented: {span_reviewer_df['normalised_annotated_text'].nunique():,}")
print(f"Tables saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
