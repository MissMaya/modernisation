"""
Examine annotation count in relation to document length.

For every reviewed document, report document length, annotation count,
annotation rate, model, archive, reviewer, and the most frequently assigned
error category and category-sub-rule.

The script also identifies documents with an annotation count or annotation rate
that exceeds the upper outlier threshold for the model that produced it. The threshold is
Q3 + 1.5 x IQR and is calculated separately for each measure and model.

Outputs:

    analysis_outputs/03_document_diagnostics/
        DOCUMENT_DIAGNOSTICS_README.md
        tables/document_diagnostics.csv
        tables/upper_outlier_documents.csv
        figures/en/document_length_and_annotation_burden.png and .svg
        figures/en/upper_outlier_document_error_profiles.png and .svg
"""

# ---------------------------------------------------------------------------
# Import the required modules
# ---------------------------------------------------------------------------

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

FAMILY_SUFFIX_PATTERN = re.compile(r"_duplicated_\d+$", flags = re.IGNORECASE)


# ---------------------------------------------------------------------------
# Specify wording for charts
# ---------------------------------------------------------------------------

CHART_TEXT = {
    "en": {
        "burden": {
            "title": "How did annotation count vary with document length?",
            "description": (
                "Annotation count is plotted against document length for all "
                "reviewed documents."
            ),
            "measure": (
                "Numbered documents are those that exceed the model-specific upper outlier  "
                "threshold for annotation count and/or annotation rate."
            ),
            "x_label": "Document length (modernised tokens)",
            "y_label": "Distinct reviewer annotations",
        },
        "profiles": {
            "title": (
                "Which error categories were assigned to the numbered documents on the "
                "scatterplot?"
            ),
            "description": (
                "The chart shows the error-category composition of reviewer "
                "feedback for the numbered documents in the preceding "
                "scatter plot."
            ),
            "measure": (
                "Distinct annotations by error category. An annotation with "
                "several sub-rules contributes once to each category."
            ),
            "x_label": "Distinct annotations by error category",
        },
    }
}


README_TEXT = """# 03 · Document diagnostics

## Aim of analysis

To examine annotation count in relation to document length and identify
documents with unusually high annotation counts or annotation rates.

It anlayses both documents with many annotations and documents with a
high annotation rate. This matters because a short document can have a high
rate per 1,000 tokens even when its absolute annotation count is moderate.

## Upper outlier thresholds

`upper_outlier_documents.csv` contains documents exceeding the upper outlier
threshold for either:

- distinct included annotation count; or
- distinct included annotations per 1,000 modernised tokens.

The upper outlier threshold is Q3 + 1.5 x IQR, where Q3 is the 75th percentile
and IQR is the range between the 25th and 75th percentiles. It is calculated
separately for each measure and model.

These thresholds identify documents that are unusual relative to other
documents produced by the same model. 

The numbered documents are ordered to try to flag the most useful to look at for 
prompt refinement. Documents exceeding both count and rate thresholds come first, 
followed by count-only and then rate-only outliers.
Within each group, documents are ordered by annotation count and then annotation
rate, both descending. This puts evidence supported by both measures first and
places rate-only cases, which can be affected by short documents, last. The
numbers indicate review order, not model performance.

`document_diagnostics.csv` contains one row for every document plotted in the
scatter plot. It reports the model, document length, annotation count,
annotation rate, model-specific thresholds and whether each document exceeds
the count threshold, the rate threshold, both or neither.

## Inputs

- `outputs/document_analysis.csv`: one row per sample document.
- `outputs/annotation_analysis.csv`: included category and sub-rule assignments.

## Tables produced by this script

- `document_diagnostics.csv`: all reviewed documents, with annotation burden,
  document context and the most frequently assigned error category and
  category-sub-rule. Repeated-document family fields are retained for later
  analysis.
- `upper_outlier_documents.csv`: documents exceeding either upper outlier
  threshold, with the criterion recorded.

## Figures produced by this script

- `document_length_and_annotation_burden`: annotation counts against document
  length, with numbered upper outliers identified in an adjacent key.
- `upper_outlier_document_error_profiles`: category composition for the same
  numbered documents.
"""


# ---------------------------------------------------------------------------
# Check and load the two analysis tables
# ---------------------------------------------------------------------------

check_required_files(
    [DOCUMENT_ANALYSIS_PATH, ANNOTATION_ANALYSIS_PATH],
    preceding_command = "python scripts/construct_tables.py",
)

documents_df = pd.read_csv(
    DOCUMENT_ANALYSIS_PATH,
    dtype = {"filename_stem": "string"},
)
annotations_df = pd.read_csv(
    ANNOTATION_ANALYSIS_PATH,
    dtype = {
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
    documents_df[column] = pd.to_numeric(documents_df[column], errors = "coerce")


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
    on = "filename_stem",
    how = "inner",
    validate = "many_to_one",
)

if included_annotations_df["effective_error_category"].isna().any():
    raise ValueError(
        "At least one included annotation assignment has no effective error "
        "category. Check the Stage 6 resolution output."
    )


# ---------------------------------------------------------------------------
# Retain repeated-document family identifiers for later analysis
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
# Summarise the most frequently assigned category and sub-rule by document
# ---------------------------------------------------------------------------

def join_joint_modes(values):
    """Return all equally frequent non-missing labels in alphabetical order."""

    counts = pd.Series(values).dropna().astype(str).value_counts()
    if counts.empty:
        return pd.NA
    modes = sorted(counts.index[counts.eq(counts.max())])
    return " | ".join(modes)


# Count each annotation once per category. Two sub-rules from the same category
# still add only one annotation to that category total.
category_annotations_df = included_annotations_df.drop_duplicates(
    subset = ["filename_stem", "annotation_id", "effective_error_category"]
)
most_frequent_category_df = (
    category_annotations_df.groupby("filename_stem")["effective_error_category"]
    .agg(join_joint_modes)
    .rename("most_frequent_error_category")
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
    subset = [
        "filename_stem",
        "annotation_id",
        "effective_error_category",
        "subrule_for_summary",
    ]
)
most_frequent_subrule_df = (
    subrule_annotations_df.groupby("filename_stem")["category_and_subrule"]
    .agg(join_joint_modes)
    .rename("most_frequent_category_subrule")
    .reset_index()
)

analysis_df = analysis_df.merge(
    most_frequent_category_df,
    on = "filename_stem",
    how = "left",
    validate = "one_to_one",
)
analysis_df = analysis_df.merge(
    most_frequent_subrule_df,
    on = "filename_stem",
    how = "left",
    validate = "one_to_one",
)

zero_annotation_document = analysis_df["n_included_annotations"].eq(0)
analysis_df.loc[zero_annotation_document, "most_frequent_error_category"] = (
    "No included annotations"
)
analysis_df.loc[zero_annotation_document, "most_frequent_category_subrule"] = (
    "No included annotations"
)


# ---------------------------------------------------------------------------
# Identify upper outliers within each model group
# ---------------------------------------------------------------------------

def upper_outlier_threshold(values):
    """Return Q3 + 1.5 × IQR for one measure and model group."""

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return q3 + 1.5 * (q3 - q1)


count_thresholds = analysis_df.groupby("model")[
    "n_included_annotations"
].transform(upper_outlier_threshold)
rate_thresholds = analysis_df.groupby("model")[
    "included_annotations_per_1000_tokens"
].transform(upper_outlier_threshold)

analysis_df["annotation_count_upper_outlier_threshold"] = count_thresholds
analysis_df["annotation_rate_upper_outlier_threshold"] = rate_thresholds
analysis_df["unusually_high_annotation_count"] = analysis_df[
    "n_included_annotations"
].gt(count_thresholds)
analysis_df["unusually_high_annotation_rate"] = analysis_df[
    "included_annotations_per_1000_tokens"
].gt(rate_thresholds)
analysis_df["above_either_upper_outlier_threshold"] = (
    analysis_df["unusually_high_annotation_count"]
    | analysis_df["unusually_high_annotation_rate"]
)


def describe_outlier_reason(row):
    """State which upper outlier threshold a document exceeds."""

    if (
        row["unusually_high_annotation_count"]
        and row["unusually_high_annotation_rate"]
    ):
        return "count and rate"
    if row["unusually_high_annotation_count"]:
        return "count"
    if row["unusually_high_annotation_rate"]:
        return "rate"
    return "neither"


analysis_df["upper_outlier_reason"] = analysis_df.apply(
    describe_outlier_reason,
    axis = 1,
)

# Retain the distribution statistics used to audit each model-specific
# threshold in the completion report.
threshold_summary_records = []
for model, model_df in analysis_df.groupby("model", sort = True):
    for measure, column in (
        ("annotation count", "n_included_annotations"),
        ("annotation rate", "included_annotations_per_1000_tokens"),
    ):
        values = model_df[column]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        threshold_summary_records.append(
            {
                "model": model,
                "measure": measure,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "threshold": q3 + 1.5 * iqr,
                "maximum": values.max(),
            }
        )
threshold_summary_df = pd.DataFrame(threshold_summary_records)


# ---------------------------------------------------------------------------
# Replace this section's earlier outputs on a re-run
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
SECTION_README_PATH.write_text(README_TEXT, encoding = "utf-8")


# ---------------------------------------------------------------------------
# Save the complete diagnostic table and upper-outlier table
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
    "most_frequent_error_category",
    "most_frequent_category_subrule",
    "document_family",
    "reviewed_documents_in_family",
    "repeated_document_family",
    "annotation_count_upper_outlier_threshold",
    "annotation_rate_upper_outlier_threshold",
    "unusually_high_annotation_count",
    "unusually_high_annotation_rate",
    "above_either_upper_outlier_threshold",
    "upper_outlier_reason",
]

document_diagnostics_df = (
    analysis_df[diagnostic_columns]
    .sort_values(
        ["model", "filename_stem"],
        ascending = [True, True],
    )
    .reset_index(drop = True)
)
upper_outlier_documents_df = (
    document_diagnostics_df.loc[
        document_diagnostics_df["above_either_upper_outlier_threshold"]
    ]
    .copy()
)

# Order the selected documents for prompt review. Evidence from both measures
# comes first, followed by count-only and then rate-only evidence. Within each
# group, larger annotation counts come first; annotation rate and filename give
# deterministic tie-breaks. 
review_group_order = {
    "count and rate": 1,
    "count": 2,
    "rate": 3,
}
upper_outlier_documents_df["prompt_review_group_order"] = (
    upper_outlier_documents_df["upper_outlier_reason"].map(review_group_order)
)
if upper_outlier_documents_df["prompt_review_group_order"].isna().any():
    raise RuntimeError(
        "Every upper-outlier document must have a prompt-review group."
    )
upper_outlier_documents_df = upper_outlier_documents_df.sort_values(
    [
        "prompt_review_group_order",
        "n_included_annotations",
        "included_annotations_per_1000_tokens",
        "filename_stem",
    ],
    ascending = [True, False, False, True],
).reset_index(drop = True)

# Give every upper-outlier document a short review-order number. The same number
# appears on the scatter plot, in its key, in the bar chart and in the CSV.
upper_outlier_documents_df.insert(
    0,
    "document_number",
    range(1, len(upper_outlier_documents_df) + 1),
)

# Confirm that the numbered table contains every document exceeding either
# model-specific threshold and no document that meets neither criterion.
expected_upper_outlier_stems = set(
    analysis_df.loc[
        analysis_df["n_included_annotations"].gt(
            analysis_df["annotation_count_upper_outlier_threshold"]
        )
        | analysis_df["included_annotations_per_1000_tokens"].gt(
            analysis_df["annotation_rate_upper_outlier_threshold"]
        ),
        "filename_stem",
    ]
)
observed_upper_outlier_stems = set(
    upper_outlier_documents_df["filename_stem"]
)
if observed_upper_outlier_stems != expected_upper_outlier_stems:
    raise RuntimeError(
        "The numbered document table does not match the model-specific upper "
        "outlier criteria."
    )
if upper_outlier_documents_df["document_number"].duplicated().any():
    raise RuntimeError("Document numbers must be unique.")
if upper_outlier_documents_df["document_number"].tolist() != list(
    range(1, len(upper_outlier_documents_df) + 1)
):
    raise RuntimeError("Document numbers must follow prompt-review order.")

save_table(
    document_diagnostics_df,
    tables_directory / "document_diagnostics.csv",
)
save_table(
    upper_outlier_documents_df,
    tables_directory / "upper_outlier_documents.csv",
)


# ---------------------------------------------------------------------------
# Prepare category counts for the upper-outlier profile figure
# ---------------------------------------------------------------------------

upper_outlier_stems = set(upper_outlier_documents_df["filename_stem"])
upper_outlier_categories_df = category_annotations_df.loc[
    category_annotations_df["filename_stem"].isin(upper_outlier_stems)
]
category_counts_df = (
    upper_outlier_categories_df.groupby(
        ["filename_stem", "effective_error_category"]
    )
    .size()
    .unstack(fill_value = 0)
)

category_order = (
    category_counts_df.sum(axis = 0).sort_values(ascending = False).index.tolist()
)
# Matplotlib by default draws the first horizontal bar at the bottom.
# This reverses the sequence here so document 1 appears at the top of the chart.
upper_outlier_plot_order = upper_outlier_documents_df.iloc[::-1][
    "filename_stem"
].tolist()
category_counts_df = category_counts_df.reindex(
    index = upper_outlier_plot_order,
    columns = category_order,
    fill_value = 0,
)

# The bar chart shows the numbered documents in the scatter.
if set(category_counts_df.index) != observed_upper_outlier_stems:
    raise RuntimeError(
        "The error-category profile does not contain the same documents as "
        "the numbered scatter plot."
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

    # Figure 1: compare annotation count with document length. Numbered markers
    # identify documents above either upper outlier threshold.
    scatter_figure_height = max(
        8.5,
        4.8 + 0.18 * len(upper_outlier_documents_df),
    )
    fig, ax = plt.subplots(figsize = (15, scatter_figure_height))
    fig.subplots_adjust(top = 0.72, bottom = 0.14, left = 0.09, right = 0.69)

    for model, model_df in analysis_df.groupby("model", sort = True):
        ax.scatter(
            model_df["n_modernised_tokens"],
            model_df["n_included_annotations"],
            label = str(model),
            color = model_colours[str(model)],
            s = 42,
            alpha = 0.70,
            edgecolor = COLOURS["panel"],
            linewidth = 0.5,
            zorder = 2,
        )

    upper_outlier_plot_df = upper_outlier_documents_df.copy()
    ax.scatter(
        upper_outlier_plot_df["n_modernised_tokens"],
        upper_outlier_plot_df["n_included_annotations"],
        facecolors = [
            model_colours[str(model)] for model in upper_outlier_plot_df["model"]
        ],
        edgecolors = COLOURS["text"],
        s = 150,
        linewidth = 1.0,
        zorder = 3,
    )

    # Numbers remain legible even where several documents are close
    # together. Full filenames are retained in the key and output table.
    for _, row in upper_outlier_plot_df.iterrows():
        ax.text(
            row["n_modernised_tokens"],
            row["n_included_annotations"],
            str(row["document_number"]),
            fontsize = 6.2,
            fontweight = "bold",
            color = COLOURS["panel"],
            ha = "center",
            va = "center",
            zorder = 4,
        )

    ax.set_xlabel(text["burden"]["x_label"])
    ax.set_ylabel(text["burden"]["y_label"])
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon = False, loc = "upper left")

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
        "Documents with unusually high counts and/or rates within each model",
        fontsize = 9,
        fontweight = "bold",
        color = COLOURS["text"],
        va = "bottom",
        transform = key_ax.transAxes,
    )
    key_ax.text(
        0.045,
        0.995,
        "Threshold criterion in brackets",
        fontsize = 6.4,
        color = COLOURS["muted_text"],
        va = "top",
        transform = key_ax.transAxes,
    )
    key_line_height = 0.94 / max(len(upper_outlier_plot_df), 1)
    for key_position, (_, row) in enumerate(upper_outlier_plot_df.iterrows()):
        key_y_position = 0.94 - key_position * key_line_height

        # Circle with model colour beside each filename so the key can be read
        # directly without tracing the numbered point back to the scatter.
        key_ax.text(
            0.015,
            key_y_position,
            "●",
            fontsize = 7,
            color = model_colours[str(row["model"])],
            ha = "center",
            va = "top",
            transform = key_ax.transAxes,
        )
        key_ax.text(
            0.045,
            key_y_position,
            fill(
                f"{row['document_number']}. {row['filename_stem']} "
                f"({row['upper_outlier_reason']})",
                width = 43,
            ),
            fontsize = 6.4,
            color = COLOURS["muted_text"],
            va = "top",
            linespacing = 0.95,
            transform = key_ax.transAxes,
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
        "Upper outlier threshold = Q3 + 1.5 x IQR, where Q3 is the 75th percentile and IQR is the range between the 25th and 75th percentiles. Calculated separately for each model.",
    )
    save_figure(fig, language_directory, "document_length_and_annotation_burden")

    # Figure 2: show the error categories assigned to each numbered document.
    # Each label keeps the document, reviewer, archive and model on one line.
    figure_height = max(8.0, 4.8 + 0.32 * len(category_counts_df))
    fig, ax = plt.subplots(figsize = (17.5, figure_height))

    # The wide figure provides enough room for one-line metadata labels without
    # allowing the label column to consume nearly half of the canvas.
    fig.subplots_adjust(top = 0.72, bottom = 0.14, left = 0.43, right = 0.96)

    left_values = pd.Series(0, index = category_counts_df.index, dtype = float)
    for category_position, category in enumerate(category_order):
        values = category_counts_df[category]
        ax.barh(
            category_counts_df.index,
            values,
            left = left_values,
            label = category,
            color = ERROR_CATEGORY_COLOURS[
                category_position % len(ERROR_CATEGORY_COLOURS)
            ],
            edgecolor = "none",
        )
        left_values = left_values + values

    upper_outlier_metadata = upper_outlier_documents_df.set_index(
        "filename_stem"
    )
    y_labels = []
    for filename_stem in category_counts_df.index:
        row = upper_outlier_metadata.loc[filename_stem]
        y_labels.append(
            f"{row['document_number']}. {filename_stem} · "
            f"{row['reviewer_name']} · {row['archive']} · {row['model']}"
        )

    ax.set_yticks(range(len(category_counts_df.index)))
    ax.set_yticklabels(y_labels, fontsize = 7.5)

    # Explain the four pieces of metadata concatenated in each y-axis label.
    # This heading sits above the label column rather than inside the data area.
    fig.text(
        0.425,
        0.735,
        "document · reviewer · archive · model",
        fontsize = 8.5,
        fontweight = "bold",
        color = COLOURS["muted_text"],
        ha = "right",
        va = "bottom",
    )
    ax.set_xlabel(text["profiles"]["x_label"])
    ax.grid(axis = "x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis = "y", length = 0)
    ax.legend(
        frameon = False,
        loc = "lower center",
        bbox_to_anchor = (0.5, 0.045),
        bbox_transform = fig.transFigure,
        ncol = 3,
        fontsize = 8,
    )
    add_chart_header(
        fig,
        **{
            key: text["profiles"][key]
            for key in ("title", "description", "measure")
        },
    )
    save_figure(
        fig,
        language_directory,
        "upper_outlier_document_error_profiles",
    )


# ---------------------------------------------------------------------------
# Print a brief completion report
# ---------------------------------------------------------------------------

print(f"\nReviewed documents included: {len(analysis_df)}")
print(
    "Documents above either upper outlier threshold: "
    f"{len(upper_outlier_documents_df)}"
)
print("\nModel-specific upper outlier calculations:")
for _, row in threshold_summary_df.iterrows():
    unit = " per 1,000 tokens" if row["measure"] == "annotation rate" else ""
    print(
        f"- {row['model']} · {row['measure']}: "
        f"Q1 = {row['q1']:.2f}; "
        f"Q3 = {row['q3']:.2f}; "
        f"IQR = {row['iqr']:.2f}; "
        f"threshold = {row['threshold']:.2f}; "
        f"maximum = {row['maximum']:.2f}{unit}"
    )
print(f"\nTables saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
