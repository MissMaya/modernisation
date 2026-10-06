"""
Identify repeatedly annotated words and phrases.

This script classifes an annotated text span as recurrent when 
the same conservatively normalised text occurs in at least two reviewed 
documents and in at least two reviewer packets. 

The normalisation process applies Unicode NFC, collapses whitespace, 
trims and case-folds. It does not remove accents or punctuation, stem, 
lemmatise, split phrases or combine spelling variants. 

A single annotation is identified by document, annotation ID and normalised text. 
Repeated rows created by multiple error labels therefore do not inflate 
the annotation count.

For each span, the script reports every reviewer-assigned error label. It also
reports the most frequently assigned label, its observed coverage and the 95%
Wilson lower confidence bound for that coverage. These values just describe label
consistency; they do not prove that reviewers were correct or that either model
failed.

Outputs are three CSVs, a 10-span summary figure and a complete visual dictionary
of all repeatedly annotated texts. A ten-span limit is used to construct a one page 
chart to act as a summary. Every recurrent span remains in the tables and dictionary.
"""

import math
import os
from pathlib import Path
import re
import shutil
import textwrap
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import required modules
# ---------------------------------------------------------------------------
from analysis_utils import (
    COLOURS, OUTPUT_LANGUAGES, add_chart_header,
    apply_plot_style, check_required_files, coerce_boolean,
    create_output_folders, model_colour_map, save_figure, save_table,
)


# ---------------------------------------------------------------------------
# Set up paths and rules for the analysis
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(os.environ.get(
    "MODERNISATION_PROJECT_DIR", "/workspaces/modernisation"
))
DOCUMENT_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "document_analysis.csv"
ANNOTATION_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "annotation_analysis.csv"
ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "05_recurrent_annotated_text"
SECTION_README_PATH = SECTION_OUTPUT_DIR / "RECURRENT_ANNOTATED_TEXT_README.md"

MIN_DOCUMENTS = 2
MIN_REVIEWER_PACKETS = 2
SUMMARY_SPAN_LIMIT = 10
DICTIONARY_ROWS_PER_PAGE = 15
WILSON_Z_95 = 1.959963984540054
NO_SUBRULE_LABEL = "No sub-rule assigned"

CHART_TEXT = {
    "en": {
        "summary_title": (
            "Which text spans did reviewers annotate most frequently? "
        ),
        "summary_description": (
            "Chart shows the ten recurrent text spans with the largest "
            "numbers of distinct reviewer annotations."
        ),
        "summary_measure": (
            "To be considered recurrent, a piece of text must occur in at least two documents and two reviewer "
            "packets. The chart orders texts in descending order of annotation frequency. "
        ),
        "dictionary_title": (
            "Dictionary of repeatedly annotated text spans "
        ),
        "dictionary_description": (
            "These charts form a complete visual dictionary of all recurrent text spans "
            "and the error label reviewers assigned most frequently to each."
        ),
        "dictionary_measure": (
            "To be considered recurrent, a piece of text must occur in at least two documents and two reviewer "
            "packets. "
        ),
        "x_label": "Distinct reviewer annotations",
    }
}

README_TEXT = f"""# 05 · Recurrently annotated words and phrases

## Purpose

Identify the words and phrases repeatedly marked by reviewers and the associated 
error labels. 

## What counts as a recurrent text span?

A text span is **recurrent** when the same conservatively normalised text occurs
in at least {MIN_DOCUMENTS} reviewed documents and at least {MIN_REVIEWER_PACKETS}
reviewer packets. We assess recurrence separately across both models because models.

Normalisation uses Unicode NFC, collapses whitespace, trims and case-folds. It
does not remove accents or punctuation, stem or lemmatise, split phrases, or
merge spelling variants. A single annotation is identified by document, annotation
ID and normalised text, so multiple labels attached to one annotation do not
inflate the annotation count. Missing or empty annotated text is excluded.


## Label consistency

An error label is the category-sub-rule combination assigned by a reviewer.
The most frequently assigned label is the one attached to the largest number
of distinct annotations of that span. If labels are tied in frequency, they are all
reported. 

Observed coverage is the proportion of the span's annotations carrying a most frequent label. 
The 95% Wilson lower bound is the lower end of a confidence interval for that proportion: 
it reduces when the evidence is sparse and rises when the same label is repeatedly assigned.

If one annotation has several different labels, each is retained and reported separately. 
These measures describe reviewer-label consistency. They do not establish  whether
reviewers annotated correctly or whether a model genuinely failed.

## Outputs

- `annotated_text_summary.csv`: every non-empty annotated word or phrase,
  recurrence status, evidence breadth, label measures and model split.
- `recurrent_annotated_text.csv`: every recurrent span in order of annotation
  frequency.
- `annotated_text_evidence.csv`: every individual annotation and error-label
  assignment, including the recurrence flag. 
- `recurrent_annotated_text_top_10`: the ten recurrent spans with the largest
  numbers of distinct annotations.
- `recurrent_annotated_text_index_*`: every recurrent span in alphabetical 
  order. Displayed as up to {DICTIONARY_ROWS_PER_PAGE} rows for readability. 
  Each filename records the first and last span on that page.

## Ranking

The summary and recurrent-span CSV are ordered by distinct annotations, then
affected documents, reviewer packets, archives and the Wilson lower bound for
the most frequently assigned label. The first ten rows appear in the summary 
figure (ten is just a number chosen for readability on a single page.)
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_annotated_text(value):
    """Conservative normalisation"""
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFC", str(value))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text if text else pd.NA


def join_unique(values, limit = None):
    result = []
    for value in values:
        if pd.isna(value):
            continue
        value = str(value)
        if value not in result:
            result.append(value)
        if limit is not None and len(result) >= limit:
            break
    return " | ".join(result)


def safe_identifier(value):
    identifier = re.sub(
        r"[^a-z0-9]+", "_", str(value).casefold()
    ).strip("_")
    return identifier or "unknown_model"


def filename_fragment(value):
    """Return an ASCII filename fragment without changing analytical text."""
    ascii_value = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    fragment = re.sub(
        r"[^a-z0-9]+", "_", ascii_value.casefold()
    ).strip("_")
    return fragment or "unnamed"


def wrap_error_label(value, width = 90):
    """Wrap tied error labels together within the available column width."""
    labels = [label.strip() for label in str(value).split(" | ") if label.strip()]
    return textwrap.fill(
        "; ".join(labels),
        width = width,
        break_long_words = False,
        break_on_hyphens = False,
    )


def wrap_annotated_text(value, width = 28):
    """Wrap an annotated word or phrase within its dedicated column."""
    return textwrap.fill(
        str(value), width = width, break_long_words = True,
        break_on_hyphens = False,
    )


def wilson_lower_bound(successes, trials, z = WILSON_Z_95):
    """Lower endpoint of a two-sided 95% Wilson interval."""
    if trials <= 0:
        return np.nan
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = proportion + z**2 / (2 * trials)
    adjustment = z * np.sqrt(
        proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
    )
    return (centre - adjustment) / denominator


def add_model_columns(summary, annotations, models):
    result = summary.copy()
    for model in models:
        model_identifier = safe_identifier(model)
        counts = (
            annotations.loc[annotations["model"].eq(model)]
            .groupby("normalised_annotated_text", dropna = False)
            .agg(
                annotations = ("annotation_identity", "nunique"),
                documents = ("filename_stem", "nunique"),
            )
        )
        result = result.merge(
            counts.rename(columns = {
                "annotations": f"{model_identifier}_annotations",
                "documents": f"{model_identifier}_documents",
            }),
            left_on = "normalised_annotated_text", right_index = True,
            how = "left", validate = "one_to_one",
        )
        for suffix in ("annotations", "documents"):
            column = f"{model_identifier}_{suffix}"
            result[column] = result[column].fillna(0).astype(int)
    return result


def draw_span_chart(frame, models, colours, title, description, measure,
                    output_directory, filename_stem, panel_title,
                    description_y = 0.900, measure_y = 0.830):
    """Draw a text index and model bars in separate, aligned panels."""
    display = frame.iloc[::-1].reset_index(drop = True)
    row_count = len(display)
    fig = plt.figure(figsize = (12, 7.2))
    grid = fig.add_gridspec(
        nrows = 1, ncols = 3, width_ratios = [0.38, 0.95, 1.02],
        left = 0.06, right = 0.96, bottom = 0.17, top = 0.73,
        wspace = 0.015,
    )
    span_ax = fig.add_subplot(grid[0, 0])
    label_ax = fig.add_subplot(grid[0, 1], sharey = span_ax)
    ax = fig.add_subplot(grid[0, 2], sharey = span_ax)
    y_positions = np.arange(row_count, dtype = float)
    group_height = 0.72
    bar_height = group_height / max(len(models), 1)
    maximum = 0

    for position, model in enumerate(models):
        model_identifier = safe_identifier(model)
        counts = display[f"{model_identifier}_annotations"].to_numpy()
        documents = display[f"{model_identifier}_documents"].to_numpy()
        maximum = max(maximum, int(counts.max()) if len(counts) else 0)
        offsets = (y_positions - group_height / 2 + bar_height / 2
                   + position * bar_height)
        bars = ax.barh(
            offsets, counts, height = bar_height * 0.82,
            color = colours[model], label = model, zorder = 3,
        )
        for bar, count, document_count in zip(bars, counts, documents):
            if count:
                ax.annotate(
                    f"{int(count)} annotations · {int(document_count)} docs",
                    xy = (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext = (4, 0), textcoords = "offset points", va = "center",
                    fontsize = 6.1, color = COLOURS["text"],
                )

    for text_axis in (span_ax, label_ax):
        text_axis.set_xlim(0, 1)
        text_axis.set_ylim(-0.5, row_count - 0.5)
        text_axis.axis("off")
    span_text_x = (
        (0.08 - span_ax.get_position().x0) / span_ax.get_position().width
    )
    span_ax.text(
        span_text_x, 1.015, "Annotated text",
        transform = span_ax.transAxes,
        ha = "left", va = "bottom", fontsize = 8.2, fontweight = "bold",
        color = COLOURS["text"], clip_on = False,
    )
    label_ax.text(
        0.00, 1.015, "Error label assigned most frequently",
        transform = label_ax.transAxes, ha = "left", va = "bottom",
        fontsize = 8.2, fontweight = "bold", color = COLOURS["text"],
        clip_on = False,
    )
    for y_position, row in zip(y_positions, display.itertuples(index = False)):
        span_ax.text(
            span_text_x, y_position,
            wrap_annotated_text(row.normalised_annotated_text),
            transform = span_ax.get_yaxis_transform(), ha = "left", va = "center",
            fontsize = 7.4, fontweight = "bold", fontstyle = "italic",
            linespacing = 1.05, color = COLOURS["text"], clip_on = True,
        )
        label_ax.text(
            0.00, y_position,
            wrap_error_label(row.most_frequently_assigned_error_label),
            transform = label_ax.get_yaxis_transform(), ha = "left", va = "center",
            fontsize = 7.0, linespacing = 1.08,
            color = COLOURS["muted_text"], clip_on = True,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.tick_params(axis = "y", length = 0)
    ax.set_title(panel_title, loc = "left", fontsize = 10, fontweight = "bold", pad = 10)
    ax.set_xlabel(CHART_TEXT["en"]["x_label"])
    ax.set_xlim(0, maximum * 1.27 if maximum else 1)
    ax.grid(axis = "x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon = False, loc = "lower center",
        bbox_to_anchor = (0.5, 0.035), ncol = max(1, len(models)),
    )
    add_chart_header(fig, title = title, description = description, measure = measure)
    if description_y is not None:
        wrapped_description = textwrap.fill(description, width = 105)
        for text_artist in fig.texts:
            if text_artist.get_text() == wrapped_description:
                text_artist.set_y(description_y)
                break
    wrapped_measure = textwrap.fill(measure, width = 110)
    for text_artist in fig.texts:
        if text_artist.get_text() == wrapped_measure:
            text_artist.set_y(measure_y)
            break
    save_figure(fig, output_directory, filename_stem)


# ---------------------------------------------------------------------------
# Load and validate pre-assembled tables
# ---------------------------------------------------------------------------

check_required_files(
    [DOCUMENT_ANALYSIS_PATH, ANNOTATION_ANALYSIS_PATH],
    preceding_command = "python scripts/construct_tables.py",
)
documents_df = pd.read_csv(
    DOCUMENT_ANALYSIS_PATH, dtype = {"filename_stem": "string"}
)

duplicate_documents = documents_df["filename_stem"].duplicated(keep = False)
if duplicate_documents.any():
    examples = join_unique(
        documents_df.loc[duplicate_documents, "filename_stem"], 5
    )
    raise ValueError(
        "document_analysis.csv contains more than one row for a document: "
        f"{examples}"
    )
annotations_df = pd.read_csv(
    ANNOTATION_ANALYSIS_PATH,
    dtype = {
        "filename_stem": "string", "annotation_id": "string",
        "annotated_text": "string", "effective_error_category": "string",
        "effective_subrule": "string",
    },
)

required_document_columns = {
    "filename_stem", "model", "archive", "reviewer_packet", "reviewer_name"
}
required_annotation_columns = {
    "filename_stem", "annotation_id", "annotated_text",
    "include_in_analysis", "effective_error_category", "effective_subrule",
}
missing_documents = required_document_columns - set(documents_df.columns)
missing_annotations = required_annotation_columns - set(annotations_df.columns)
if missing_documents:
    raise ValueError(
        "document_analysis.csv is missing required columns: "
        f"{sorted(missing_documents)}."
    )
if missing_annotations:
    raise ValueError(
        "annotation_analysis.csv is missing required columns: "
        f"{sorted(missing_annotations)}."
    )

annotations_df["include_in_analysis"] = coerce_boolean(
    annotations_df["include_in_analysis"]
)
included_df = annotations_df.loc[
    annotations_df["include_in_analysis"].fillna(False)
].copy()
metadata_columns = [
    "filename_stem", "model", "archive", "reviewer_packet", "reviewer_name"
]
included_df = included_df.drop(columns = [
    column for column in metadata_columns[1:] if column in included_df
]).merge(
    documents_df[metadata_columns], on = "filename_stem", how = "left",
    validate = "many_to_one",
)
if included_df["model"].isna().any():
    raise ValueError("Some included annotations lack matching document metadata.")
if included_df["reviewer_packet"].isna().any():
    raise ValueError("Some included annotations lack a reviewer packet.")
if included_df["annotation_id"].isna().any():
    raise ValueError("Some included assignments lack an annotation ID.")
if included_df["effective_error_category"].isna().any():
    raise ValueError("Some included assignments lack an error category.")

included_df["normalised_annotated_text"] = included_df["annotated_text"].map(
    normalise_annotated_text
)
missing_annotated_text_assignments = int(
    included_df["normalised_annotated_text"].isna().sum()
)
included_df = included_df.dropna(subset = ["normalised_annotated_text"]).copy()
included_df["effective_subrule"] = included_df["effective_subrule"].fillna(
    NO_SUBRULE_LABEL
)
included_df["annotation_identity"] = (
    included_df["filename_stem"].astype(str) + "\x1f"
    + included_df["annotation_id"].astype(str) + "\x1f"
    + included_df["normalised_annotated_text"].astype(str)
)


# ---------------------------------------------------------------------------
# Count annotated text spans and reviewer-assigned labels
# ---------------------------------------------------------------------------

span_identity = ["filename_stem", "annotation_id", "normalised_annotated_text"]
span_annotations_df = included_df.drop_duplicates(subset = span_identity).copy()
error_label_identity = span_identity + ["effective_error_category", "effective_subrule"]
error_label_assignments_df = included_df.drop_duplicates(subset = error_label_identity).copy()
error_label_assignments_df["error_label"] = (
    error_label_assignments_df["effective_error_category"] + " — "
    + error_label_assignments_df["effective_subrule"]
)

span_summary_df = (
    span_annotations_df.groupby("normalised_annotated_text", dropna = False)
    .agg(
        distinct_annotations = ("annotation_identity", "nunique"),
        affected_documents = ("filename_stem", "nunique"),
        affected_reviewer_packets = ("reviewer_packet", "nunique"),
        affected_reviewers = ("reviewer_name", "nunique"),
        affected_archives = ("archive", "nunique"),
        affected_models = ("model", "nunique"),
        example_surface_text = ("annotated_text", lambda x: join_unique(x, 8)),
        example_documents = ("filename_stem", lambda x: join_unique(x, 8)),
        reviewer_packets = ("reviewer_packet", join_unique),
        reviewers = ("reviewer_name", join_unique),
        archives = ("archive", join_unique),
    ).reset_index()
)

error_label_counts_df = (
    error_label_assignments_df.groupby(
        ["normalised_annotated_text", "effective_error_category", "effective_subrule"],
        dropna = False,
    )["annotation_identity"].nunique().rename("labelled_annotations").reset_index()
    .sort_values(
        ["normalised_annotated_text", "labelled_annotations",
         "effective_error_category", "effective_subrule"],
        ascending = [True, False, True, True],
    )
)
error_label_counts_df["error_label"] = (
    error_label_counts_df["effective_error_category"] + " — "
    + error_label_counts_df["effective_subrule"]
)
error_label_counts_df["maximum_labelled_annotations"] = (
    error_label_counts_df.groupby("normalised_annotated_text")[
        "labelled_annotations"
    ].transform("max")
)
most_frequent_error_labels_df = (
    error_label_counts_df.loc[
        error_label_counts_df["labelled_annotations"].eq(
            error_label_counts_df["maximum_labelled_annotations"]
        )
    ]
    .groupby("normalised_annotated_text", as_index = False)
    .agg(
        most_frequently_assigned_error_label = ("error_label", join_unique),
        number_of_joint_most_frequent_labels = ("error_label", "nunique"),
        annotations_with_most_frequent_error_label = ("labelled_annotations", "first"),
    )
)
all_error_labels_df = (
    error_label_counts_df.assign(
        error_label_with_count = lambda frame: (
            frame["error_label"] + " [" + frame["labelled_annotations"].astype(str) + "]"
        )
    ).groupby("normalised_annotated_text", dropna = False)["error_label_with_count"]
    .agg(join_unique).rename("all_assigned_error_labels_with_counts").reset_index()
)

# Count the number of distinct rules attached to each original annotation.
# This preserves the multiple labels without counting the marked span twice.
annotation_error_label_counts_df = (
    error_label_assignments_df.groupby(span_identity, dropna = False)
    .size().rename("error_labels_on_annotation").reset_index()
)
multiple_error_label_summary_df = (
    annotation_error_label_counts_df.groupby("normalised_annotated_text", dropna = False)
    .agg(
        total_error_label_assignments = ("error_labels_on_annotation", "sum"),
        annotations_with_multiple_error_labels = (
            "error_labels_on_annotation", lambda values: int(values.gt(1).sum())
        ),
    ).reset_index()
)
span_summary_df = (
    span_summary_df.merge(
        most_frequent_error_labels_df, on = "normalised_annotated_text", how = "left",
        validate = "one_to_one",
    ).merge(
        all_error_labels_df, on = "normalised_annotated_text", how = "left",
        validate = "one_to_one",
    ).merge(
        multiple_error_label_summary_df, on = "normalised_annotated_text", how = "left",
        validate = "one_to_one",
    )
)
span_summary_df["most_frequent_error_label_coverage"] = (
    span_summary_df["annotations_with_most_frequent_error_label"]
    / span_summary_df["distinct_annotations"]
)
span_summary_df["most_frequent_error_label_wilson_lower_95"] = span_summary_df.apply(
    lambda row: wilson_lower_bound(
        row["annotations_with_most_frequent_error_label"], row["distinct_annotations"]
    ), axis = 1,
)
span_summary_df["multiple_error_label_annotation_share"] = (
    span_summary_df["annotations_with_multiple_error_labels"]
    / span_summary_df["distinct_annotations"]
)
span_summary_df["recurrent_across_documents_and_reviewers"] = (
    span_summary_df["affected_documents"].ge(MIN_DOCUMENTS)
    & span_summary_df["affected_reviewer_packets"].ge(MIN_REVIEWER_PACKETS)
)
models = sorted(str(model) for model in included_df["model"].dropna().unique())
span_summary_df = add_model_columns(span_summary_df, span_annotations_df, models)
span_summary_df = span_summary_df.sort_values(
    [
        "recurrent_across_documents_and_reviewers", "distinct_annotations",
        "affected_documents", "affected_reviewer_packets", "affected_archives",
        "most_frequent_error_label_wilson_lower_95",
        "normalised_annotated_text",
    ],
    ascending = [False, False, False, False, False, False, True],
).reset_index(drop = True)
span_summary_df.insert(0, "span_rank", range(1, len(span_summary_df) + 1))
recurrent_spans_df = span_summary_df.loc[
    span_summary_df["recurrent_across_documents_and_reviewers"]
].copy()


# ---------------------------------------------------------------------------
# Preserve the individual annotations
# ---------------------------------------------------------------------------

summary_fields = [
    "normalised_annotated_text", "span_rank", "distinct_annotations",
    "affected_documents", "affected_reviewer_packets", "affected_archives",
    "affected_reviewers",
    "total_error_label_assignments", "annotations_with_multiple_error_labels",
    "multiple_error_label_annotation_share", "most_frequently_assigned_error_label",
    "number_of_joint_most_frequent_labels",
    "most_frequent_error_label_coverage", "most_frequent_error_label_wilson_lower_95",
    "recurrent_across_documents_and_reviewers",
]
preferred_example_columns = [
    "model", "filename_stem", "archive", "reviewer_packet", "reviewer_name",
    "annotation_id", "paragraph", "annotated_text", "normalised_annotated_text",
    "effective_error_category", "effective_subrule", "start", "end",
    "modernised_txt_path", "pre_modernisation_txt_path",
]
example_columns = [
    column for column in preferred_example_columns
    if column in error_label_assignments_df.columns
]
examples_df = error_label_assignments_df[example_columns].merge(
    span_summary_df[summary_fields], on = "normalised_annotated_text",
    how = "left", validate = "many_to_one",
).sort_values(
    ["recurrent_across_documents_and_reviewers", "span_rank", "model",
     "filename_stem", "annotation_id", "effective_error_category",
     "effective_subrule"],
    ascending = [False, True, True, True, True, True, True],
)


# ---------------------------------------------------------------------------
# Replace this stage's outputs and save tables on a re-run
# ---------------------------------------------------------------------------

if (SECTION_OUTPUT_DIR.name != "05_recurrent_annotated_text"
        or SECTION_OUTPUT_DIR.parent != ANALYSIS_OUTPUT_DIR):
    raise RuntimeError(f"Unsafe output directory: {SECTION_OUTPUT_DIR}")
if SECTION_OUTPUT_DIR.exists():
    shutil.rmtree(SECTION_OUTPUT_DIR)
tables_directory, figures_directory = create_output_folders(SECTION_OUTPUT_DIR)
SECTION_README_PATH.write_text(README_TEXT, encoding = "utf-8")
save_table(span_summary_df, tables_directory / "annotated_text_summary.csv")
save_table(
    recurrent_spans_df,
    tables_directory / "recurrent_annotated_text.csv",
)
save_table(examples_df, tables_directory / "annotated_text_evidence.csv")


# ---------------------------------------------------------------------------
# Create the 10-span summary and the complete dictionary
# ---------------------------------------------------------------------------

if recurrent_spans_df.empty:
    print("\nNo text span met the recurrence rule; tables and README were created.")
else:
    apply_plot_style()
    colours = model_colour_map(models)
    for language in OUTPUT_LANGUAGES:
        if language not in CHART_TEXT:
            raise ValueError(f"No chart wording supplied for: {language}.")
        text = CHART_TEXT[language]
        language_directory = figures_directory / language
        draw_span_chart(
            recurrent_spans_df.head(SUMMARY_SPAN_LIMIT), models, colours,
            text["summary_title"], text["summary_description"],
            text["summary_measure"], language_directory,
            "recurrent_annotated_text_top_10",
            "Ten most frequently annotated spans",
            measure_y = 0.850,
        )

        alphabetical_df = recurrent_spans_df.sort_values(
            "normalised_annotated_text",
            key = lambda values: values.str.casefold(),
        ).reset_index(drop = True)
        page_count = math.ceil(len(alphabetical_df) / DICTIONARY_ROWS_PER_PAGE)
        for page_number in range(1, page_count + 1):
            start = (page_number - 1) * DICTIONARY_ROWS_PER_PAGE
            page_df = alphabetical_df.iloc[start:start + DICTIONARY_ROWS_PER_PAGE]
            first_span = page_df.iloc[0]["normalised_annotated_text"]
            last_span = page_df.iloc[-1]["normalised_annotated_text"]
            first_span_fragment = filename_fragment(first_span)
            last_span_fragment = filename_fragment(last_span)
            span_range = f"{first_span_fragment}_to_{last_span_fragment}"
            draw_span_chart(
                page_df, models, colours, text["dictionary_title"],
                text["dictionary_description"], text["dictionary_measure"],
                language_directory,
                f"recurrent_annotated_text_index_{span_range}",
                f"Alphabetical index: {first_span}–{last_span}",
            )

print(f"\nObserved non-empty normalised spans: {len(span_summary_df)}")
print(f"Excluded assignments without annotated text: {missing_annotated_text_assignments}")
print(f"Recurrent text spans: {len(recurrent_spans_df)}")
print(
    "Spans in summary figure: "
    f"{min(SUMMARY_SPAN_LIMIT, len(recurrent_spans_df))}"
)
print("Alphabetical dictionary pages: " + str(
    math.ceil(len(recurrent_spans_df) / DICTIONARY_ROWS_PER_PAGE)
    if len(recurrent_spans_df) else 0
))
print(f"Individual span-label evidence rows saved: {len(examples_df)}")
print(f"Tables saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
