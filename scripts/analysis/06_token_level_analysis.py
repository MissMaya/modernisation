"""
Identify recurrent reviewer-annotated forms that could be fed back into a prompt.

A form is recurrent when the same normalised form occurs in at
least two documents and is recorded by at least two independent reviewer
packets. The criteria was combined across models because model exposure 
was unequal.

Normalisation applies Unicode NFC, collapses whitespace, trims and normalises
cases. The process does not remove accents or punctuation, stem, lemmatise, 
split phrases or combine spelling variants. 

One annotation is identified by filename, annotation ID and normalised form, 
so if a token receives many different types of correction, having multiple rows 
in the dataframe does not inflate form counts. 

The most frequently assigned reviewer rule (error_code + optionally field code), 
its coverage and its 95% Wilson lower bound measure labelling consistency. Rules are not assumed to be
mutually exclusive: multi-label annotations are reported separately. These
measures do not prove reviewer correctness or a genuine model failure.

Outputs are three CSVs, a 10-form summary figure and a complete alphabetically
paginated visual dictionary. The figure limits are presentational only.
"""

# ---------------------------------------------------------------------------
# Import required modules 
# ---------------------------------------------------------------------------
import math
import os
from pathlib import Path
import re
import shutil
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import (
    COLOURS, OUTPUT_LANGUAGES, add_chart_header, add_figure_note,
    apply_plot_style, check_required_files, coerce_boolean,
    create_output_folders, model_colour_map, save_figure, save_table,
)


# ---------------------------------------------------------------------------
# Define paths, set analytical rules and chart labelling 
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(os.environ.get(
    "MODERNISATION_PROJECT_DIR", "/workspaces/modernisation"
))
DOCUMENT_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "document_analysis.csv"
ANNOTATION_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "annotation_analysis.csv"
ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "06_token_level_analysis"
SECTION_README_PATH = SECTION_OUTPUT_DIR / "TOKEN_LEVEL_ANALYSIS_README.md"

MIN_DOCUMENTS = 2
MIN_REVIEWER_PACKETS = 2
SUMMARY_FORM_LIMIT = 10
DICTIONARY_ROWS_PER_PAGE = 28
WILSON_Z_95 = 1.959963984540054
NO_SUBRULE_LABEL = "No sub-rule assigned"
MISSING_TEXT_LABEL = "[missing annotated text]"

CHART_TEXT = {
    "en": {
        "summary_title": "Which recurrent annotated text spans occurred most frequently?",
        "summary_description": (
            "A summary of the annotated text spans These reviewer-marked forms recur across independent documents and "
            "reviewer packets and are useful starting points for prompt review."
        ),
        "summary_measure": (
            "The 10 annotated text spans recurring most widely across documents and reviewers. "
            "The bars show distinct annotations by model."
        ),
        "dictionary_title": (
            "All annotated text spans arranged alphabetically"
        ),
        "dictionary_description": (
            "This A-Z of annotated text spans pairs each recurrent text span with the "
            "label reviewers assigned to it most frequently."
        ),
        "dictionary_measure": (
            "Every span shown here occurred in at least two documents and was recorded by "
            "at least two reviewer packets. The arrangement is alphabetical"
        ),
        "x_label": "Distinct reviewer annotations",
    }
}

README_TEXT = f"""# 06 · Recurrent annotated forms that could inform the prompt

## Purpose

This section identifies text forms repeatedly marked by reviewers and pairs
each with its most frequently assigned category–sub-rule label. It generates
candidates for prompt review; it does not declare genuine model errors.

## Recurrence rule

A form is **recurrent** when the same conservatively normalised form occurs in
at least {MIN_DOCUMENTS} documents and is recorded by at least
{MIN_REVIEWER_PACKETS} independent reviewer packets. Eligibility is pooled
across models because model exposure was unequal.

Normalisation uses Unicode NFC, collapses whitespace, trims and case-folds. It
does not remove accents or punctuation, stem or lemmatise, split phrases, or
merge spelling variants. One annotation is identified by filename, annotation
ID and normalised form, so flattened rule assignments do not inflate counts.

Archives, models and annotation volume are reported as evidence, not entry
thresholds. There is no archive minimum, annotation minimum, per-model
recurrence requirement or cumulative-coverage cutoff.

## Label consistency

The most frequently assigned rule is the category–sub-rule attached to the
largest number of distinct annotations of that form.
`most_frequent_rule_coverage` is the proportion of the form's annotations
carrying that rule; `most_frequent_rule_wilson_lower_95` is the lower endpoint
of its two-sided 95% Wilson interval. Rules are not assumed to be mutually
exclusive, so the table also reports total rule assignments, the number of
annotations carrying multiple rules and the multi-label share. These describe
reviewer-label consistency. They do not establish reviewer correctness or a
genuine model failure.

## Outputs

- `annotated_form_summary.csv`: every form, recurrence status, evidence
  breadth, most frequent rule, multi-label measures and model split.
- `recurring_forms.csv`: recurrent forms in replication-first order.
- `token_examples_for_review.csv`: individual annotation/rule metadata. It
  deliberately excludes unverified context reconstruction.
- `recurrent_forms_summary`: the first {SUMMARY_FORM_LIMIT} recurrent forms.
  Ten is only a presentation limit.
- `recurrent_forms_dictionary_*`: every recurrent form, alphabetical and
  paginated at up to {DICTIONARY_ROWS_PER_PAGE} rows for readability.

## Ranking

The summary and CSV use a lexicographic replication-first ordering: affected
documents, affected reviewer packets, most-frequent-rule Wilson lower bound,
affected archives, then distinct annotations. No weighted score is used.

## Caution

Validate examples before revising a prompt. Context is not reconstructed here:
annotation offsets must first be verified against the modernised text. Verified
modernised/source context belongs in Stage 07.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_form(value):
    """Normalise conservatively without changing linguistic content."""
    if pd.isna(value):
        return MISSING_TEXT_LABEL
    text = unicodedata.normalize("NFC", str(value))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text if text else MISSING_TEXT_LABEL


def join_unique(values, limit=None):
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


def safe_slug(value):
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    return slug or "unknown_model"


def wilson_lower_bound(successes, trials, z=WILSON_Z_95):
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
        slug = safe_slug(model)
        counts = (
            annotations.loc[annotations["model"].eq(model)]
            .groupby("normalised_annotated_form", dropna=False)
            .agg(
                annotations=("annotation_identity", "nunique"),
                documents=("filename_stem", "nunique"),
            )
        )
        result = result.merge(
            counts.rename(columns={
                "annotations": f"{slug}_annotations",
                "documents": f"{slug}_documents",
            }),
            left_on="normalised_annotated_form", right_index=True,
            how="left", validate="one_to_one",
        )
        for suffix in ("annotations", "documents"):
            column = f"{slug}_{suffix}"
            result[column] = result[column].fillna(0).astype(int)
    return result


def draw_form_chart(frame, models, colours, title, description, measure,
                    output_directory, filename_stem, panel_title):
    """Draw a form/rule table aligned with model-specific bars."""
    display = frame.iloc[::-1].reset_index(drop=True)
    row_count = len(display)
    fig, ax = plt.subplots(figsize=(17.5, max(9.5, 6.1 + row_count * 0.38)))
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.52, right=0.96)
    y_positions = np.arange(row_count, dtype=float)
    group_height = 0.72
    bar_height = group_height / max(len(models), 1)
    maximum = 0

    for position, model in enumerate(models):
        slug = safe_slug(model)
        counts = display[f"{slug}_annotations"].to_numpy()
        documents = display[f"{slug}_documents"].to_numpy()
        maximum = max(maximum, int(counts.max()) if len(counts) else 0)
        offsets = (y_positions - group_height / 2 + bar_height / 2
                   + position * bar_height)
        bars = ax.barh(
            offsets, counts, height=bar_height * 0.82,
            color=colours[model], label=model, zorder=3,
        )
        for bar, count, document_count in zip(bars, counts, documents):
            if count:
                ax.annotate(
                    f"{int(count)} · {int(document_count)} docs",
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=6.1, color=COLOURS["text"],
                )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.text(
        -0.91, 1.015, "Annotated form", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8.2, fontweight="bold",
        color=COLOURS["text"], clip_on=False,
    )
    ax.text(
        -0.63, 1.015, "Reviewer label assigned most frequently",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2,
        fontweight="bold", color=COLOURS["text"], clip_on=False,
    )
    for y_position, row in zip(y_positions, display.itertuples(index=False)):
        ax.text(
            -0.91, y_position, row.normalised_annotated_form,
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=7.4, fontweight="bold", fontstyle="italic",
            color=COLOURS["text"], clip_on=False,
        )
        ax.text(
            -0.63, y_position, row.most_frequently_assigned_rule,
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=6.4, color=COLOURS["muted_text"], clip_on=False,
        )

    ax.set_title(panel_title, loc="left", fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel(CHART_TEXT["en"]["x_label"])
    ax.set_xlim(0, maximum * 1.27 if maximum else 1)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, 0.065), ncol=max(1, len(models)),
    )
    add_chart_header(fig, title=title, description=description, measure=measure)
    add_figure_note(
        fig,
        "Bold italics identify the annotated form. The adjacent text is its most frequently assigned category–sub-rule label. Bar labels show annotations · affected documents. These are candidates for validation, not proven model errors.",
    )
    save_figure(fig, output_directory, filename_stem)


# ---------------------------------------------------------------------------
# Load and validate assembled tables
# ---------------------------------------------------------------------------

check_required_files(
    [DOCUMENT_ANALYSIS_PATH, ANNOTATION_ANALYSIS_PATH],
    preceding_command="python scripts/construct_tables.py",
)
documents_df = pd.read_csv(
    DOCUMENT_ANALYSIS_PATH, dtype={"filename_stem": "string"}
)
annotations_df = pd.read_csv(
    ANNOTATION_ANALYSIS_PATH,
    dtype={
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
included_df = included_df.drop(columns=[
    column for column in metadata_columns[1:] if column in included_df
]).merge(
    documents_df[metadata_columns], on="filename_stem", how="left",
    validate="many_to_one",
)
if included_df["model"].isna().any():
    raise ValueError("Some included annotations lack matching document metadata.")

included_df["normalised_annotated_form"] = included_df["annotated_text"].map(
    normalise_form
)
included_df["effective_error_category"] = included_df[
    "effective_error_category"
].fillna("Unmapped category")
included_df["effective_subrule"] = included_df["effective_subrule"].fillna(
    NO_SUBRULE_LABEL
)
included_df["annotation_identity"] = (
    included_df["filename_stem"].astype(str) + "\x1f"
    + included_df["annotation_id"].astype(str) + "\x1f"
    + included_df["normalised_annotated_form"].astype(str)
)


# ---------------------------------------------------------------------------
# Count forms and reviewer-assigned labels
# ---------------------------------------------------------------------------

form_identity = ["filename_stem", "annotation_id", "normalised_annotated_form"]
form_annotations_df = included_df.drop_duplicates(subset=form_identity).copy()
rule_identity = form_identity + ["effective_error_category", "effective_subrule"]
rule_assignments_df = included_df.drop_duplicates(subset=rule_identity).copy()
rule_assignments_df["rule_label"] = (
    rule_assignments_df["effective_error_category"] + " — "
    + rule_assignments_df["effective_subrule"]
)

form_summary_df = (
    form_annotations_df.groupby("normalised_annotated_form", dropna=False)
    .agg(
        distinct_annotations=("annotation_identity", "nunique"),
        affected_documents=("filename_stem", "nunique"),
        affected_reviewer_packets=("reviewer_packet", "nunique"),
        affected_archives=("archive", "nunique"),
        affected_models=("model", "nunique"),
        example_surface_forms=("annotated_text", lambda x: join_unique(x, 8)),
        example_documents=("filename_stem", lambda x: join_unique(x, 8)),
        reviewer_packets=("reviewer_packet", join_unique),
        reviewers=("reviewer_name", join_unique),
        archives=("archive", join_unique),
    ).reset_index()
)

rule_counts_df = (
    rule_assignments_df.groupby(
        ["normalised_annotated_form", "effective_error_category", "effective_subrule"],
        dropna=False,
    )["annotation_identity"].nunique().rename("labelled_annotations").reset_index()
    .sort_values(
        ["normalised_annotated_form", "labelled_annotations",
         "effective_error_category", "effective_subrule"],
        ascending=[True, False, True, True],
    )
)
rule_counts_df["rule_label"] = (
    rule_counts_df["effective_error_category"] + " — "
    + rule_counts_df["effective_subrule"]
)
most_frequent_rules_df = (
    rule_counts_df.drop_duplicates("normalised_annotated_form", keep="first")
    .rename(columns={
        "effective_error_category": "most_frequent_error_category",
        "effective_subrule": "most_frequent_subrule",
        "labelled_annotations": "annotations_with_most_frequent_rule",
        "rule_label": "most_frequently_assigned_rule",
    })[[
        "normalised_annotated_form", "most_frequent_error_category",
        "most_frequent_subrule", "most_frequently_assigned_rule",
        "annotations_with_most_frequent_rule",
    ]]
)
all_labels_df = (
    rule_counts_df.assign(
        rule_with_count=lambda frame: (
            frame["rule_label"] + " [" + frame["labelled_annotations"].astype(str) + "]"
        )
    ).groupby("normalised_annotated_form", dropna=False)["rule_with_count"]
    .agg(join_unique).rename("all_assigned_rules_with_counts").reset_index()
)

# Count the number of distinct rules attached to each original annotation.
# This preserves multi-label evidence without counting the marked form twice.
annotation_rule_counts_df = (
    rule_assignments_df.groupby(form_identity, dropna=False)
    .size().rename("rules_on_annotation").reset_index()
)
multi_label_summary_df = (
    annotation_rule_counts_df.groupby("normalised_annotated_form", dropna=False)
    .agg(
        total_rule_assignments=("rules_on_annotation", "sum"),
        annotations_with_multiple_rules=(
            "rules_on_annotation", lambda values: int(values.gt(1).sum())
        ),
    ).reset_index()
)
form_summary_df = (
    form_summary_df.merge(
        most_frequent_rules_df, on="normalised_annotated_form", how="left",
        validate="one_to_one",
    ).merge(
        all_labels_df, on="normalised_annotated_form", how="left",
        validate="one_to_one",
    ).merge(
        multi_label_summary_df, on="normalised_annotated_form", how="left",
        validate="one_to_one",
    )
)
form_summary_df["most_frequent_rule_coverage"] = (
    form_summary_df["annotations_with_most_frequent_rule"]
    / form_summary_df["distinct_annotations"]
)
form_summary_df["most_frequent_rule_wilson_lower_95"] = form_summary_df.apply(
    lambda row: wilson_lower_bound(
        row["annotations_with_most_frequent_rule"], row["distinct_annotations"]
    ), axis=1,
)
form_summary_df["multiple_rule_annotation_share"] = (
    form_summary_df["annotations_with_multiple_rules"]
    / form_summary_df["distinct_annotations"]
)
form_summary_df["recurrent_across_documents_and_reviewers"] = (
    form_summary_df["affected_documents"].ge(MIN_DOCUMENTS)
    & form_summary_df["affected_reviewer_packets"].ge(MIN_REVIEWER_PACKETS)
)
models = sorted(str(model) for model in included_df["model"].dropna().unique())
form_summary_df = add_model_columns(form_summary_df, form_annotations_df, models)
form_summary_df = form_summary_df.sort_values(
    [
        "recurrent_across_documents_and_reviewers", "affected_documents",
        "affected_reviewer_packets", "most_frequent_rule_wilson_lower_95",
        "affected_archives", "distinct_annotations", "normalised_annotated_form",
    ],
    ascending=[False, False, False, False, False, False, True],
).reset_index(drop=True)
form_summary_df.insert(0, "form_rank", range(1, len(form_summary_df) + 1))
recurring_forms_df = form_summary_df.loc[
    form_summary_df["recurrent_across_documents_and_reviewers"]
].copy()


# ---------------------------------------------------------------------------
# Preserve individual evidence without trusting unverified text offsets
# ---------------------------------------------------------------------------

summary_fields = [
    "normalised_annotated_form", "form_rank", "distinct_annotations",
    "affected_documents", "affected_reviewer_packets", "affected_archives",
    "total_rule_assignments", "annotations_with_multiple_rules",
    "multiple_rule_annotation_share", "most_frequently_assigned_rule",
    "most_frequent_rule_coverage", "most_frequent_rule_wilson_lower_95",
    "recurrent_across_documents_and_reviewers",
]
preferred_example_columns = [
    "model", "filename_stem", "archive", "reviewer_packet", "reviewer_name",
    "annotation_id", "paragraph", "annotated_text", "normalised_annotated_form",
    "effective_error_category", "effective_subrule", "start", "end",
    "modernised_txt_path", "pre_modernisation_txt_path",
]
example_columns = [
    column for column in preferred_example_columns
    if column in rule_assignments_df.columns
]
examples_df = rule_assignments_df[example_columns].merge(
    form_summary_df[summary_fields], on="normalised_annotated_form",
    how="left", validate="many_to_one",
).sort_values(
    ["recurrent_across_documents_and_reviewers", "form_rank", "model",
     "filename_stem", "annotation_id", "effective_error_category",
     "effective_subrule"],
    ascending=[False, True, True, True, True, True, True],
)


# ---------------------------------------------------------------------------
# Replace this stage's outputs and save tables
# ---------------------------------------------------------------------------

if (SECTION_OUTPUT_DIR.name != "06_token_failure_analysis"
        or SECTION_OUTPUT_DIR.parent != ANALYSIS_OUTPUT_DIR):
    raise RuntimeError(f"Unsafe output directory: {SECTION_OUTPUT_DIR}")
if SECTION_OUTPUT_DIR.exists():
    shutil.rmtree(SECTION_OUTPUT_DIR)
tables_directory, figures_directory = create_output_folders(SECTION_OUTPUT_DIR)
SECTION_README_PATH.write_text(README_TEXT, encoding="utf-8")
save_table(form_summary_df, tables_directory / "annotated_form_summary.csv")
save_table(recurring_forms_df, tables_directory / "recurring_forms.csv")
save_table(examples_df, tables_directory / "token_examples_for_review.csv")


# ---------------------------------------------------------------------------
# Create the 10-form summary and complete alphabetical dictionary
# ---------------------------------------------------------------------------

if recurring_forms_df.empty:
    print("\nNo form met the recurrence rule; tables and README were created.")
else:
    apply_plot_style()
    colours = model_colour_map(models)
    for language in OUTPUT_LANGUAGES:
        if language not in CHART_TEXT:
            raise ValueError(f"No chart wording supplied for: {language}.")
        text = CHART_TEXT[language]
        language_directory = figures_directory / language
        draw_form_chart(
            recurring_forms_df.head(SUMMARY_FORM_LIMIT), models, colours,
            text["summary_title"], text["summary_description"],
            text["summary_measure"], language_directory,
            "recurrent_forms_summary", "Replication-first ordering",
        )

        alphabetical_df = recurring_forms_df.sort_values(
            "normalised_annotated_form",
            key=lambda values: values.str.casefold(),
        ).reset_index(drop=True)
        page_count = math.ceil(len(alphabetical_df) / DICTIONARY_ROWS_PER_PAGE)
        for page_number in range(1, page_count + 1):
            start = (page_number - 1) * DICTIONARY_ROWS_PER_PAGE
            page_df = alphabetical_df.iloc[start:start + DICTIONARY_ROWS_PER_PAGE]
            first_form = page_df.iloc[0]["normalised_annotated_form"]
            last_form = page_df.iloc[-1]["normalised_annotated_form"]
            draw_form_chart(
                page_df, models, colours, text["dictionary_title"],
                text["dictionary_description"], text["dictionary_measure"],
                language_directory,
                f"recurrent_forms_dictionary_{page_number}_of_{page_count}",
                f"Alphabetical page {page_number} of {page_count}: "
                f"{first_form}–{last_form}",
            )

print(f"\nObserved normalised forms: {len(form_summary_df)}")
print(f"Recurrent forms: {len(recurring_forms_df)}")
print(f"Forms in summary figure: {min(SUMMARY_FORM_LIMIT, len(recurring_forms_df))}")
print("Alphabetical dictionary pages: " + str(
    math.ceil(len(recurring_forms_df) / DICTIONARY_ROWS_PER_PAGE)
    if len(recurring_forms_df) else 0
))
print(f"Individual form-rule evidence rows saved: {len(examples_df)}")
print(f"Tables saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
