"""Identify annotated forms that could lead to concrete prompt improvements.

This stage combines the text marked by a reviewer with the error category and
sub-rule assigned to it. This is more informative than listing tokens alone:
``magistratura · Abreviaturas — Expansión completa`` states both what was
marked and why the reviewer believed it was wrong.

The figure is deliberately limited to recurring evidence. An annotated form
qualifies as a cross-context prompt candidate for a model when it appears in at
least two documents, two archives and two reviewer packets. Forms are ranked
by annotation count and retained until at least 80% of their annotations is
represented; ties are included. Each form appears only once in the chart and
its most commonly assigned rule is shown beneath it.

The chart prioritises investigation; it does not prove that a model was wrong.
The example table therefore preserves modernised context and document metadata
for human validation before any prompt is changed.

Outputs:
    analysis_outputs/06_token_failure_analysis/
        TOKEN_FAILURE_ANALYSIS_README.md
        tables/annotated_form_summary.csv
        tables/token_examples_for_review.csv
        figures/en/prompt_candidate_forms_and_rules_1_of_3.png and .svg
        figures/en/prompt_candidate_forms_and_rules_2_of_3.png and .svg
        figures/en/prompt_candidate_forms_and_rules_3_of_3.png and .svg
"""

import os
from pathlib import Path
import re
import shutil
import unicodedata

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
# Paths and transparent analytical rules
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(
    os.environ.get("MODERNISATION_PROJECT_DIR", "/workspaces/modernisation")
)
DOCUMENT_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "document_analysis.csv"
ANNOTATION_ANALYSIS_PATH = PROJECT_DIR / "outputs" / "annotation_analysis.csv"
ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "06_token_failure_analysis"
SECTION_README_PATH = SECTION_OUTPUT_DIR / "TOKEN_FAILURE_ANALYSIS_README.md"

MIN_DOCUMENTS = 2
MIN_ARCHIVES = 2
MIN_REVIEWERS = 2
CUMULATIVE_COVERAGE = 0.80
CONTEXT_CHARACTERS = 90
NO_SUBRULE_LABEL = "No sub-rule assigned"
MISSING_TEXT_LABEL = "[missing annotated text]"


CHART_TEXT = {
    "en": {
        "title": (
            "Which recurring forms and reviewer-assigned errors could inform "
            "the prompt?"
        ),
        "description": (
            "Each form is paired with the error category and sub-rule that "
            "reviewers assigned to it most frequently across both model groups."
        ),
        "measure": (
            "Distinct annotations by model; labels also report affected "
            "documents. Candidates recur across documents, reviewers and archives. "
            "Axis ranges differ between charts to preserve readability."
        ),
        "x_label": "Distinct reviewer annotations",
    }
}


README_TEXT = f"""# 06 · Annotated forms that could inform the prompt

## Purpose

This section combines the token or phrase marked by a reviewer with its error
category and sub-rule. It therefore shows both *what* was marked and *why* the
reviewer believed it was wrong.

## Figure selection

An annotated form is a cross-context prompt candidate for one model when it occurs
in at least {MIN_DOCUMENTS} documents, {MIN_REVIEWERS} reviewer packets and
{MIN_ARCHIVES} archives. Candidates are ranked by distinct annotation count
and retained until at least {CUMULATIVE_COVERAGE:.0%} of qualifying annotations
is represented. Ties at the cutoff are included. Bar-end labels show
`annotations · documents`.

Selection is based on frequency, but the selected forms are displayed
alphabetically so individual forms are easy to find. Beneath each bold italic
form, the figure shows the category–sub-rule combination that reviewers
assigned to that form most frequently across both model groups. This is a
description of the reviewer labels, not a claim that the label is correct.

## Outputs

- `annotated_form_summary.csv` contains every annotated form and its annotation,
  document, archive and reviewer coverage by model. It also records the most
  commonly assigned rule and all other rules assigned to that form.
- `token_examples_for_review.csv` contains individual examples with modernised
  context and document metadata for adjudication.
- Three numbered `prompt_candidate_forms_and_rules` figures divide the
  alphabetically ordered forms into approximately equal thirds.

Each figure uses an x-axis range suited to the values in that alphabetical
section. This keeps shorter bars readable; use the printed annotation counts,
rather than apparent bar length across different figures, for comparisons.

## Caution

These are reviewer-assigned apparent errors, not automatically genuine model
errors. Inspect the example table alongside the source transcription before
turning a pattern into a prompt instruction or example. Annotation offsets
belong to the modernised text and cannot safely slice the source text at the
same positions.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_annotated_text(value):
    """Group equivalent forms without removing accents or punctuation."""

    if pd.isna(value):
        return MISSING_TEXT_LABEL
    text = unicodedata.normalize("NFC", str(value))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text if text else MISSING_TEXT_LABEL


def join_examples(values, limit=5):
    """Join a few distinct values for an inspectable CSV cell."""

    examples = []
    for value in values:
        if pd.isna(value):
            continue
        value = str(value)
        if value not in examples:
            examples.append(value)
        if len(examples) == limit:
            break
    return " | ".join(examples)


def resolve_project_path(value):
    """Resolve a path stored relative to the project directory."""

    if pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_DIR / path


def read_text_if_available(value):
    """Read one text file, returning a missing value if it is unavailable."""

    path = resolve_project_path(value)
    if path is None or not path.is_file():
        return pd.NA
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.NA


def modernised_context(text, start, end, window=CONTEXT_CHARACTERS):
    """Return a compact excerpt around an annotation in the modernised text."""

    if pd.isna(text) or pd.isna(start) or pd.isna(end):
        return pd.NA
    start, end = int(start), int(end)
    if start < 0 or end < start or start > len(text):
        return pd.NA
    left, right = max(0, start - window), min(len(text), end + window)
    before = re.sub(r"\s+", " ", text[left:start]).strip()
    marked = re.sub(r"\s+", " ", text[start:end]).strip()
    after = re.sub(r"\s+", " ", text[end:right]).strip()
    prefix, suffix = ("…" if left else ""), ("…" if right < len(text) else "")
    return f"{prefix}{before} [[{marked}]] {after}{suffix}".strip()


def retain_until_coverage(summary_df):
    """Retain ranked candidates until the stated evidence coverage is met."""

    if summary_df.empty:
        return summary_df.copy()
    ranked = summary_df.sort_values(
        ["distinct_annotations", "affected_documents", "display_label"],
        ascending=[False, False, True],
    ).copy()
    ranked["cumulative_share"] = (
        ranked["distinct_annotations"].cumsum()
        / ranked["distinct_annotations"].sum()
    )
    cutoff_index = ranked.index[
        ranked["cumulative_share"].ge(CUMULATIVE_COVERAGE)
    ][0]
    cutoff_count = ranked.loc[cutoff_index, "distinct_annotations"]
    return ranked.loc[ranked["distinct_annotations"].ge(cutoff_count)].copy()


# ---------------------------------------------------------------------------
# Load and validate the assembled analysis tables
# ---------------------------------------------------------------------------

check_required_files(
    [DOCUMENT_ANALYSIS_PATH, ANNOTATION_ANALYSIS_PATH],
    preceding_command="python scripts/construct_tables.py",
)
documents_df = pd.read_csv(DOCUMENT_ANALYSIS_PATH, dtype={"filename_stem": "string"})
annotations_df = pd.read_csv(
    ANNOTATION_ANALYSIS_PATH,
    dtype={
        "filename_stem": "string",
        "annotated_text": "string",
        "effective_error_category": "string",
        "effective_subrule": "string",
        "modernised_txt_path": "string",
        "pre_modernisation_txt_path": "string",
    },
)

required_document_columns = {
    "filename_stem", "model", "archive", "reviewer_name"
}
required_annotation_columns = {
    "filename_stem", "annotation_id", "annotated_text", "start", "end",
    "include_in_analysis", "effective_error_category", "effective_subrule",
    "modernised_txt_path", "pre_modernisation_txt_path",
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

# Use the document table as the single source for experimental metadata.
included_df = included_df.drop(
    columns=[c for c in ("model", "archive", "reviewer_name") if c in included_df]
).merge(
    documents_df[["filename_stem", "model", "archive", "reviewer_name"]],
    on="filename_stem",
    how="left",
    validate="many_to_one",
)
if included_df["model"].isna().any():
    raise ValueError("Some included annotations lack matching document metadata.")

included_df["normalised_annotated_text"] = included_df["annotated_text"].map(
    normalise_annotated_text
)
included_df["effective_error_category"] = included_df[
    "effective_error_category"
].fillna("Unmapped category")
included_df["effective_subrule"] = included_df["effective_subrule"].fillna(
    NO_SUBRULE_LABEL
)


# ---------------------------------------------------------------------------
# Summarise distinct form-rule assignments
# ---------------------------------------------------------------------------

identity_columns = [
    "model", "filename_stem", "annotation_id", "normalised_annotated_text",
    "effective_error_category", "effective_subrule",
]
assignment_df = included_df.drop_duplicates(subset=identity_columns).copy()
group_columns = [
    "model", "normalised_annotated_text", "effective_error_category",
    "effective_subrule",
]

summary_df = (
    assignment_df.groupby(group_columns, dropna=False, sort=False)
    .agg(
        distinct_annotations=("annotation_id", "size"),
        affected_documents=("filename_stem", "nunique"),
        affected_archives=("archive", "nunique"),
        affected_reviewers=("reviewer_name", "nunique"),
        example_surface_forms=("annotated_text", join_examples),
        example_documents=("filename_stem", join_examples),
        archives=("archive", join_examples),
        reviewers=("reviewer_name", join_examples),
    )
    .reset_index()
)
summary_df["cross_context_prompt_candidate"] = (
    summary_df["affected_documents"].ge(MIN_DOCUMENTS)
    & summary_df["affected_archives"].ge(MIN_ARCHIVES)
    & summary_df["affected_reviewers"].ge(MIN_REVIEWERS)
)
summary_df["display_label"] = (
    summary_df["normalised_annotated_text"] + " · "
    + summary_df["effective_error_category"] + " — "
    + summary_df["effective_subrule"]
)
summary_df = summary_df.sort_values(
    ["cross_context_prompt_candidate", "distinct_annotations",
     "affected_documents", "display_label", "model"],
    ascending=[False, False, False, True, True],
).reset_index(drop=True)
summary_df.insert(0, "form_rule_rank", range(1, len(summary_df) + 1))

# Build the form-level table used by the figure. An annotation that carries
# more than one rule assignment is counted only once in the form total.
token_identity = [
    "model", "filename_stem", "annotation_id", "normalised_annotated_text"
]
token_annotation_df = assignment_df.drop_duplicates(subset=token_identity)
token_group = ["model", "normalised_annotated_text"]
token_summary_df = (
    token_annotation_df.groupby(token_group, dropna=False, sort=False)
    .agg(
        distinct_annotations=("annotation_id", "size"),
        affected_documents=("filename_stem", "nunique"),
        affected_archives=("archive", "nunique"),
        affected_reviewers=("reviewer_name", "nunique"),
        example_surface_forms=("annotated_text", join_examples),
        example_documents=("filename_stem", join_examples),
        archives=("archive", join_examples),
        reviewers=("reviewer_name", join_examples),
    )
    .reset_index()
)

# Rank rule assignments within each model/form. The first row is the most
# commonly assigned rule; the complete list remains in all_rule_assignments.
rule_counts_df = (
    assignment_df.groupby(
        token_group + ["effective_error_category", "effective_subrule"],
        dropna=False,
    )
    .size()
    .rename("rule_assignment_count")
    .reset_index()
    .sort_values(
        token_group
        + ["rule_assignment_count", "effective_error_category", "effective_subrule"],
        ascending=[True, True, False, True, True],
    )
)
rule_counts_df["rule_label"] = (
    rule_counts_df["effective_error_category"]
    + " — "
    + rule_counts_df["effective_subrule"]
)
dominant_rules_df = (
    rule_counts_df.drop_duplicates(subset=token_group, keep="first")
    [token_group + ["effective_error_category", "effective_subrule", "rule_assignment_count"]]
    .rename(
        columns={
            "effective_error_category": "most_common_error_category",
            "effective_subrule": "most_common_subrule",
            "rule_assignment_count": "most_common_rule_assignments",
        }
    )
)
all_rules_df = (
    rule_counts_df.groupby(token_group, dropna=False)["rule_label"]
    .agg(lambda values: " | ".join(dict.fromkeys(values)))
    .rename("all_rule_assignments")
    .reset_index()
)
token_summary_df = token_summary_df.merge(
    dominant_rules_df, on=token_group, how="left", validate="one_to_one"
).merge(
    all_rules_df, on=token_group, how="left", validate="one_to_one"
)
token_summary_df["cross_context_prompt_candidate"] = (
    token_summary_df["affected_documents"].ge(MIN_DOCUMENTS)
    & token_summary_df["affected_archives"].ge(MIN_ARCHIVES)
    & token_summary_df["affected_reviewers"].ge(MIN_REVIEWERS)
)
token_summary_df["display_label"] = token_summary_df["normalised_annotated_text"]
token_summary_df = token_summary_df.sort_values(
    ["cross_context_prompt_candidate", "distinct_annotations",
     "affected_documents", "normalised_annotated_text", "model"],
    ascending=[False, False, False, True, True],
).reset_index(drop=True)
token_summary_df.insert(0, "annotated_form_rank", range(1, len(token_summary_df) + 1))


# ---------------------------------------------------------------------------
# Preserve individual examples and extract modernised-text context
# ---------------------------------------------------------------------------

text_cache = {
    path: read_text_if_available(path)
    for path in assignment_df["modernised_txt_path"].dropna().unique()
}
assignment_df["modernised_text"] = assignment_df["modernised_txt_path"].map(
    text_cache
)
assignment_df["modernised_context"] = assignment_df.apply(
    lambda row: modernised_context(
        row["modernised_text"], row["start"], row["end"]
    ),
    axis=1,
)
example_columns = [
    "model", "filename_stem", "archive", "reviewer_name", "annotation_id",
    "annotated_text", "normalised_annotated_text", "effective_error_category",
    "effective_subrule", "start", "end", "modernised_context",
    "modernised_txt_path", "pre_modernisation_txt_path",
]
examples_df = assignment_df[example_columns].merge(
    summary_df[group_columns + [
        "distinct_annotations", "affected_documents", "affected_archives",
        "affected_reviewers", "cross_context_prompt_candidate",
    ]],
    on=group_columns,
    how="left",
    validate="many_to_one",
).sort_values(
    ["cross_context_prompt_candidate", "distinct_annotations", "model",
     "normalised_annotated_text", "filename_stem", "annotation_id"],
    ascending=[False, False, True, True, True, True],
)


# ---------------------------------------------------------------------------
# Select plotted candidates and replace this stage's previous outputs
# ---------------------------------------------------------------------------

candidates_df = token_summary_df.loc[
    token_summary_df["cross_context_prompt_candidate"]
].copy()
coverage_selection_df = retain_until_coverage(candidates_df)

# Coverage is calculated from model-specific rows, but once a form-rule label
# is selected, include every qualifying model row for that label. This ensures
# the figure compares like with like rather than hiding the other model merely
# because its row fell immediately below the cumulative cutoff.
selected_labels = coverage_selection_df["display_label"].unique()
# Once a form has qualified through at least one model, show any observations
# for the other model too. The recurrence rule determines which forms enter the
# figure; it should not hide a legitimate cross-model comparison afterwards.
plotted_df = token_summary_df.loc[
    token_summary_df["display_label"].isin(selected_labels)
].copy()

if (SECTION_OUTPUT_DIR.name != "06_token_failure_analysis"
        or SECTION_OUTPUT_DIR.parent != ANALYSIS_OUTPUT_DIR):
    raise RuntimeError(f"Unsafe output directory: {SECTION_OUTPUT_DIR}")
if SECTION_OUTPUT_DIR.exists():
    shutil.rmtree(SECTION_OUTPUT_DIR)
tables_directory, figures_directory = create_output_folders(SECTION_OUTPUT_DIR)
SECTION_README_PATH.write_text(README_TEXT, encoding="utf-8")
save_table(token_summary_df, tables_directory / "annotated_form_summary.csv")
save_table(examples_df, tables_directory / "token_examples_for_review.csv")


# ---------------------------------------------------------------------------
# Create one prompt-focused comparison figure
# ---------------------------------------------------------------------------

if plotted_df.empty:
    print(
        "\nNo annotated form met the cross-context recurrence rules. "
        "The tables and README were still created."
    )
else:
    apply_plot_style()
    models = sorted(
        str(model) for model in token_summary_df["model"].dropna().unique()
    )
    model_colours = model_colour_map(models)

    # Alphabetical order makes a specific token easy to locate. The ordered
    # list is divided into three separate figures rather than squeezed into
    # three narrow panels on one canvas.
    alphabetical_labels = sorted(
        plotted_df["display_label"].drop_duplicates(), key=str.casefold
    )
    count_pivot = plotted_df.pivot_table(
        index="display_label", columns="model", values="distinct_annotations",
        aggfunc="sum", fill_value=0,
    ).reindex(alphabetical_labels)
    document_pivot = plotted_df.pivot_table(
        index="display_label", columns="model", values="affected_documents",
        aggfunc="max", fill_value=0,
    ).reindex(alphabetical_labels)

    # Give each form one explanatory rule label. When reviewers assigned more
    # than one rule, this is the most frequent assignment across both models;
    # every alternative remains visible in token_examples_for_review.csv.
    selected_assignments_df = assignment_df.loc[
        assignment_df["normalised_annotated_text"].isin(alphabetical_labels)
    ]
    overall_rule_counts_df = (
        selected_assignments_df.groupby(
            ["normalised_annotated_text", "effective_error_category",
             "effective_subrule"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["normalised_annotated_text", "count", "effective_error_category",
             "effective_subrule"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates("normalised_annotated_text")
    )
    overall_rule_counts_df["rule_label"] = (
        overall_rule_counts_df["effective_error_category"]
        + " — "
        + overall_rule_counts_df["effective_subrule"]
    )
    rule_label_map = overall_rule_counts_df.set_index(
        "normalised_annotated_text"
    )["rule_label"].to_dict()

    figure_count = min(3, len(alphabetical_labels))
    label_chunks = [
        list(chunk)
        for chunk in np.array_split(
            np.array(alphabetical_labels, dtype=object), figure_count
        )
        if len(chunk)
    ]
    for language in OUTPUT_LANGUAGES:
        if language not in CHART_TEXT:
            raise ValueError(f"No chart wording supplied for: {language}.")
        text = CHART_TEXT[language]
        language_directory = figures_directory / language
        for figure_number, chunk in enumerate(label_chunks, start=1):
            # The wide left margin forms a compact two-column label area:
            # annotated form first, then its most frequently assigned rule.
            # A taller canvas gives every row enough vertical separation.
            fig, ax = plt.subplots(figsize=(17.5, 17))
            fig.subplots_adjust(top=0.75, bottom=0.13, left=0.52, right=0.97)

            # Matplotlib draws the final horizontal-bar category at the top, so
            # reverse each alphabetical chunk to read A-to-Z from top to bottom.
            panel_labels = list(reversed(chunk))
            y_positions = np.arange(len(panel_labels), dtype=float)
            group_height = 0.72
            bar_height = group_height / max(len(models), 1)

            for position, model in enumerate(models):
                counts = count_pivot.reindex(panel_labels).get(
                    model, pd.Series(0, index=panel_labels)
                )
                documents = document_pivot.reindex(panel_labels).get(
                    model, pd.Series(0, index=panel_labels)
                )
                offsets = (
                    y_positions - group_height / 2 + bar_height / 2
                    + position * bar_height
                )
                bars = ax.barh(
                    offsets, counts.to_numpy(), height=bar_height * 0.82,
                    color=model_colours[model], label=model, zorder=3,
                )
                for bar, count, document_count in zip(
                    bars, counts.to_numpy(), documents.to_numpy()
                ):
                    if count > 0:
                        ax.annotate(
                            f"{int(count)} · {int(document_count)} docs",
                            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                            xytext=(4, 0), textcoords="offset points", va="center",
                            fontsize=5.8, color=COLOURS["text"],
                        )

            ax.set_title(
                f"Alphabetical section {figure_number} of {len(label_chunks)}: "
                f"{chunk[0]}–{chunk[-1]}",
                loc="left", fontsize=10, fontweight="bold", pad=10,
            )
            ax.set_yticks(y_positions)
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

            # Headings explain the two pieces of information aligned beside
            # every bar group. Coordinates are relative to the plotting area.
            ax.text(
                -0.90, 1.012, "Annotated form",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.2, fontweight="bold", color=COLOURS["text"],
                clip_on=False,
            )
            ax.text(
                -0.62, 1.012, "Reviewer label assigned most frequently",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.2, fontweight="bold", color=COLOURS["text"],
                clip_on=False,
            )

            # Keep the form and reviewer label on one row. Bold italics make
            # the actual annotated text distinguishable at a glance.
            for y_position, label in zip(y_positions, panel_labels):
                ax.text(
                    -0.90, y_position, label,
                    transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=7.4,
                    fontweight="bold",
                    fontstyle="italic",
                    color=COLOURS["text"],
                    clip_on=False,
                )
                ax.text(
                    -0.62, y_position,
                    rule_label_map.get(label, "Rule unavailable"),
                    transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=6.4,
                    color=COLOURS["muted_text"],
                    clip_on=False,
                )
            ax.set_xlabel(text["x_label"])
            # Scale each alphabetical chart independently. A single global
            # scale made most bars illegibly short because a few forms had
            # much larger counts than the rest.
            chart_maximum = float(
                count_pivot.reindex(chunk).to_numpy().max()
            )
            ax.set_xlim(0, chart_maximum * 1.25 if chart_maximum else 1)
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            handles, legend_labels = ax.get_legend_handles_labels()
            fig.legend(
                handles, legend_labels, frameon=False, loc="lower center",
                bbox_to_anchor=(0.5, 0.065), ncol=2,
            )

            add_chart_header(
                fig, title=text["title"], description=text["description"],
                measure=text["measure"],
            )
            add_figure_note(
                fig,
                "Bold italics identify the annotated form. The adjacent reviewer label is the category–sub-rule combination assigned most frequently across both model groups. Bar labels show annotations · affected documents. Axis ranges differ between charts.",
            )
            save_figure(
                fig,
                language_directory,
                f"prompt_candidate_forms_and_rules_{figure_number}_of_"
                f"{len(label_chunks)}",
            )


print(f"\nAnnotated forms summarised by model: {len(token_summary_df)}")
print(f"Cross-context prompt candidates: {len(candidates_df)}")
print(
    "Unique annotated forms selected for the figures: "
    f"{plotted_df['display_label'].nunique()}"
)
print(f"Model-form rows displayed in the figure: {len(plotted_df)}")
print(f"Individual evidence rows saved: {len(examples_df)}")
print(f"Tables saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
