"""Examine archive patterns and explicitly repeated document families.

Stage 3 identifies individual documents with unusually high annotation counts
or rates. This section asks whether those documents cluster within particular
source archives or within repeated versions of the same underlying document.

Archive rates are pooled: annotations and modernised tokens are summed within
the archive before annotations per 1,000 tokens are calculated. This prevents
short documents from having the same influence as long documents when an
archive-level rate is calculated. The figure includes every represented
archive. Most archives contain only three or four sampled documents, reflecting
the stratified sampling design, so the rates are descriptive signals for
follow-up rather than stable archive rankings.

The archive table also contains a simple ``archive_prefix_group`` summary. The
group is the text before the first underscore in the archive name, converted to
upper case. For example, ``bnf_267`` and ``bnf_032`` both become ``BNF``. This
allows an apparent concentration of BnF documents to be checked collectively,
but the automatically derived groups should be confirmed against collection
knowledge before formal reporting.

Repeated-document families use the conservative identifier created in Stage 3:
only a final ``_duplicated_<number>`` suffix is removed. They are retained in a
table for completeness, but no figure is produced because the small number of
repeated families in this sample does not support a revealing visual comparison.

The script produces two tables and one figure:

    analysis_outputs/04_archive_and_family_analysis/
        ARCHIVE_AND_FAMILY_ANALYSIS_README.md
        tables/archive_context_summary.csv
        tables/document_family_summary.csv
        figures/en/archive_annotation_rates.png and .svg
"""

import os
from pathlib import Path
import shutil
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
    save_figure,
    save_table,
)


# ---------------------------------------------------------------------------
# Set up the input and output paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(
    os.environ.get("MODERNISATION_PROJECT_DIR", "/workspaces/modernisation")
)
DOCUMENT_DIAGNOSTICS_PATH = (
    PROJECT_DIR
    / "analysis_outputs"
    / "03_document_diagnostics"
    / "tables"
    / "document_diagnostics.csv"
)

ANALYSIS_OUTPUT_DIR = PROJECT_DIR / "analysis_outputs"
SECTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "04_archive_and_family_analysis"
SECTION_README_PATH = (
    SECTION_OUTPUT_DIR / "ARCHIVE_AND_FAMILY_ANALYSIS_README.md"
)

# ---------------------------------------------------------------------------
# Define the visible chart wording
# ---------------------------------------------------------------------------

CHART_TEXT = {
    "en": {
        "archives": {
            "title": "How did annotation rates vary across represented archives?",
            "description": (
                "This comparison shows whether the document-level patterns "
                "identified earlier were concentrated in particular archives."
            ),
            "measure": (
                "Pooled distinct annotations per 1,000 modernised tokens; "
                "all represented archives are shown with their reviewed-document counts."
            ),
            "x_label": "Annotations per 1,000 tokens",
        },
    }
}


README_TEXT = """# 04 · Archive and repeated-family analysis

## What this section is trying to show

This section asks whether the documents flagged in Stage 3 form recurring
patterns within particular archives or repeated versions of the same source
document.

## Archive analysis

Archive rates are calculated by summing distinct included annotations and
modernised tokens within each archive, then reporting annotations per 1,000
tokens. The figure shows every represented archive. Most archives contain only
three or four sampled documents because the sample was distributed as evenly
as possible across the 47 archives. Their rates should therefore be treated as
descriptive signals for closer examination, not as reliable archive rankings.

The table also provides an automatically derived archive-prefix group. This is
the part of the archive name before its first underscore, converted to upper
case: for example, `bnf_267` and `bnf_032` both become `BNF`. It is included so
that apparent concentrations such as BnF can be checked against their reviewed
document and token exposure. These prefix groups should be verified before
they are described as institutions or collections in final reporting.

## Repeated-document families

The family identifier comes from Stage 3. Only a final suffix matching
`_duplicated_<number>` is removed. All other filename stems remain unchanged.

Every family is retained in the table and marked as either repeated or a
singleton. No family figure is produced because only a very small number of
repeated families is available. Plotting them would risk presenting a sparse
descriptive check as a general result.

## Input

- `analysis_outputs/03_document_diagnostics/tables/document_diagnostics.csv`:
  reviewed documents, annotation burden, archive, model, reviewer and family.

## Tables produced by this script

- `archive_context_summary.csv`: archive-level and archive-prefix-group
  exposure, annotation rates, model mix, reviewer coverage and flagged-document
  counts.
- `document_family_summary.csv`: every family, including its members, exposure,
  annotation rate, model coverage and whether it is genuinely repeated.

## Figures produced by this script

- `archive_annotation_rates`: all represented archives, ranked by pooled
  annotation rate across two landscape panels.
The archive figure is saved as both PNG and SVG.

## Interpretation

These results describe reviewer feedback rather than confirmed model errors.
An archive or family pattern may reflect textual characteristics, model
allocation, reviewer behaviour or some combination of these. The table retains
model and reviewer coverage so those explanations can be investigated next.
"""


# ---------------------------------------------------------------------------
# Check and load the Stage 3 diagnostic table
# ---------------------------------------------------------------------------

check_required_files(
    [DOCUMENT_DIAGNOSTICS_PATH],
    preceding_command="python scripts/analysis/03_document_diagnostics.py",
)

documents_df = pd.read_csv(
    DOCUMENT_DIAGNOSTICS_PATH,
    dtype={"filename_stem": "string", "document_family": "string"},
)

required_columns = {
    "filename_stem",
    "archive",
    "model",
    "reviewer_name",
    "n_modernised_tokens",
    "n_included_annotations",
    "n_included_assignments",
    "included_annotations_per_1000_tokens",
    "document_family",
    "repeated_document_family",
    "selected_for_review",
}

missing_columns = required_columns - set(documents_df.columns)
if missing_columns:
    raise ValueError(
        "document_diagnostics.csv is missing required columns: "
        f"{sorted(missing_columns)}."
    )

if documents_df["filename_stem"].duplicated().any():
    raise ValueError(
        "document_diagnostics.csv must contain one row per reviewed document."
    )

documents_df["repeated_document_family"] = coerce_boolean(
    documents_df["repeated_document_family"]
)
documents_df["selected_for_review"] = coerce_boolean(
    documents_df["selected_for_review"]
)

for column in (
    "n_modernised_tokens",
    "n_included_annotations",
    "n_included_assignments",
    "included_annotations_per_1000_tokens",
):
    documents_df[column] = pd.to_numeric(documents_df[column], errors="coerce")

invalid_denominator = (
    documents_df["n_modernised_tokens"].isna()
    | documents_df["n_modernised_tokens"].le(0)
)
if invalid_denominator.any():
    raise ValueError(
        "Stage 3 contains a document without a positive modernised-token "
        "count. Rebuild Stage 3 before running this analysis."
    )

models = sorted(str(model) for model in documents_df["model"].dropna().unique())
if len(models) != 2:
    raise ValueError("The archive comparison expects exactly two models.")


# ---------------------------------------------------------------------------
# Derive a simple archive-prefix grouping for broader concentration checks
# ---------------------------------------------------------------------------

def derive_archive_prefix_group(archive):
    """Return the upper-case text before the first archive-name underscore."""

    archive_text = str(archive).strip()
    if not archive_text:
        return "MISSING"
    return archive_text.split("_", maxsplit=1)[0].upper()


documents_df["archive_prefix_group"] = documents_df["archive"].map(
    derive_archive_prefix_group
)


# ---------------------------------------------------------------------------
# Replace this section's earlier outputs
# ---------------------------------------------------------------------------

if (
    SECTION_OUTPUT_DIR.name != "04_archive_and_family_analysis"
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
# Summarise an archive or archive-prefix group
# ---------------------------------------------------------------------------

def summarise_archive_context(group_df):
    """Return exposure, feedback and allocation measures for one group."""

    total_tokens = group_df["n_modernised_tokens"].sum()
    total_annotations = group_df["n_included_annotations"].sum()
    flagged_documents = group_df["selected_for_review"].fillna(False).sum()

    summary = {
        "reviewed_documents": len(group_df),
        "modernised_tokens": int(total_tokens),
        "included_annotations": int(total_annotations),
        "included_assignments": int(group_df["n_included_assignments"].sum()),
        "pooled_annotations_per_1000_tokens": (
            total_annotations / total_tokens * 1000
        ),
        "median_document_annotation_rate": group_df[
            "included_annotations_per_1000_tokens"
        ].median(),
        "flagged_documents": int(flagged_documents),
        "flagged_documents_pct": flagged_documents / len(group_df) * 100,
        "distinct_reviewers": group_df["reviewer_name"].nunique(dropna=True),
        "distinct_models": group_df["model"].nunique(dropna=True),
    }

    # Retain the realised model mix and model-specific rate in stable numbered
    # columns. Model display names remain values rather than becoming headings.
    for model_position, model in enumerate(models, start=1):
        model_df = group_df.loc[group_df["model"].astype(str).eq(model)]
        model_tokens = model_df["n_modernised_tokens"].sum()
        model_annotations = model_df["n_included_annotations"].sum()

        summary[f"model_{model_position}"] = model
        summary[f"model_{model_position}_documents"] = len(model_df)
        summary[f"model_{model_position}_tokens"] = int(model_tokens)
        summary[f"model_{model_position}_annotations"] = int(model_annotations)
        summary[f"model_{model_position}_rate_per_1000_tokens"] = (
            model_annotations / model_tokens * 1000
            if model_tokens > 0
            else pd.NA
        )

    return pd.Series(summary)


# ---------------------------------------------------------------------------
# Build one table containing both archive summary levels
# ---------------------------------------------------------------------------

archive_rows = []

for summary_level, grouping_column in (
    ("archive_prefix_group", "archive_prefix_group"),
    ("archive", "archive"),
):
    for group_name, group_df in documents_df.groupby(
        grouping_column,
        dropna=False,
        sort=True,
    ):
        summary = summarise_archive_context(group_df).to_dict()
        summary["summary_level"] = summary_level
        summary["group_name"] = group_name
        archive_rows.append(summary)

archive_context_summary_df = pd.DataFrame(archive_rows)

archive_column_order = [
    "summary_level",
    "group_name",
    "reviewed_documents",
    "modernised_tokens",
    "included_annotations",
    "included_assignments",
    "pooled_annotations_per_1000_tokens",
    "median_document_annotation_rate",
    "flagged_documents",
    "flagged_documents_pct",
    "distinct_reviewers",
    "distinct_models",
]
for model_position in range(1, len(models) + 1):
    archive_column_order.extend(
        [
            f"model_{model_position}",
            f"model_{model_position}_documents",
            f"model_{model_position}_tokens",
            f"model_{model_position}_annotations",
            f"model_{model_position}_rate_per_1000_tokens",
        ]
    )

archive_context_summary_df = archive_context_summary_df[
    archive_column_order
].sort_values(
    ["summary_level", "pooled_annotations_per_1000_tokens"],
    ascending=[True, False],
)

save_table(
    archive_context_summary_df,
    tables_directory / "archive_context_summary.csv",
)


# ---------------------------------------------------------------------------
# Build the complete document-family summary
# ---------------------------------------------------------------------------

family_rows = []

for document_family, family_df in documents_df.groupby(
    "document_family",
    dropna=False,
    sort=True,
):
    total_tokens = family_df["n_modernised_tokens"].sum()
    total_annotations = family_df["n_included_annotations"].sum()

    row = {
        "document_family": document_family,
        "family_type": "repeated" if len(family_df) >= 2 else "singleton",
        "reviewed_documents": len(family_df),
        "member_documents": " | ".join(
            sorted(family_df["filename_stem"].astype(str))
        ),
        "distinct_archives": family_df["archive"].nunique(dropna=True),
        "distinct_reviewers": family_df["reviewer_name"].nunique(dropna=True),
        "models_represented": family_df["model"].nunique(dropna=True),
        "modernised_tokens": int(total_tokens),
        "included_annotations": int(total_annotations),
        "included_assignments": int(
            family_df["n_included_assignments"].sum()
        ),
        "pooled_annotations_per_1000_tokens": (
            total_annotations / total_tokens * 1000
        ),
        "minimum_document_annotation_rate": family_df[
            "included_annotations_per_1000_tokens"
        ].min(),
        "median_document_annotation_rate": family_df[
            "included_annotations_per_1000_tokens"
        ].median(),
        "maximum_document_annotation_rate": family_df[
            "included_annotations_per_1000_tokens"
        ].max(),
        "flagged_documents": int(
            family_df["selected_for_review"].fillna(False).sum()
        ),
    }

    for model_position, model in enumerate(models, start=1):
        model_df = family_df.loc[family_df["model"].astype(str).eq(model)]
        row[f"model_{model_position}"] = model
        row[f"model_{model_position}_documents"] = len(model_df)

    family_rows.append(row)

document_family_summary_df = pd.DataFrame(family_rows).sort_values(
    ["family_type", "pooled_annotations_per_1000_tokens"],
    ascending=[True, False],
).reset_index(drop=True)

save_table(
    document_family_summary_df,
    tables_directory / "document_family_summary.csv",
)


# ---------------------------------------------------------------------------
# Prepare the archive figure data
# ---------------------------------------------------------------------------

archive_plot_df = archive_context_summary_df.loc[
    archive_context_summary_df["summary_level"].eq("archive")
].copy()
archive_plot_df = archive_plot_df.sort_values(
    "pooled_annotations_per_1000_tokens",
    ascending=True,
)

repeated_family_names = set(
    document_family_summary_df.loc[
        document_family_summary_df["family_type"].eq("repeated"),
        "document_family",
    ]
)
# ---------------------------------------------------------------------------
# Generate the archive figure
# ---------------------------------------------------------------------------

apply_plot_style()

for language in OUTPUT_LANGUAGES:
    if language not in CHART_TEXT:
        raise ValueError(
            f"No archive-and-family chart wording has been supplied for: {language}."
        )

    language_directory = figures_directory / language
    text = CHART_TEXT[language]

    # Show every represented archive rather than applying a minimum document
    # threshold. The ranked list is split over two side-by-side panels so
    # that all labels remain legible in a landscape figure. Read down the left
    # panel first, followed by the right panel.
    if not archive_plot_df.empty:
        ranked_archives_df = archive_plot_df.sort_values(
            "pooled_annotations_per_1000_tokens",
            ascending=False,
        ).reset_index(drop=True)
        panel_size = (len(ranked_archives_df) + 1) // 2
        archive_panels = (
            ranked_archives_df.iloc[:panel_size],
            ranked_archives_df.iloc[panel_size:],
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharex=True)
        fig.subplots_adjust(top=0.70, bottom=0.14, left=0.20, right=0.96, wspace=0.47)

        for panel_number, (ax, panel_df) in enumerate(
            zip(axes, archive_panels),
            start=1,
        ):
            if panel_df.empty:
                ax.set_axis_off()
                continue

            display_labels = panel_df.apply(
                lambda row: (
                    f"{row['group_name']}  "
                    f"(n={int(row['reviewed_documents'])}; "
                    f"flagged={int(row['flagged_documents'])})"
                ),
                axis=1,
            ).tolist()
            ax.barh(
                display_labels,
                panel_df["pooled_annotations_per_1000_tokens"],
                color=COLOURS["gold"],
                alpha=0.82,
                edgecolor="none",
            )
            ax.invert_yaxis()
            ax.set_title(
                f"Ranked archives: panel {panel_number}",
                fontsize=10,
                fontweight="bold",
                loc="left",
                pad=10,
            )
            ax.set_xlabel(text["archives"]["x_label"])
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0, labelsize=7.5)

        add_chart_header(
            fig,
            **{
                key: text["archives"][key]
                for key in ("title", "description", "measure")
            },
        )
        add_figure_note(
            fig,
            "Read down the left panel, then the right. Most archive rates are based on only three or four sampled documents and are descriptive signals, not stable archive rankings.",
        )
        save_figure(fig, language_directory, "archive_annotation_rates")


# ---------------------------------------------------------------------------
# Print a brief completion report
# ---------------------------------------------------------------------------

print(f"\nReviewed documents summarised: {len(documents_df)}")
print(
    "Archive-prefix groups: "
    f"{documents_df['archive_prefix_group'].nunique()}"
)
print(f"Distinct archives: {documents_df['archive'].nunique()}")
print(
    "Archives shown in the figure: "
    f"{len(archive_plot_df)}"
)
print(f"Repeated document families: {len(repeated_family_names)}")
print("Repeated-family details retained in the table; no figure is produced.")

print(f"\nTables saved to: {tables_directory}")
print(f"Figures saved to: {figures_directory}")
print(f"README saved to: {SECTION_README_PATH}")
