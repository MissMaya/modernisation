"""Assemble the final document-level and annotation-level analysis tables.

This script combines the intermediate tables created by Stages 1 to 7 and
creates two final CSV files:

    outputs/document_analysis.csv
        One row for every expected document. This table retains documents with
        no recorded annotations and documents whose annotation data is missing.

    outputs/annotation_analysis.csv
        One row for every annotation–error-category–field assignment. It
        retains excluded assignments for audit purposes and includes the
        include_in_analysis flag created by Stage 6.

DOCUMENT-LEVEL COUNTS

A source annotation can produce more than one row in the annotation table when
it has two error categories or multiple fields. The document table therefore
keeps separate counts for:

    annotations
        Distinct annotation IDs recorded by the reviewer.

    assignments
        Flattened error-category and field assignments.

Counts beginning n_recorded_ include every reviewer-recorded assignment before
the Stage 6 inclusion decision. Counts beginning n_included_ include only rows
where include_in_analysis is True.

MISSING REVIEW DATA AND ZERO ANNOTATIONS

An available annotation JSON containing no records represents a completed
review in which no errors were recorded. Its annotation counts are set to zero.
If the annotation JSON is unavailable, the counts remain blank because the
review result is unknown. The review_data_status column records this distinction.

TOKEN-NORMALISED RATES

Rates per 1,000 modernised tokens are calculated only when annotation data is
available and n_modernised_tokens is greater than zero. Otherwise they remain
blank.
"""

# ---------------------------------------------------------------------------
# Import the required modules
# ---------------------------------------------------------------------------

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Set up input and output paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path("/workspaces/modernisation")
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DOCUMENT_MANIFEST_PATH = INTERMEDIATE_DIR / "document_manifest.csv"
FILE_MANIFEST_PATH = INTERMEDIATE_DIR / "file_manifest.csv"
REVIEWER_MANIFEST_PATH = INTERMEDIATE_DIR / "reviewer_manifest.csv"
RESOLVED_ANNOTATIONS_PATH = (
    INTERMEDIATE_DIR / "resolved_annotations.csv"
)
DOCUMENT_FEATURES_PATH = INTERMEDIATE_DIR / "document_features.csv"

DOCUMENT_OUTPUT_PATH = OUTPUT_DIR / "document_analysis.csv"
ANNOTATION_OUTPUT_PATH = OUTPUT_DIR / "annotation_analysis.csv"


# ---------------------------------------------------------------------------
# Check that all required intermediate tables exist
# ---------------------------------------------------------------------------

required_outputs = [
    (DOCUMENT_MANIFEST_PATH, "01_build_document_manifest.py"),
    (FILE_MANIFEST_PATH, "02_build_file_manifest.py"),
    (REVIEWER_MANIFEST_PATH, "04_build_reviewer_manifest.py"),
    (RESOLVED_ANNOTATIONS_PATH, "06_resolve_invalid_combinations.py"),
    (DOCUMENT_FEATURES_PATH, "07_build_document_features.py"),
]

for output_path, producing_script in required_outputs:
    if not output_path.is_file():
        raise SystemExit(
            f"{output_path.name} was not found in {INTERMEDIATE_DIR}. "
            f"Run {producing_script} first."
        )


# ---------------------------------------------------------------------------
# Load the intermediate tables
# ---------------------------------------------------------------------------

documents_df = pd.read_csv(
    DOCUMENT_MANIFEST_PATH,
    dtype = {"filename_stem": "string"},
)

file_manifest_df = pd.read_csv(
    FILE_MANIFEST_PATH,
    dtype = {
        "filename_stem": "string",
        "annotation_json_found": "boolean",
    },
)

reviewers_df = pd.read_csv(REVIEWER_MANIFEST_PATH)

resolved_annotations_df = pd.read_csv(
    RESOLVED_ANNOTATIONS_PATH,
    dtype = {
        "filename_stem": "string",
        "error_code": "string",
        "field_code": "string",
        "effective_error_code": "string",
        "effective_field_code": "string",
        "include_in_analysis": "boolean",
        "rule_combination_valid": "boolean",
    },
)

document_features_df = pd.read_csv(
    DOCUMENT_FEATURES_PATH,
    dtype = {"filename_stem": "string"},
)


# ---------------------------------------------------------------------------
# Standardise the reviewer-packet column name used by Stage 1
# ---------------------------------------------------------------------------

# Stage 1 retains the original Excel column name reviewer, although the values
# are packet numbers rather than reviewer names. Later tables call this column
# reviewer_packet. Accept either form so Stage 8 remains compatible if Stage 1
# is made more explicit in the future.
if "reviewer_packet" not in documents_df.columns:
    if "reviewer" not in documents_df.columns:
        raise ValueError(
            "document_manifest.csv must contain reviewer or reviewer_packet."
        )

    documents_df = documents_df.rename(
        columns = {"reviewer": "reviewer_packet"}
    )


# ---------------------------------------------------------------------------
# Validate the join keys before combining the tables
# ---------------------------------------------------------------------------

required_document_columns = {
    "filename_stem",
    "archive",
    "reviewer_packet",
    "model",
}

missing_document_columns = (
    required_document_columns - set(documents_df.columns)
)

if missing_document_columns:
    raise ValueError(
        "document_manifest.csv is missing required columns: "
        f"{sorted(missing_document_columns)}."
    )

for table_name, dataframe in [
    ("document_manifest.csv", documents_df),
    ("file_manifest.csv", file_manifest_df),
    ("document_features.csv", document_features_df),
]:
    if dataframe["filename_stem"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["filename_stem"].duplicated(keep = False),
            "filename_stem",
        ]
        raise ValueError(
            f"{table_name} contains duplicate filename_stem values:\n"
            f"{duplicates.to_string(index = False)}"
        )

required_reviewer_columns = {
    "reviewer_packet",
    "reviewer_name",
}

missing_reviewer_columns = (
    required_reviewer_columns - set(reviewers_df.columns)
)

if missing_reviewer_columns:
    raise ValueError(
        "reviewer_manifest.csv is missing required columns: "
        f"{sorted(missing_reviewer_columns)}."
    )

if reviewers_df["reviewer_packet"].duplicated().any():
    raise ValueError(
        "reviewer_manifest.csv contains more than one row for the same "
        "reviewer packet."
    )

required_resolved_columns = {
    "filename_stem",
    "reviewer_packet",
    "annotation_id",
    "include_in_analysis",
    "rule_combination_valid",
    "resolution_action",
}

missing_resolved_columns = (
    required_resolved_columns
    - set(resolved_annotations_df.columns)
)

if missing_resolved_columns:
    raise ValueError(
        "resolved_annotations.csv is missing required columns: "
        f"{sorted(missing_resolved_columns)}."
    )

required_feature_columns = {
    "filename_stem",
    "n_modernised_tokens",
}

missing_feature_columns = (
    required_feature_columns
    - set(document_features_df.columns)
)

if missing_feature_columns:
    raise ValueError(
        "document_features.csv is missing required columns: "
        f"{sorted(missing_feature_columns)}."
    )

# Use the same nullable integer type for packet numbers before joining tables
# produced from CSV files with slightly different type inference.
for dataframe in [
    documents_df,
    file_manifest_df,
    reviewers_df,
    resolved_annotations_df,
]:
    dataframe["reviewer_packet"] = pd.to_numeric(
        dataframe["reviewer_packet"],
        errors = "raise",
    ).astype("Int64")


# ---------------------------------------------------------------------------
# Build the annotation-level analysis table
# ---------------------------------------------------------------------------

# Join on both filename and packet so a misplaced annotation cannot silently
# inherit the metadata belonging to a document in another packet.
annotation_analysis_df = resolved_annotations_df.merge(
    documents_df[
        [
            "filename_stem",
            "reviewer_packet",
            "archive",
            "model",
        ]
    ],
    on = ["filename_stem", "reviewer_packet"],
    how = "left",
    validate = "many_to_one",
    indicator = "document_join",
)

unmatched_annotations = annotation_analysis_df.loc[
    annotation_analysis_df["document_join"].ne("both")
]

if not unmatched_annotations.empty:
    raise ValueError(
        "Some resolved annotations could not be matched to the document "
        "manifest using filename_stem and reviewer_packet:\n"
        + unmatched_annotations[
            ["filename_stem", "reviewer_packet"]
        ]
        .drop_duplicates()
        .to_string(index = False)
    )

annotation_analysis_df = annotation_analysis_df.drop(
    columns = ["document_join"]
)

annotation_analysis_df = annotation_analysis_df.merge(
    reviewers_df,
    on = "reviewer_packet",
    how = "left",
    validate = "many_to_one",
)

# Retain the text paths in the annotation-level table so later token/context
# analyses can reopen the corresponding pre-modernisation and modernised texts
# without performing another filename search.
annotation_analysis_df = annotation_analysis_df.merge(
    file_manifest_df[
        [
            "filename_stem",
            "reviewer_packet",
            "modernised_txt_path",
            "pre_modernisation_txt_path",
            "annotation_json_found",
            "modernised_txt_found",
            "pre_modernisation_txt_found",
        ]
    ],
    on = ["filename_stem", "reviewer_packet"],
    how = "left",
    validate = "many_to_one",
)

annotation_analysis_df = annotation_analysis_df.merge(
    document_features_df,
    on = "filename_stem",
    how = "left",
    validate = "many_to_one",
)


# ---------------------------------------------------------------------------
# Summarise the annotation rows for the document-level table
# ---------------------------------------------------------------------------

resolved_annotations_df["include_in_analysis"] = (
    resolved_annotations_df["include_in_analysis"].astype("boolean")
)

resolved_annotations_df["rule_combination_valid"] = (
    resolved_annotations_df["rule_combination_valid"].astype("boolean")
)

recorded_summary_df = (
    resolved_annotations_df
    .groupby("filename_stem", as_index = False)
    .agg(
        n_recorded_annotations = (
            "annotation_id",
            "nunique",
        ),
        n_recorded_assignments = (
            "annotation_id",
            "size",
        ),
        n_included_assignments = (
            "include_in_analysis",
            "sum",
        ),
    )
)

included_annotations_df = resolved_annotations_df.loc[
    resolved_annotations_df["include_in_analysis"]
]

included_annotation_counts_df = (
    included_annotations_df
    .groupby("filename_stem")["annotation_id"]
    .nunique()
    .rename("n_included_annotations")
    .reset_index()
)

recorded_summary_df = recorded_summary_df.merge(
    included_annotation_counts_df,
    on = "filename_stem",
    how = "left",
    validate = "one_to_one",
)

recorded_summary_df["n_included_annotations"] = (
    recorded_summary_df["n_included_annotations"]
    .fillna(0)
    .astype("Int64")
)

quality_summary_df = (
    resolved_annotations_df
    .assign(
        excluded_assignment = (
            ~resolved_annotations_df["include_in_analysis"]
        ),
        originally_invalid_assignment = (
            ~resolved_annotations_df["rule_combination_valid"]
        ),
        corrected_assignment = (
            resolved_annotations_df["resolution_action"]
            .eq("corrected")
        ),
        accepted_invalid_assignment = (
            resolved_annotations_df["resolution_action"]
            .eq("accepted")
            & ~resolved_annotations_df["rule_combination_valid"]
        ),
    )
    .groupby("filename_stem", as_index = False)
    .agg(
        n_excluded_assignments = (
            "excluded_assignment",
            "sum",
        ),
        n_originally_invalid_assignments = (
            "originally_invalid_assignment",
            "sum",
        ),
        n_corrected_assignments = (
            "corrected_assignment",
            "sum",
        ),
        n_accepted_invalid_assignments = (
            "accepted_invalid_assignment",
            "sum",
        ),
    )
)

annotation_summary_df = recorded_summary_df.merge(
    quality_summary_df,
    on = "filename_stem",
    how = "left",
    validate = "one_to_one",
)


# ---------------------------------------------------------------------------
# Build the document-level analysis table
# ---------------------------------------------------------------------------

document_analysis_df = documents_df.merge(
    file_manifest_df,
    on = ["filename_stem", "reviewer_packet"],
    how = "left",
    validate = "one_to_one",
)

document_analysis_df = document_analysis_df.merge(
    reviewers_df,
    on = "reviewer_packet",
    how = "left",
    validate = "many_to_one",
)

document_analysis_df = document_analysis_df.merge(
    document_features_df,
    on = "filename_stem",
    how = "left",
    validate = "one_to_one",
)

document_analysis_df = document_analysis_df.merge(
    annotation_summary_df,
    on = "filename_stem",
    how = "left",
    validate = "one_to_one",
)


# ---------------------------------------------------------------------------
# Distinguish completed zero-annotation reviews from unavailable review data
# ---------------------------------------------------------------------------

document_analysis_df["annotation_json_found"] = (
    document_analysis_df["annotation_json_found"].astype("boolean")
)

count_columns = [
    "n_recorded_annotations",
    "n_recorded_assignments",
    "n_included_annotations",
    "n_included_assignments",
    "n_excluded_assignments",
    "n_originally_invalid_assignments",
    "n_corrected_assignments",
    "n_accepted_invalid_assignments",
]

# Only completed reviews receive zeroes. Counts remain blank where the
# annotation JSON was unavailable because no review result can be inferred.
completed_review = document_analysis_df[
    "annotation_json_found"
].fillna(False)

for column in count_columns:
    document_analysis_df.loc[
        completed_review & document_analysis_df[column].isna(),
        column,
    ] = 0
    document_analysis_df[column] = (
        document_analysis_df[column].astype("Int64")
    )

document_analysis_df["review_data_status"] = (
    "annotation_data_unavailable"
)

document_analysis_df.loc[
    completed_review
    & document_analysis_df["n_recorded_annotations"].eq(0),
    "review_data_status",
] = "reviewed_no_annotations"

document_analysis_df.loc[
    completed_review
    & document_analysis_df["n_recorded_annotations"].gt(0),
    "review_data_status",
] = "reviewed_with_annotations"


# ---------------------------------------------------------------------------
# Calculate token-normalised document-level annotation rates
# ---------------------------------------------------------------------------

valid_rate_denominator = (
    completed_review
    & document_analysis_df["n_modernised_tokens"].notna()
    & document_analysis_df["n_modernised_tokens"].gt(0)
)

document_analysis_df["included_annotations_per_1000_tokens"] = (
    pd.Series(
        pd.NA,
        index = document_analysis_df.index,
        dtype = "Float64",
    )
)

document_analysis_df.loc[
    valid_rate_denominator,
    "included_annotations_per_1000_tokens",
] = (
    document_analysis_df.loc[
        valid_rate_denominator,
        "n_included_annotations",
    ]
    / document_analysis_df.loc[
        valid_rate_denominator,
        "n_modernised_tokens",
    ]
    * 1000
)

document_analysis_df["included_assignments_per_1000_tokens"] = (
    pd.Series(
        pd.NA,
        index = document_analysis_df.index,
        dtype = "Float64",
    )
)

document_analysis_df.loc[
    valid_rate_denominator,
    "included_assignments_per_1000_tokens",
] = (
    document_analysis_df.loc[
        valid_rate_denominator,
        "n_included_assignments",
    ]
    / document_analysis_df.loc[
        valid_rate_denominator,
        "n_modernised_tokens",
    ]
    * 1000
)


# ---------------------------------------------------------------------------
# Final safeguards and save both analytical tables
# ---------------------------------------------------------------------------

if len(document_analysis_df) != len(documents_df):
    raise ValueError(
        "The document-level join changed the number of expected documents."
    )

if len(annotation_analysis_df) != len(resolved_annotations_df):
    raise ValueError(
        "The annotation-level join changed the number of assignment rows."
    )

OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

document_analysis_df.to_csv(
    DOCUMENT_OUTPUT_PATH,
    index = False,
    encoding = "utf-8-sig",
)

annotation_analysis_df.to_csv(
    ANNOTATION_OUTPUT_PATH,
    index = False,
    encoding = "utf-8-sig",
)


# ---------------------------------------------------------------------------
# Print concise final checks
# ---------------------------------------------------------------------------

print(
    f"\nDocument-level rows: "
    f"{len(document_analysis_df)}"
)

print("\nDocuments by review-data status:")
print(
    document_analysis_df["review_data_status"]
    .value_counts(dropna = False)
    .to_string()
)

print(
    f"\nAnnotation-assignment rows: "
    f"{len(annotation_analysis_df)}"
)

print(
    f"Assignments included in analysis: "
    f"{annotation_analysis_df['include_in_analysis'].sum()}"
)

print(
    f"Assignments excluded from analysis: "
    f"{(~annotation_analysis_df['include_in_analysis'].astype('boolean')).sum()}"
)

print(
    f"\nSaved document-level table to: "
    f"{DOCUMENT_OUTPUT_PATH}"
)

print(
    f"Saved annotation-level table to: "
    f"{ANNOTATION_OUTPUT_PATH}"
)
