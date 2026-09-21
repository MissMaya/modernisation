"""
Creates the token-level annotation table.

The script uses file_manifest.csv to locate the available token-level annotation
JSONs. 

Each _GT_moderno.ann.json file contains the annotations recorded for the corresponding
modernised document.
An annotation can have up to two error categories, and each category can have
multiple fields. The script expands these structures so that each row represents
one annotation error category-field combination.

The error category name is added by matching the error code. The sub-rule name
is added only when the error code and field code form a valid combination in
rule_manifest.csv.

Every error-category and field assignment extracted from the annotation JSONs
is saved in annotations.csv, including assignments that do not match a valid
combination in rule_manifest.csv.

Invalid combinations are flagged in annotations.csv and copied to
invalid_rule_combinations_for_review.csv so that they can be checked by a
human. This script does not alter or remove them. Human corrections and
decisions about whether to include them in the analysis will be applied by a
later script.

If a document's annotation JSON is unavailable, this script skips that document
because there are no annotation records to read. The missing file remains
recorded in file_manifest.csv.

A missing annotation JSON is different from an available JSON containing an
empty list. A missing JSON means that the review result is unavailable. An empty
JSON means that the review result is available and the reviewer recorded no
errors.

The completed table is saved as intermediate/annotations.csv.
"""

# ---------------------------------------------------------------------------
# Import the required modules
# ---------------------------------------------------------------------------
import json
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Setup the paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path("/workspaces/modernisation")

INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"

FILE_MANIFEST_PATH = (
    INTERMEDIATE_DIR / "file_manifest.csv"
)

RULE_MANIFEST_PATH = (
    INTERMEDIATE_DIR / "rule_manifest.csv"
)

OUTPUT_PATH = (
    INTERMEDIATE_DIR / "annotations.csv"
)

INVALID_REPORT_PATH = (
    INTERMEDIATE_DIR
    / "invalid_rule_combinations_for_review.csv"
)


# ---------------------------------------------------------------------------
# Check that the required intermediate tables exist 
# ---------------------------------------------------------------------------

required_outputs = [
    (
        FILE_MANIFEST_PATH,
        "02_build_file_manifest.py",
    ),
    (
        RULE_MANIFEST_PATH,
        "03_build_rule_manifest.py",
    ),
]

for output_path, producing_script in required_outputs:
    if not output_path.is_file():
        raise SystemExit(
            f"{output_path.name} was not found in "
            f"{INTERMEDIATE_DIR}. "
            f"Run {producing_script} first."
        )


# ---------------------------------------------------------------------------
# Load the file and rule manifests
# ---------------------------------------------------------------------------

file_manifest_df = pd.read_csv(
    FILE_MANIFEST_PATH,
    dtype = {"filename_stem": "string"},
)

rules_df = pd.read_csv(
    RULE_MANIFEST_PATH,
    dtype = {
        "error_code": "string",
        "field_code": "string",
    }
)


# ---------------------------------------------------------------------------
# Expand an annotation JSON record into entity-field rows
# ---------------------------------------------------------------------------

def flatten_annotation(
    annotation,
    filename_stem,
    reviewer_packet,
    annotation_json_path,
):

    """
    This function expand a token-level annotation into separate error-classification rows.

    In the source JSON, error categories are called 'entities'. Each annotation
    has space for a first and second error category, and each category can have one
    or more fields (or sub-rules).

    The function creates one row for every error-category and field combination.
    For example, one error category with two fields produces two rows. Two error
    categories with two fields each produce four rows.

    If either error-category position is unused, it is ignored. If an error
    category has been assigned without a field code, the category is retained in
    one row and the field code is left blank.
    """

    rows = []

    entity_details = [
        {
            "entity_position": 1,
            "error_code": annotation.get(
                "firstEntityCode",
                "",
            ),
            "field_codes": annotation.get(
                "fieldsFirstEntity",
                [],
            ),
        },
        {
            "entity_position": 2,
            "error_code": annotation.get(
                "secondEntityCode",
                "",
            ),
            "field_codes": annotation.get(
                "fieldsSecondEntity",
                [],
            ),
        },
    ]

    for entity in entity_details:
        error_code = entity["error_code"]
        field_codes = entity["field_codes"]

        # An empty entity code means that this entity position was not used.
        if not error_code:
            continue

        # Preserve an assigned entity even if it has no accompanying field.
        if not field_codes:
            field_codes = [pd.NA]

        for field_code in field_codes:
            rows.append(
                {
                    "filename_stem": filename_stem,
                    "reviewer_packet": reviewer_packet,
                    "annotation_id": annotation.get("id"),
                    "annotated_text": annotation.get("text"),
                    "start": annotation.get("start"),
                    "end": annotation.get("end"),
                    "paragraph": annotation.get("paragraph"),
                    "document_id_internal": annotation.get(
                        "documentId"
                    ),
                    "entity_position": entity[
                        "entity_position"
                    ],
                    "error_code": error_code,
                    "field_code": field_code,
                    "is_unique_mode": annotation.get(
                        "isUniqueMode"
                    ),
                    "status": annotation.get("status"),
                    "related_to": annotation.get("relatedTo"),
                    "related_text": annotation.get(
                        "relatedText"
                    ),
                    "related_key": annotation.get(
                        "relatedKey"
                    ),
                    "annotation_json_path": (
                        annotation_json_path
                    ),
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Read each of the _GT_moderno.ann.json files
# ---------------------------------------------------------------------------

annotation_rows = []

json_files_read = 0
empty_json_files = 0
raw_annotation_records = 0

for manifest_row in file_manifest_df.itertuples(
    index = False
):
    annotation_json_path = (
        manifest_row.annotation_json_path
    )

    # A missing path means that the annotation JSON was not available when the
    # file manifest was created. Skip it without treating it as a zero-error
    # document. Including this because 1 .json is currently missing from a packet
    # at the time of writing 
    if pd.isna(annotation_json_path):
        continue

    full_json_path = (
        PROJECT_DIR / annotation_json_path
    )

    # Safeguard. If a path is listed in the manifest but no longer exists, the manifest is
    # out of date and should be rebuilt.
    if not full_json_path.is_file():
        raise FileNotFoundError(
            f"Annotation JSON listed in the file manifest "
            f"was not found: {full_json_path}. "
            "Rerun 02_build_file_manifest.py."
        )

    with full_json_path.open(
        "r",
        encoding = "utf-8-sig",
    ) as json_file:
        document_annotations = json.load(
            json_file
        )

    json_files_read += 1
    raw_annotation_records += len(
        document_annotations
    )

    # An empty JSON list means that the review data is available but the
    # reviewer recorded no errors for this document.
    if not document_annotations:
        empty_json_files += 1
        continue

    for annotation in document_annotations:
        annotation_rows.extend(
            flatten_annotation(
                annotation = annotation,
                filename_stem = (
                    manifest_row.filename_stem
                ),
                reviewer_packet = (
                    manifest_row.reviewer_packet
                ),
                annotation_json_path = (
                    annotation_json_path
                ),
            )
        )


# ---------------------------------------------------------------------------
# Now create the annotation dataframe
# ---------------------------------------------------------------------------

annotation_columns = [
    "filename_stem",
    "reviewer_packet",
    "annotation_id",
    "annotated_text",
    "start",
    "end",
    "paragraph",
    "document_id_internal",
    "entity_position",
    "error_code",
    "field_code",
    "is_unique_mode",
    "status",
    "related_to",
    "related_text",
    "related_key",
    "annotation_json_path",
]

annotations_df = pd.DataFrame(
    annotation_rows,
    columns = annotation_columns,
)

annotations_df["error_code"] = (
    annotations_df["error_code"].astype("string")
)

annotations_df["field_code"] = (
    annotations_df["field_code"].astype("string")
)


# ---------------------------------------------------------------------------
# Add the human-readable error-category names
# ---------------------------------------------------------------------------

# Match category names using the error code alone. This preserves the category
# name even when its accompanying field code is invalid.
category_lookup_df = (
    rules_df[
        ["error_code", "error_category"]
    ]
    .drop_duplicates()
)

# Each error code should describe only one error category.
if category_lookup_df["error_code"].duplicated().any():
    raise ValueError(
        "The rule manifest assigns more than one error-category name "
        "to the same error code. Check rule_manifest.csv."
    )

annotations_df = annotations_df.merge(
    category_lookup_df,
    on = "error_code",
    how = "left",
    validate = "many_to_one",
)


# ---------------------------------------------------------------------------
# Add sub-rule names and validate the error-code and field-code combinations
# ---------------------------------------------------------------------------

# A sub-rule is valid only when its error code and field code occur together in
# the rule manifest. Matching on both columns detects fields assigned to the
# wrong error category.
rule_pair_lookup_df = (
    rules_df[
        ["error_code", "field_code", "subrule"]
    ]
    .drop_duplicates()
)

if rule_pair_lookup_df.duplicated(
    subset = ["error_code", "field_code"]
).any():
    raise ValueError(
        "The rule manifest contains more than one sub-rule name for the "
        "same error-code and field-code combination."
    )

annotations_df = annotations_df.merge(
    rule_pair_lookup_df,
    on = ["error_code", "field_code"],
    how = "left",
    validate = "many_to_one",
)

# An assignment is valid when its error code exists and either:
#   1. no field was assigned; or
#   2. its error-code and field-code combination exists in the rule manifest.
annotations_df["rule_combination_valid"] = (
    annotations_df["error_category"].notna()
    & (
        annotations_df["field_code"].isna()
        | annotations_df["subrule"].notna()
    )
)

# Record a plain-language reason for every invalid assignment.
annotations_df["validation_issue"] = pd.Series(
    pd.NA,
    index = annotations_df.index,
    dtype = "string",
)

unknown_error_code = (
    annotations_df["error_category"].isna()
)

invalid_field_for_category = (
    annotations_df["error_category"].notna()
    & annotations_df["field_code"].notna()
    & annotations_df["subrule"].isna()
)

annotations_df.loc[
    unknown_error_code,
    "validation_issue",
] = "error_code_not_found"

annotations_df.loc[
    invalid_field_for_category,
    "validation_issue",
] = "field_code_not_valid_for_error_code"


# Arrange the rule names beside their corresponding codes.
annotations_df = annotations_df[
    [
        "filename_stem",
        "reviewer_packet",
        "annotation_id",
        "annotated_text",
        "start",
        "end",
        "paragraph",
        "document_id_internal",
        "entity_position",
        "error_code",
        "error_category",
        "field_code",
        "subrule",
        "rule_combination_valid",
        "validation_issue",
        "is_unique_mode",
        "status",
        "related_to",
        "related_text",
        "related_key",
        "annotation_json_path",
    ]
]


# ---------------------------------------------------------------------------
# Create the report of invalid combinations for human review
# ---------------------------------------------------------------------------

# Obtain the field's name independently of its assigned error category. This
# helps reviewers recognise a field that may have been paired with the wrong
# category. Field codes should be unique across the rule manifest.
field_name_lookup_df = (
    rules_df[
        ["field_code", "subrule"]
    ]
    .dropna(subset = ["field_code"])
    .drop_duplicates()
    .rename(columns = {"subrule": "field_name"})
)

if field_name_lookup_df["field_code"].duplicated().any():
    raise ValueError(
        "The rule manifest assigns more than one name to the same field "
        "code. Check rule_manifest.csv."
    )

invalid_combinations_df = (
    annotations_df.loc[
        ~annotations_df["rule_combination_valid"]
    ]
    .merge(
        field_name_lookup_df,
        on = "field_code",
        how = "left",
        validate = "many_to_one",
    )
)

invalid_report_columns = [
    "reviewer_packet",
    "filename_stem",
    "annotation_id",
    "annotated_text",
    "start",
    "end",
    "entity_position",
    "error_code",
    "error_category",
    "field_code",
    "field_name",
    "validation_issue",
]

invalid_review_df = invalid_combinations_df[
    invalid_report_columns
].copy()

# These blank columns are provided for the decisions returned after human
# review. Stage 5 does not fill in or apply those decisions.
invalid_review_df["decision"] = pd.NA
invalid_review_df["corrected_error_code"] = pd.NA
invalid_review_df["corrected_field_code"] = pd.NA
invalid_review_df["review_note"] = pd.NA


# ---------------------------------------------------------------------------
# Save the annotation dataframe as a table in csv form
# ---------------------------------------------------------------------------

annotations_df.to_csv(
    OUTPUT_PATH,
    index = False,
    encoding = "utf-8-sig",
)

invalid_review_df.to_csv(
    INVALID_REPORT_PATH,
    index = False,
    encoding = "utf-8-sig",
)


# ---------------------------------------------------------------------------
# Some quick sense checks
# ---------------------------------------------------------------------------

missing_json_files = (
    file_manifest_df[
        "annotation_json_path"
    ]
    .isna()
    .sum()
)

print(
    f"\nDocuments in file manifest: "
    f"{len(file_manifest_df)}"
)

print(
    f"Annotation JSONs read: "
    f"{json_files_read}"
)

print(
    f"Missing annotation JSONs skipped: "
    f"{missing_json_files}"
)

print(
    f"Available JSONs containing no annotations: "
    f"{empty_json_files}"
)

print(
    f"Original annotation records: "
    f"{raw_annotation_records}"
)

print(
    f"Entity-field rows created: "
    f"{len(annotations_df)}"
)

print(
    f"Valid entity-field assignments: "
    f"{annotations_df['rule_combination_valid'].sum()}"
)

print(
    f"Invalid entity-field assignments: "
    f"{len(invalid_review_df)}"
)

distinct_invalid_combinations = (
    invalid_review_df[
        ["error_code", "field_code"]
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    f"Distinct invalid code combinations: "
    f"{distinct_invalid_combinations}"
)

print(
    f"\nSaved annotation table to: "
    f"{OUTPUT_PATH}"
)

print(
    f"Saved invalid-combination review report to: "
    f"{INVALID_REPORT_PATH}"
)
