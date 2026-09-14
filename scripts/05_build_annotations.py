"""Create the token-level annotation table.

The script uses file_manifest.csv to locate the available token-level annotation
JSONs. It does not search the reviewer packet folders or inspect ZIP files.

Each JSON contains the annotations recorded for one modernised document. An
annotation can have up to two error categories, and each category can have
multiple fields. The script expands these structures so that each row represents
one annotation–error category–field combination.

The error and field codes are joined to rule_manifest.csv to add their
human-readable names.

Documents whose annotation JSON is missing are skipped. They remain recorded as
missing review data in file_manifest.csv and must not subsequently be treated
as documents with zero errors.

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
# Merge the rule manifest into the annotation dataframe to map codes to code expansions 
# ---------------------------------------------------------------------------

annotations_df = annotations_df.merge(
    rules_df,
    on = ["error_code", "field_code"],
    how = "left",
)


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
        "is_unique_mode",
        "status",
        "related_to",
        "related_text",
        "related_key",
        "annotation_json_path",
    ]
]


# ---------------------------------------------------------------------------
# Safeguard. Report any annotation codes not found in the rule manifest
# ---------------------------------------------------------------------------

unmatched_rules_df = (
    annotations_df.loc[
        annotations_df["error_category"].isna(),
        ["error_code", "field_code"],
    ]
    .drop_duplicates()
)

if not unmatched_rules_df.empty:
    print(
        "\nAnnotation codes without a matching "
        "rule-manifest entry:"
    )

    print(
        unmatched_rules_df.to_string(
            index = False
        )
    )


# ---------------------------------------------------------------------------
# Save the annotation dataframe as a table in csv form
# ---------------------------------------------------------------------------

annotations_df.to_csv(
    OUTPUT_PATH,
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
    f"Rows without a matching rule definition: "
    f"{len(unmatched_rules_df)}"
)

print(
    f"\nSaved annotation table to: "
    f"{OUTPUT_PATH}"
)