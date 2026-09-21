"""
This script resolves the invalid category-field combinations identified by
running "stage 5": 05_build_annotations.py.

Stage 5 saves every extracted annotation assignment in annotations.csv and
marks whether its error-code and field-code combination matches the rule
manifest. This script decides which of those rows may be used in analysis.

Set INVALID_COMBINATION_POLICY to one of the following values:

    "exclude_all"
        Exclude every invalid combination from analysis.

    "include_all"
        Include every invalid combination, while preserving its invalid flag.

    "human_decisions"
        Read human_review/annotation_decisions.xlsx. Each invalid assignment
        must have a decision to accept it, exclude it, or replace its codes with
        a corrected valid combination.

HUMAN DECISION WORKBOOK

The human decision file must be an Excel workbook in .xlsx format and must be
placed at:

    /workspaces/modernisation/human_review/annotation_decisions.xlsx

The script reads the first worksheet. The first row must contain these exact
column names:

    reviewer_packet
    filename_stem
    annotation_id
    entity_position
    error_code
    field_code
    decision
    corrected_error_code
    corrected_field_code
    review_note

Use one row for every invalid assignment listed in
intermediate/invalid_rule_combinations_for_review.csv. The first six columns
identify the exact original assignment and should be copied without alteration.
They are all needed because the same error-code and field-code combination can
occur more than once in the same document.

Enter one of the following values in the decision column:

    accept
        Include the original combination in the analysis despite its invalid
        code-map match.

    exclude
        Retain the original assignment in the resolved table, but exclude it
        from analysis. "ignore" and "reject" are also accepted and are treated
        as "exclude".

    correct
        Replace the original codes with the values supplied in
        corrected_error_code and corrected_field_code. corrected_error_code is
        required. corrected_field_code may be blank when the corrected error
        category should have no field. The corrected combination must exist in
        rule_manifest.csv.

review_note is optional and may be left blank. For decisions of "accept" or
"exclude", both corrected-code columns should be left blank.

The source annotations.csv file is never altered. The resolved result is saved
as intermediate/resolved_annotations.csv, so the original extraction and the output
after reviewing invalid assignments are both preserved. 
"""

# ---------------------------------------------------------------------------
# Import the required modules
# ---------------------------------------------------------------------------

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Select how invalid combinations should be handled
# ---------------------------------------------------------------------------

# Change only this value in order to change the handling. Valid options are
# "exclude_all", "include_all" and "human_decisions".
INVALID_COMBINATION_POLICY = "exclude_all"

ALLOWED_POLICIES = {
    "exclude_all",
    "include_all",
    "human_decisions",
}


# ---------------------------------------------------------------------------
# Set up the paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path("/workspaces/modernisation")
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"
HUMAN_REVIEW_DIR = PROJECT_DIR / "human_review"

ANNOTATIONS_PATH = INTERMEDIATE_DIR / "annotations.csv"
RULE_MANIFEST_PATH = INTERMEDIATE_DIR / "rule_manifest.csv"

# Keep the human-supplied workbook in a totally different directory. 
# This prevents it frombeing deleted by a reset of automatically 
# generated intermediate files.
DECISIONS_PATH = HUMAN_REVIEW_DIR / "annotation_decisions.xlsx"

OUTPUT_PATH = INTERMEDIATE_DIR / "resolved_annotations.csv"


# ---------------------------------------------------------------------------
# Check the selected policy and required inputs
# ---------------------------------------------------------------------------

if INVALID_COMBINATION_POLICY not in ALLOWED_POLICIES:
    raise ValueError(
        "INVALID_COMBINATION_POLICY must be 'exclude_all', "
        "'include_all' or 'human_decisions'."
    )

required_outputs = [
    (ANNOTATIONS_PATH, "05_build_annotations.py"),
    (RULE_MANIFEST_PATH, "03_build_rule_manifest.py"),
]

for output_path, producing_script in required_outputs:
    if not output_path.is_file():
        raise SystemExit(
            f"{output_path.name} was not found in {INTERMEDIATE_DIR}. "
            f"Run {producing_script} first."
        )

if (
    INVALID_COMBINATION_POLICY == "human_decisions"
    and not DECISIONS_PATH.is_file()
):
    raise SystemExit(
        "INVALID_COMBINATION_POLICY is set to 'human_decisions', but "
        f"{DECISIONS_PATH} was not found. Supply the completed decision "
        "workbook or select another policy."
    )


# ---------------------------------------------------------------------------
# Load the Stage 5 annotations and the rule manifest
# ---------------------------------------------------------------------------

annotations_df = pd.read_csv(
    ANNOTATIONS_PATH,
    dtype = {
        "filename_stem": "string",
        "error_code": "string",
        "field_code": "string",
        "validation_issue": "string",
    },
)

rules_df = pd.read_csv(
    RULE_MANIFEST_PATH,
    dtype = {
        "error_code": "string",
        "field_code": "string",
    },
)

required_annotation_columns = {
    "reviewer_packet",
    "filename_stem",
    "annotation_id",
    "entity_position",
    "error_code",
    "field_code",
    "error_category",
    "subrule",
    "rule_combination_valid",
}

missing_annotation_columns = (
    required_annotation_columns
    - set(annotations_df.columns)
)

if missing_annotation_columns:
    raise ValueError(
        "annotations.csv is missing required columns: "
        f"{sorted(missing_annotation_columns)}. "
        "Rerun the redesigned 05_build_annotations.py."
    )


# ---------------------------------------------------------------------------
# Prepare columns that record the analytical decision
# ---------------------------------------------------------------------------

resolved_df = annotations_df.copy()

# These effective columns contain the codes and names that later analytical
# tables should use. They initially reproduce the original Stage 5 values.
resolved_df["effective_error_code"] = resolved_df["error_code"]
resolved_df["effective_error_category"] = resolved_df["error_category"]
resolved_df["effective_field_code"] = resolved_df["field_code"]
resolved_df["effective_subrule"] = resolved_df["subrule"]
resolved_df["effective_rule_combination_valid"] = (
    resolved_df["rule_combination_valid"]
)

# Every combination that passed Stage 5 validation is included automatically.
resolved_df["include_in_analysis"] = (
    resolved_df["rule_combination_valid"]
)
resolved_df["resolution_action"] = "valid"
resolved_df["resolution_source"] = "automatic_validation"
resolved_df["review_note"] = pd.NA

invalid_mask = ~resolved_df["rule_combination_valid"]


# ---------------------------------------------------------------------------
# Apply a single decision to all invalid combinations when no human file is used
# ---------------------------------------------------------------------------

if INVALID_COMBINATION_POLICY == "exclude_all":
    resolved_df.loc[
        invalid_mask,
        "include_in_analysis",
    ] = False
    resolved_df.loc[
        invalid_mask,
        "resolution_action",
    ] = "excluded"
    resolved_df.loc[
        invalid_mask,
        "resolution_source",
    ] = "blanket_policy"

elif INVALID_COMBINATION_POLICY == "include_all":
    resolved_df.loc[
        invalid_mask,
        "include_in_analysis",
    ] = True
    resolved_df.loc[
        invalid_mask,
        "resolution_action",
    ] = "accepted"
    resolved_df.loc[
        invalid_mask,
        "resolution_source",
    ] = "blanket_policy"


# ---------------------------------------------------------------------------
# Apply the separately supplied human decisions (if supplied)
# ---------------------------------------------------------------------------

else:
    decisions_df = pd.read_excel(
        DECISIONS_PATH,
        dtype = {
            "filename_stem": "string",
            "error_code": "string",
            "field_code": "string",
            "decision": "string",
            "corrected_error_code": "string",
            "corrected_field_code": "string",
            "review_note": "string",
        },
    )

    # These columns identify the exact category-field assignment being reviewed
    # and exactly which annotation is affected
    decision_keys = [
        "reviewer_packet",
        "filename_stem",
        "annotation_id",
        "entity_position",
        "error_code",
        "field_code",
    ]

    required_decision_columns = set(decision_keys) | {
        "decision",
        "corrected_error_code",
        "corrected_field_code",
    }

    missing_decision_columns = (
        required_decision_columns
        - set(decisions_df.columns)
    )

    if missing_decision_columns:
        raise ValueError(
            "annotation_decisions.xlsx is missing required columns: "
            f"{sorted(missing_decision_columns)}."
        )

    if "review_note" not in decisions_df.columns:
        decisions_df["review_note"] = pd.NA

    # Make the numeric identifiers comparable between CSV and Excel input.
    numeric_keys = [
        "reviewer_packet",
        "annotation_id",
        "entity_position",
    ]

    for column in numeric_keys:
        resolved_df[column] = pd.to_numeric(
            resolved_df[column],
            errors = "raise",
        ).astype("Int64")
        decisions_df[column] = pd.to_numeric(
            decisions_df[column],
            errors = "raise",
        ).astype("Int64")

    # Remove accidental surrounding spaces and convert common alternatives to
    # the three decision values used by the script.
    decisions_df["decision"] = (
        decisions_df["decision"]
        .str.strip()
        .str.lower()
    )

    decision_aliases = {
        "accept": "accept",
        "include": "accept",
        "exclude": "exclude",
        "ignore": "exclude",
        "reject": "exclude",
        "correct": "correct",
    }

    decisions_df["decision"] = (
        decisions_df["decision"]
        .map(decision_aliases)
    )

    if decisions_df["decision"].isna().any():
        raise ValueError(
            "Every row in annotation_decisions.xlsx must contain one of "
            "these decisions: accept, exclude or correct."
        )

    if decisions_df.duplicated(
        subset = decision_keys,
        keep = False,
    ).any():
        duplicate_rows = decisions_df.loc[
            decisions_df.duplicated(
                subset = decision_keys,
                keep = False,
            ),
            decision_keys,
        ]
        raise ValueError(
            "annotation_decisions.xlsx contains duplicate decisions for:\n"
            f"{duplicate_rows.to_string(index = False)}"
        )

    invalid_assignments_df = resolved_df.loc[
        invalid_mask,
        decision_keys,
    ].copy()

    decision_check_df = invalid_assignments_df.merge(
        decisions_df[decision_keys + ["decision"]],
        on = decision_keys,
        how = "outer",
        indicator = True,
        validate = "one_to_one",
    )

    missing_decisions_df = decision_check_df.loc[
        decision_check_df["_merge"] == "left_only",
        decision_keys,
    ]

    unmatched_decisions_df = decision_check_df.loc[
        decision_check_df["_merge"] == "right_only",
        decision_keys,
    ]

    if not missing_decisions_df.empty:
        raise ValueError(
            "The decision workbook does not contain a decision for every "
            "invalid assignment. Missing decisions:\n"
            f"{missing_decisions_df.to_string(index = False)}"
        )

    if not unmatched_decisions_df.empty:
        raise ValueError(
            "The decision workbook contains rows that do not match current "
            "invalid assignments. Unmatched decisions:\n"
            f"{unmatched_decisions_df.to_string(index = False)}"
        )

    # Add the reviewed decisions to the invalid rows. 
    resolved_df = resolved_df.merge(
        decisions_df[
            decision_keys
            + [
                "decision",
                "corrected_error_code",
                "corrected_field_code",
                "review_note",
            ]
        ],
        on = decision_keys,
        how = "left",
        validate = "one_to_one",
        suffixes = ("", "_human"),
    )

    resolved_df["human_decision"] = resolved_df["decision"]

    human_decision_mask = resolved_df["decision"].notna()
    accepted_mask = resolved_df["decision"].eq("accept")
    excluded_mask = resolved_df["decision"].eq("exclude")
    corrected_mask = resolved_df["decision"].eq("correct")

    resolved_df.loc[
        human_decision_mask,
        "resolution_source",
    ] = "human_decision"

    resolved_df.loc[
        accepted_mask,
        "include_in_analysis",
    ] = True
    resolved_df.loc[
        accepted_mask,
        "resolution_action",
    ] = "accepted"

    resolved_df.loc[
        excluded_mask,
        "include_in_analysis",
    ] = False
    resolved_df.loc[
        excluded_mask,
        "resolution_action",
    ] = "excluded"

    if resolved_df.loc[
        corrected_mask,
        "corrected_error_code",
    ].isna().any():
        raise ValueError(
            "Every row with decision 'correct' must provide a "
            "corrected_error_code. A blank corrected_field_code is allowed "
            "and means that the corrected category has no field."
        )

    resolved_df.loc[
        corrected_mask,
        "effective_error_code",
    ] = resolved_df.loc[
        corrected_mask,
        "corrected_error_code",
    ]
    resolved_df.loc[
        corrected_mask,
        "effective_field_code",
    ] = resolved_df.loc[
        corrected_mask,
        "corrected_field_code",
    ]
    resolved_df.loc[
        corrected_mask,
        "resolution_action",
    ] = "corrected"
    resolved_df.loc[
        corrected_mask,
        "include_in_analysis",
    ] = True

    # Copy any notes from the decision file into the review_note column
    if "review_note_human" in resolved_df.columns:
        resolved_df["review_note"] = (
            resolved_df["review_note_human"]
            .combine_first(resolved_df["review_note"])
        )
        resolved_df = resolved_df.drop(
            columns = ["review_note_human"]
        )

    resolved_df = resolved_df.drop(columns = ["decision"])


# ---------------------------------------------------------------------------
# Revalidate the codes after applying any human corrections
# ---------------------------------------------------------------------------

# Always include the human-decision and corrected-code columns in the output.
# When the selected option is exclude_all or include_all, no human decision
# workbook is used, so these columns are added but left blank. 
for optional_column in [
    "human_decision",
    "corrected_error_code",
    "corrected_field_code",
]:
    if optional_column not in resolved_df.columns:
        resolved_df[optional_column] = pd.NA

category_lookup_df = (
    rules_df[
        ["error_code", "error_category"]
    ]
    .drop_duplicates()
)

if category_lookup_df["error_code"].duplicated().any():
    raise ValueError(
        "The rule manifest assigns more than one category name to the same "
        "error code."
    )

pair_lookup_df = (
    rules_df[
        ["error_code", "field_code", "subrule"]
    ]
    .drop_duplicates()
)

if pair_lookup_df.duplicated(
    subset = ["error_code", "field_code"]
).any():
    raise ValueError(
        "The rule manifest assigns more than one sub-rule name to the same "
        "error-code and field-code combination."
    )

effective_names_df = (
    resolved_df[
        ["effective_error_code", "effective_field_code"]
    ]
    .rename(
        columns = {
            "effective_error_code": "error_code",
            "effective_field_code": "field_code",
        }
    )
    .merge(
        category_lookup_df,
        on = "error_code",
        how = "left",
        validate = "many_to_one",
    )
    .merge(
        pair_lookup_df,
        on = ["error_code", "field_code"],
        how = "left",
        validate = "many_to_one",
    )
)

resolved_df["effective_error_category"] = (
    effective_names_df["error_category"].array
)
resolved_df["effective_subrule"] = (
    effective_names_df["subrule"].array
)

resolved_df["effective_rule_combination_valid"] = (
    resolved_df["effective_error_category"].notna()
    & (
        resolved_df["effective_field_code"].isna()
        | resolved_df["effective_subrule"].notna()
    )
)

# A proposed correction must produce a valid combination. Accepted original
# combinations may remain invalid because the human decision is an explicit
# instruction to include them despite the code-map mismatch.
corrected_but_invalid = (
    resolved_df["resolution_action"].eq("corrected")
    & ~resolved_df["effective_rule_combination_valid"]
)

if corrected_but_invalid.any():
    invalid_corrections = resolved_df.loc[
        corrected_but_invalid,
        [
            "reviewer_packet",
            "filename_stem",
            "annotation_id",
            "entity_position",
            "effective_error_code",
            "effective_field_code",
        ],
    ]
    raise ValueError(
        "The following proposed corrections are not valid combinations in "
        "rule_manifest.csv:\n"
        f"{invalid_corrections.to_string(index = False)}"
    )


# ---------------------------------------------------------------------------
# Record the selected policy and save the resolved annotation table
# ---------------------------------------------------------------------------

resolved_df["invalid_combination_policy"] = (
    INVALID_COMBINATION_POLICY
)

resolved_df.to_csv(
    OUTPUT_PATH,
    index = False,
    encoding = "utf-8-sig",
)


# ---------------------------------------------------------------------------
# Brief summary of the decisions
# ---------------------------------------------------------------------------

print(
    f"\nInvalid-combination policy: "
    f"{INVALID_COMBINATION_POLICY}"
)

print(
    f"Annotation assignment rows: "
    f"{len(resolved_df)}"
)

print(
    f"Originally valid assignments: "
    f"{resolved_df['rule_combination_valid'].sum()}"
)

print(
    f"Originally invalid assignments: "
    f"{(~resolved_df['rule_combination_valid']).sum()}"
)

print("\nResolution actions:")
print(
    resolved_df["resolution_action"]
    .value_counts(dropna = False)
    .to_string()
)

print(
    f"\nAssignments included in analysis: "
    f"{resolved_df['include_in_analysis'].sum()}"
)

print(
    f"Assignments excluded from analysis: "
    f"{(~resolved_df['include_in_analysis']).sum()}"
)

print(
    f"\nSaved resolved annotation table to: "
    f"{OUTPUT_PATH}"
)
