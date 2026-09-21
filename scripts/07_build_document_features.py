"""
Creates a row of text-length features for every document.

The file manifest contains one row for each document and records the path to
its modernised TXT file. This script reads each available text and calculates
document-level counts that are later used as denominators when comparing
human annotations across documents and models.

The principal measure is n_modernised_tokens because the reviewers annotated
the modernised texts. Later analyses can then calculate measures such
as annotations per 1,000 modernised tokens.

WHAT COUNTS AS A TOKEN?

A token is counted as a Unicode word sequence. Apostrophes and hyphens are
allowed inside a token, so forms such as "d'él" and "teórico-práctico" count as
one token. Standalone punctuation and whitespace are not counted as tokens.
This definition is recorded in the token_count_method column in the output.

PRE-MODERNISATION TEXT FEATURES

The pre-modernisation text is the original transcription with a filename ending
_GT.txt. It is the text that was supplied to the model for modernisation.

Stage 2 records the pre-modernisation and modernised TXT paths in
file_manifest.csv and reports any missing or duplicate files. This script uses
the paths supplied by that manifest and does not repeat the file search.

document_features.csv contains one row for every document in file_manifest.csv,
including documents for which one or both texts were unavailable. Features are
calculated for each available text. Counts that cannot be calculated are left
blank; zero is reserved for an available TXT file that contains no text.

The feature_status column states whether the row is complete or whether its
modernised text, pre-modernisation text, or both texts were unavailable. The
Stage 2 issue report contains the precise reason, such as a missing or duplicate
file.

The completed table is saved as intermediate/document_features.csv.
"""

# ---------------------------------------------------------------------------
# Import the required modules
# ---------------------------------------------------------------------------

import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Set up the paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path("/workspaces/modernisation")
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"

FILE_MANIFEST_PATH = INTERMEDIATE_DIR / "file_manifest.csv"
OUTPUT_PATH = INTERMEDIATE_DIR / "document_features.csv"


# ---------------------------------------------------------------------------
# Define the token-counting method
# ---------------------------------------------------------------------------

# Python's Unicode-aware \w matches letters, accented letters, numbers and the
# underscore. The optional repeated group keeps internally hyphenated and
# apostrophised forms together. The surrounding word boundaries prevent an
# apostrophe or hyphen at the edge of a word from being counted as part of it.
TOKEN_PATTERN = re.compile(
    r"\b\w+(?:[’'-]\w+)*\b",
    flags = re.UNICODE,
)

TOKEN_COUNT_METHOD = "unicode_word_regex_with_internal_apostrophes_and_hyphens"


# ---------------------------------------------------------------------------
# Check that the file manifest exists
# ---------------------------------------------------------------------------

if not FILE_MANIFEST_PATH.is_file():
    raise SystemExit(
        f"{FILE_MANIFEST_PATH.name} was not found in {INTERMEDIATE_DIR}. "
        "Run 02_build_file_manifest.py first."
    )


# ---------------------------------------------------------------------------
# Load and validate the file manifest
# ---------------------------------------------------------------------------

file_manifest_df = pd.read_csv(
    FILE_MANIFEST_PATH,
    dtype = {
        "filename_stem": "string",
        "modernised_txt_path": "string",
        "pre_modernisation_txt_path": "string",
    },
)

required_columns = {
    "filename_stem",
    "modernised_txt_path",
    "pre_modernisation_txt_path",
}

missing_columns = required_columns - set(file_manifest_df.columns)

if missing_columns:
    raise ValueError(
        "file_manifest.csv is missing required columns: "
        f"{sorted(missing_columns)}."
    )

if file_manifest_df["filename_stem"].duplicated().any():
    duplicate_stems = file_manifest_df.loc[
        file_manifest_df["filename_stem"].duplicated(
            keep = False
        ),
        "filename_stem",
    ]
    raise ValueError(
        "file_manifest.csv must contain one row per document. Duplicate "
        "filename_stem values were found:\n"
        f"{duplicate_stems.to_string(index = False)}"
    )

# ---------------------------------------------------------------------------
# Define the functions used to read and measure a text
# ---------------------------------------------------------------------------

def resolve_project_path(saved_path):
    """Return an absolute path for a path stored in the file manifest."""

    path = Path(saved_path)

    if path.is_absolute():
        return path

    return PROJECT_DIR / path


def count_text_features(text):
    """Calculate reproducible token, character and line counts for one text."""

    return {
        "tokens": len(TOKEN_PATTERN.findall(text)),
        "characters": len(text),
        "characters_without_whitespace": len(
            re.sub(r"\s", "", text)
        ),
        "lines": len(text.splitlines()),
    }


def read_and_measure_text(saved_path):
    """Measure a manifest-listed text, or return blank counts for no path."""

    # A missing manifest value means that no path was found for this document.
    if pd.isna(saved_path):
        return {
            "available": False,
            "tokens": pd.NA,
            "characters": pd.NA,
            "characters_without_whitespace": pd.NA,
            "lines": pd.NA,
        }

    # Stage 2 has already checked the file and recorded the path. Stage 7 uses
    # that manifest result directly rather than performing another file search.
    full_path = resolve_project_path(saved_path)
    text = full_path.read_text(encoding = "utf-8-sig")
    counts = count_text_features(text)

    return {
        "available": True,
        **counts,
    }


# ---------------------------------------------------------------------------
# Calculate one row of features for every document in the file manifest
# ---------------------------------------------------------------------------

feature_rows = []

for manifest_row in file_manifest_df.itertuples(index = False):
    modernised_features = read_and_measure_text(
        manifest_row.modernised_txt_path
    )

    pre_modernisation_features = read_and_measure_text(
        manifest_row.pre_modernisation_txt_path
    )

    feature_rows.append(
        {
            "filename_stem": manifest_row.filename_stem,
            "modernised_text_available": modernised_features[
                "available"
            ],
            "n_modernised_tokens": modernised_features[
                "tokens"
            ],
            "n_modernised_characters": modernised_features[
                "characters"
            ],
            "n_modernised_characters_without_whitespace": (
                modernised_features[
                    "characters_without_whitespace"
                ]
            ),
            "n_modernised_lines": modernised_features[
                "lines"
            ],
            "pre_modernisation_text_available": (
                pre_modernisation_features[
                    "available"
                ]
            ),
            "n_pre_modernisation_tokens": (
                pre_modernisation_features["tokens"]
            ),
            "n_pre_modernisation_characters": (
                pre_modernisation_features["characters"]
            ),
            "n_pre_modernisation_characters_without_whitespace": (
                pre_modernisation_features[
                    "characters_without_whitespace"
                ]
            ),
            "n_pre_modernisation_lines": (
                pre_modernisation_features["lines"]
            ),
            "token_count_method": TOKEN_COUNT_METHOD,
        }
    )


document_features_df = pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# Describe whether both sets of document features could be calculated
# ---------------------------------------------------------------------------

document_features_df["feature_status"] = "complete"

missing_modernised = (
    ~document_features_df["modernised_text_available"]
)

missing_pre_modernisation = (
    ~document_features_df[
        "pre_modernisation_text_available"
    ]
)

document_features_df.loc[
    missing_modernised & ~missing_pre_modernisation,
    "feature_status",
] = "modernised_text_unavailable"

document_features_df.loc[
    ~missing_modernised & missing_pre_modernisation,
    "feature_status",
] = "pre_modernisation_text_unavailable"

document_features_df.loc[
    missing_modernised & missing_pre_modernisation,
    "feature_status",
] = "both_texts_unavailable"

# Use nullable integer columns so unavailable counts remain blank while counts
# for genuinely empty files remain zero.
count_columns = [
    "n_modernised_tokens",
    "n_modernised_characters",
    "n_modernised_characters_without_whitespace",
    "n_modernised_lines",
    "n_pre_modernisation_tokens",
    "n_pre_modernisation_characters",
    "n_pre_modernisation_characters_without_whitespace",
    "n_pre_modernisation_lines",
]

for column in count_columns:
    document_features_df[column] = (
        document_features_df[column].astype("Int64")
    )

# ---------------------------------------------------------------------------
# Calculate changes between pre-modernisation and modernised texts
# ---------------------------------------------------------------------------

document_features_df["token_count_change"] = (
    document_features_df["n_modernised_tokens"]
    - document_features_df["n_pre_modernisation_tokens"]
)

document_features_df["character_count_change"] = (
    document_features_df["n_modernised_characters"]
    - document_features_df["n_pre_modernisation_characters"]
)

# Percentage change is undefined when the pre-modernisation count is zero.
# Those values remain blank rather than producing infinity.
valid_pre_modernisation_token_count = (
    document_features_df["n_pre_modernisation_tokens"].notna()
    & document_features_df["n_pre_modernisation_tokens"].ne(0)
)

valid_pre_modernisation_character_count = (
    document_features_df["n_pre_modernisation_characters"].notna()
    & document_features_df["n_pre_modernisation_characters"].ne(0)
)

document_features_df["token_count_change_pct"] = pd.Series(
    pd.NA,
    index = document_features_df.index,
    dtype = "Float64",
)
document_features_df.loc[
    valid_pre_modernisation_token_count,
    "token_count_change_pct",
] = (
    document_features_df.loc[
        valid_pre_modernisation_token_count,
        "token_count_change",
    ]
    / document_features_df.loc[
        valid_pre_modernisation_token_count,
        "n_pre_modernisation_tokens",
    ]
    * 100
)

document_features_df["character_count_change_pct"] = pd.Series(
    pd.NA,
    index = document_features_df.index,
    dtype = "Float64",
)
document_features_df.loc[
    valid_pre_modernisation_character_count,
    "character_count_change_pct",
] = (
    document_features_df.loc[
        valid_pre_modernisation_character_count,
        "character_count_change",
    ]
    / document_features_df.loc[
        valid_pre_modernisation_character_count,
        "n_pre_modernisation_characters",
    ]
    * 100
)


# ---------------------------------------------------------------------------
# Save the document-level feature table
# ---------------------------------------------------------------------------

document_features_df.to_csv(
    OUTPUT_PATH,
    index = False,
    encoding = "utf-8-sig",
)


# ---------------------------------------------------------------------------
# Print concise sense checks
# ---------------------------------------------------------------------------

print(
    f"\nDocuments in file manifest: "
    f"{len(file_manifest_df)}"
)

print(
    f"Document-feature rows created: "
    f"{len(document_features_df)}"
)

print(
    f"Modernised texts measured: "
    f"{document_features_df['modernised_text_available'].sum()}"
)

print(
    f"Modernised texts unavailable: "
    f"{(~document_features_df['modernised_text_available']).sum()}"
)

print(
    f"Pre-modernisation texts measured: "
    f"{document_features_df['pre_modernisation_text_available'].sum()}"
)

print(
    f"Pre-modernisation texts unavailable: "
    f"{(~document_features_df['pre_modernisation_text_available']).sum()}"
)

print("\nRows by feature status:")

print(
    document_features_df["feature_status"]
    .value_counts()
    .to_string()
)

print(
    f"\nSaved document-feature table to: "
    f"{OUTPUT_PATH}"
)
