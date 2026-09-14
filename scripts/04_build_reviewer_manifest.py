"""
Creates a table linking each reviewer packet to the reviewer's name.

Each reviewer packet has an annotations-legend.json file. This file contains
the same annotation<->code map as the base legend, but its top-level ``name``
value also contains the reviewer's name.

For example:

    Revisión_Modernizador_Rodrigo

The script takes the text after the final underscore and records it as the
reviewer name.

The paths to these JSON files are obtained from file_manifest.csv. 
The completed table contains one row per reviewer packet and is saved as
intermediate/reviewer_manifest.csv.
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

OUTPUT_PATH = (
    INTERMEDIATE_DIR / "reviewer_manifest.csv"
)


# ---------------------------------------------------------------------------
# Check that the file manifest has been created
# ---------------------------------------------------------------------------

if not FILE_MANIFEST_PATH.is_file():
    raise SystemExit(
        f"{FILE_MANIFEST_PATH.name} was not found in "
        f"{INTERMEDIATE_DIR}. "
        "Run 02_build_file_manifest.py first."
    )


# ---------------------------------------------------------------------------
# Load the file manifest
# ---------------------------------------------------------------------------

file_manifest_df = pd.read_csv(
    FILE_MANIFEST_PATH
)


# ---------------------------------------------------------------------------
# Extract one reviewer name for each reviewer packet
# ---------------------------------------------------------------------------

reviewer_rows = []

for reviewer_packet in sorted(
    file_manifest_df["reviewer_packet"].unique()
):
    packet_rows = file_manifest_df.loc[
        file_manifest_df["reviewer_packet"].eq(
            reviewer_packet
        )
    ]

    # file_manifest.csv contains one row per document but the reviewer-specific
    # annotation<->code map applies to the entire packet. Its path is therefore
    # repeated across the packet's 20 document rows. We remove missing and repeated
    # values to obtain the single code-map path for this packet.
    legend_paths = (
        packet_rows["annotations_legend_path"]
        .dropna()
        .drop_duplicates()
    )

    if legend_paths.empty:
        print(
            f"Reviewer packet {reviewer_packet}: "
            "no reviewer-specific annotation-code map was found."
        )

        reviewer_name = pd.NA

    elif len(legend_paths) > 1:
        print(
            f"Reviewer packet {reviewer_packet}: "
            "multiple reviewer-specific annotation-code maps were found."
        )

        reviewer_name = pd.NA

    else:
        legend_path = (
            PROJECT_DIR / legend_paths.iloc[0]
        )

        with legend_path.open(
            "r",
            encoding = "utf-8-sig",
        ) as json_file:
            legend_data = json.load(json_file)

        # The reviewer name appears after the final underscore in a value such
        # as ``Revisión_Modernizador_Rodrigo``.
        reviewer_name = (
            legend_data["name"]
            .rsplit("_", 1)[-1]
            .strip()
        )

    reviewer_rows.append(
        {
            "reviewer_packet": reviewer_packet,
            "reviewer_name": reviewer_name,
        }
    )


reviewers_df = pd.DataFrame(
    reviewer_rows
)


# ---------------------------------------------------------------------------
# Some quick sense checks
# ---------------------------------------------------------------------------

print(
    f"\nReviewer packets: "
    f"{len(reviewers_df)}"
)

print(
    f"Missing reviewer names: "
    f"{reviewers_df['reviewer_name'].isna().sum()}"
)

print(
    f"Duplicate reviewer names: "
    f"{reviewers_df['reviewer_name'].duplicated().sum()}"
)

print("\nReviewer manifest:")

print(
    reviewers_df.to_string(
        index = False
    )
)


# ---------------------------------------------------------------------------
# Save the reviewer manifest
# ---------------------------------------------------------------------------

reviewers_df.to_csv(
    OUTPUT_PATH,
    index = False,
    encoding = "utf-8-sig",
)

print(
    f"\nSaved reviewer manifest to: "
    f"{OUTPUT_PATH}"
)