"""
Creates the annotation rule manifest.

The file-location manifest contains the paths to the annotation-code legends
used in each reviewer packet. Because entities-legends_Modernizador.json is
the same across all packets, this script reads one available copy and creates
one row for each entity-field combination.

The output is saved as intermediate/rule_manifest.csv.
"""
"""
This script creates a table that maps the annotation codes
used by the annotator for the modernisation to their meanings.

The annotation JSON files identify errors using codes such as ``e_3`` and
``f_31``. This script converts those codes into a simple reference table
containing:

- the error-category code;
- the error-category name;
- the field or sub-rule code; and
- the field or sub-rule name.

The paths to the annotation-code maps are stored in file_manifest.csv. 
The same code map was supplied to every reviewer so the script reads one available
copy of entities-legends_Modernizador.json.

Each row in the resulting table represents one error category and field
combination. `Without Fields` entries are retained.

The completed table is saved as intermediate/rule_manifest.csv.
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
    INTERMEDIATE_DIR / "rule_manifest.csv"
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
# Load the file manifest and select one of the available annotation code legends
# ---------------------------------------------------------------------------

file_manifest_df = pd.read_csv(
    FILE_MANIFEST_PATH
)

legend_paths = (
    file_manifest_df["entities_legend_path"]
    .dropna()
    .drop_duplicates()
)

if legend_paths.empty:
    raise SystemExit(
        "No entities-legends_Modernizador.json path was found "
        "in file_manifest.csv."
    )

# The entities legend is identical across reviewer packets
# so we just need one available copy to build the rule manifest.
legend_path = (
    PROJECT_DIR / legend_paths.iloc[0]
)

print(
    f"Using annotation-code legend: "
    f"{legend_path}"
)


# ---------------------------------------------------------------------------
# Read the annotation-code legend
# ---------------------------------------------------------------------------

with legend_path.open(
    "r",
    encoding = "utf-8-sig",
) as json_file:
    legend_data = json.load(json_file)


# ---------------------------------------------------------------------------
# Create one row for each field within each annotation entity
# ---------------------------------------------------------------------------

rule_rows = []

for entity in legend_data["entities"]:
    for field in entity["fieldList"]:
        rule_rows.append(
            {
                "error_category": entity["entity"],
                "error_code": entity["code"],
                "subrule": field["name"],
                "field_code": field["code"],
            }
        )


rules_df = pd.DataFrame(
    rule_rows
)


# ---------------------------------------------------------------------------
# Some quick sense checks
# ---------------------------------------------------------------------------

print(
    f"\nTop-level error categories: "
    f"{rules_df['error_code'].nunique()}"
)

print(
    f"Entity-field combinations: "
    f"{len(rules_df)}"
)

print(
    f"Duplicate field codes: "
    f"{rules_df['field_code'].duplicated().sum()}"
)


# ---------------------------------------------------------------------------
# Save the rule manifest
# ---------------------------------------------------------------------------

rules_df.to_csv(
    OUTPUT_PATH,
    index = False,
    encoding = "utf-8-sig",
)

print(
    f"\nSaved rule manifest to: "
    f"{OUTPUT_PATH}"
)