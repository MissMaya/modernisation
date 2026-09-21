"""
Runs the complete table-construction pipeline from Stage 1 to Stage 8.

The eight construction scripts are listed by their full filenames.
This is to prevent future analysis from running scripts accidentally

The pipeline stops immediately if a construction stage fails. 
When all eight stages complete, the final outputs are:

    outputs/document_analysis.csv
    outputs/annotation_analysis.csv
"""

# ---------------------------------------------------------------------------
# Import the required modules
# ---------------------------------------------------------------------------

import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Set up the project paths and the construction stages in order
# ---------------------------------------------------------------------------

PROJECT_DIR = Path("/workspaces/modernisation")
SCRIPTS_DIR = PROJECT_DIR / "scripts"
OUTPUT_DIR = PROJECT_DIR / "outputs"

CONSTRUCTION_SCRIPTS = [
    "01_build_document_manifest.py",
    "02_build_file_manifest.py",
    "03_build_rule_manifest.py",
    "04_build_reviewer_manifest.py",
    "05_build_annotations.py",
    "06_resolve_invalid_combinations.py",
    "07_build_document_features.py",
    "08_assemble_analysis_tables.py",
]

FINAL_OUTPUTS = [
    OUTPUT_DIR / "document_analysis.csv",
    OUTPUT_DIR / "annotation_analysis.csv",
]


# ---------------------------------------------------------------------------
# Check that all eight table construction scripts exists before starting
# ---------------------------------------------------------------------------

missing_scripts = [
    script_name
    for script_name in CONSTRUCTION_SCRIPTS
    if not (SCRIPTS_DIR / script_name).is_file()
]

if missing_scripts:
    raise SystemExit(
        "The table-construction pipeline cannot start because these scripts "
        "were not found in the scripts directory:\n"
        + "\n".join(
            f"  {script_name}"
            for script_name in missing_scripts
        )
    )


# ---------------------------------------------------------------------------
# Run each script in order and stop immediately if one fails
# ---------------------------------------------------------------------------

print("\nStarting table-construction pipeline.")

for stage_number, script_name in enumerate(
    CONSTRUCTION_SCRIPTS,
    start = 1,
):
    script_path = SCRIPTS_DIR / script_name

    print(
        f"\n[{stage_number}/8] Running {script_name}",
        flush = True,
    )

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd = PROJECT_DIR,
            check = True,
        )

    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"\nTable construction stopped because {script_name} failed "
            f"with exit code {error.returncode}."
        ) from error


# ---------------------------------------------------------------------------
# Confirm that Stage 8 created both final tables
# ---------------------------------------------------------------------------

missing_outputs = [
    output_path
    for output_path in FINAL_OUTPUTS
    if not output_path.is_file()
]

if missing_outputs:
    raise SystemExit(
        "All construction scripts finished, but these final tables were not "
        "found:\n"
        + "\n".join(
            f"  {output_path}"
            for output_path in missing_outputs
        )
    )


print("\nTable construction completed successfully.")

for output_path in FINAL_OUTPUTS:
    print(f"  {output_path}")
