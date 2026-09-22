"""
Removes all generated data and analysis outputs created by the pipeline.

Used during testing. Removes only these three directories:

    intermediate/
        Manifests and intermediate tables created by Stages 1 to 7.

    outputs/
        The document-level and annotation-level tables created by Stage 8.

    analysis_outputs/
        Tables, figures and explanatory files created by the analysis scripts.

The raw data, scripts, human-review decisions and requirements file are not
removed. The deleted directories are recreated by the pipeline scripts when they
are re-run
"""

# Import required modules
import shutil
from pathlib import Path


# Set up the project directory and specify the directories to remove
PROJECT_DIR = Path("/workspaces/modernisation")
GENERATED_DIRECTORIES = (
    PROJECT_DIR / "intermediate",
    PROJECT_DIR / "outputs",
    PROJECT_DIR / "analysis_outputs",
)


# Confirm that every deletion target is one of the three named directories
# directly inside the project. Stop rather than delete anything if a path has
# accidentally been changed to a broader or unrelated location.
allowed_directory_names = {
    "intermediate",
    "outputs",
    "analysis_outputs",
}

for directory in GENERATED_DIRECTORIES:
    if (
        directory.parent != PROJECT_DIR
        or directory.name not in allowed_directory_names
    ):
        raise RuntimeError(f"Unsafe to remove: {directory}")


# Remove each generated directory if it exists. A directory that is already
# absent is reported and does not cause the reset to fail.
for directory in GENERATED_DIRECTORIES:
    if directory.exists():
        shutil.rmtree(directory)
        print(f"Removed generated directory: {directory}")
    else:
        print(f"Directory already absent: {directory}")


print("\nProject data reset complete.")
