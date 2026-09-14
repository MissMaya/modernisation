"""
Script used in testing to reset project to start position
"""

# Import required modules
import shutil
from pathlib import Path


# Set up the paths
PROJECT_DIR = Path("/workspaces/modernisation")
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"


# Take care to only delete the intermediate subfolder
if (
    INTERMEDIATE_DIR.name != "intermediate"
    or INTERMEDIATE_DIR.parent != PROJECT_DIR
):
    raise RuntimeError(
        f"Unsafe to remove: {INTERMEDIATE_DIR}"
    )


if INTERMEDIATE_DIR.exists():
    shutil.rmtree(INTERMEDIATE_DIR)

# Immediately recreate intermediate dir so the project is ready to run again 
INTERMEDIATE_DIR.mkdir(
    parents = True,
    exist_ok = True,
)

print(
    f"Folder successfully removed: "
    f"{INTERMEDIATE_DIR}"
)