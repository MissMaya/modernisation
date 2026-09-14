"""
This is the first table that is constructed for the modernisation review.

We create a dataframe with 1 row for each of the 180 docuements with 
its filename stem, reviewer number, model type, original filename and
archive of provenance.
"""

# Import reeuired modules
from pathlib import Path
import pandas as pd


# Define the paths to the source data 
RAW_DATA_DIR = Path("/workspaces/modernisation/raw")
SAMPLE_PATH = RAW_DATA_DIR / "stratified_sample_180_gt_files.xlsx"
ASSIGNMENTS_PATH = (RAW_DATA_DIR / "modernisation_reviewer_packet_assignments.xlsx")


# Read in all of the columns from modernisation_reviewer_packet_assignments.xlsx
documents_df = pd.read_excel(
    ASSIGNMENTS_PATH,
    usecols = ["subfolder", "reviewer", "model"],
)

# Take the subfolder name as the canonical filename stem from now on 
documents_df = documents_df.rename(columns = {"subfolder": "filename_stem"})


# Read in only the filename and archive cols from the stratified-sample spreadsheet.
sample_df = pd.read_excel(
    SAMPLE_PATH,
    usecols = ["filename", "archive"],
)

# In the stratfied_sample worksheet, each filename ends in ``_GT.txt``
# Remove last 7 chars to create the same identifier used in the packet-assignment spreadsheet
sample_df["filename_stem"] = sample_df["filename"].str[:-7]


# Add the source filename and archive to the assignment data
# Left merge to preserve the packet-assignment table's rows and their original order
documents_df = documents_df.merge(
    sample_df,
    on = "filename_stem",
    how = "left",
)


# Reorder the columns 
documents_df = documents_df[
    ["filename_stem", "filename", "archive", "reviewer", "model"]
]

#Save intermediate output
INTERMEDIATE_DIR = Path("/workspaces/modernisation/intermediate")
INTERMEDIATE_DIR.mkdir(parents = True, exist_ok = True)
OUTPUT_PATH = INTERMEDIATE_DIR / "document_manifest.csv"

documents_df.to_csv(OUTPUT_PATH, index = False, encoding = "utf-8-sig")

# Sense checks and confirmation 
print(documents_df.head())
print(f"\nNumber of documents: {len(documents_df)}")
print(f"Missing archive matches: {documents_df['archive'].isna().sum()}")
print(f"Saved document manifest to: {OUTPUT_PATH}")
