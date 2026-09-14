"""
Creates a file-location manifest for the modernisation review data.

The script uses intermediate/document_manifest.csv as the authoritative list
of the 180 expected documents and their reviewer packet assignments.

It then:

1. Extracts raw/evaluacion_modernizador.zip if necessary.
2. Locates all nine reviewer packet directories inside the unzipped folder.
3. Finds and extracts any annotation-export ZIPs stored inside those packets.
4. Locates the two annotation-code legends associated with each packet:
   entities-legends_Modernizador.json (same file in each packet) and 
   annotations-legend.json (exactly the same as entities-legends_Modernizador.json but
   contains the name of the reviewer).
5. Locates each document's token-level annotation JSON and exported modernised
   TXT inside ANN_Documents.
6. Uses the ordinary packet-level modernised TXT as a fallback when the
   annotation export does not contain its TXT copy.
7. Compares the discovered filename stems and packet numbers against the
   documents expected in document_manifest.csv.
8. Records any missing, duplicate and unexpected files across the 9 packets.

The script creates two CSV outputs:

- intermediate/file_manifest.csv contains one row for each of the 180 expected
  documents. It records the available file paths, reviewer packet number and
  Boolean flags showing which required files were found.
- intermediate/file_manifest_issues.csv records each missing, duplicate or
  unexpected file discovered during the scan.

Nested annotation-export ZIPs are extracted into
intermediate/extracted_exports/. Paths stored in file_manifest.csv are relative
to the project directory, /workspaces/modernisation.

This script only finds, extracts and maps files. It does not read or process the
contents of the annotation JSONs, legend JSONs or modernised texts.
"""

# ---------------------------------------------------------------------------
# Import the required modules
# ---------------------------------------------------------------------------

import re
import zipfile
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup the paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path("/workspaces/modernisation")

RAW_DIR = PROJECT_DIR / "raw"
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate"

REVIEW_ZIP_PATH = RAW_DIR / "evaluacion_modernizador.zip"
EXTRACTED_REVIEW_DIR = RAW_DIR / "evaluacion_modernizador"

EXTRACTED_EXPORTS_DIR = (
    INTERMEDIATE_DIR / "extracted_exports"
)

DOCUMENT_MANIFEST_PATH = (
    INTERMEDIATE_DIR / "document_manifest.csv"
)

FILE_MANIFEST_PATH = (
    INTERMEDIATE_DIR / "file_manifest.csv"
)

ISSUE_REPORT_PATH = (
    INTERMEDIATE_DIR / "file_manifest_issues.csv"
)


# ---------------------------------------------------------------------------
# Setup the expected filenames
# ---------------------------------------------------------------------------

ANNOTATION_JSON_SUFFIX = "_GT_moderno.ann.json"

MODERNISED_TXT_SUFFIX = "_GT_moderno.txt"

ENTITIES_LEGEND_FILENAME = (
    "entities-legends_Modernizador.json"
)

ANNOTATIONS_LEGEND_FILENAME = (
    "annotations-legend.json"
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_reviewer_packet(packet_dir):
    """Extract the reviewer packet number from its directory name"""
    match = re.fullmatch(
        r"modernisation_reviewer_(\d+)_packet",
        packet_dir.name,
    )

    if match is None:
        return None

    return int(match.group(1))


def relative_path(file_path):
    """Return a project-relative path or a missing value"""
    if file_path is None:
        return pd.NA

    return file_path.relative_to(PROJECT_DIR).as_posix()


def record_issue(
    issues,
    reviewer_packet,
    issue,
    filename_stem = None,
    details = None,
):
    """Add a problem to the issue report."""
    issues.append(
        {
            "reviewer_packet": reviewer_packet,
            "filename_stem": filename_stem,
            "issue": issue,
            "details": details,
        }
    )


def extract_zip(zip_path, output_dir):
    """Extract a reviewer annotation-export ZIP into the intermediate folder"""
    output_dir.mkdir(
        parents = True,
        exist_ok = True,
    )

    # Extract the ZIP on every run, overwriting matching files from any previous
    # extraction. This ensures that the extracted directory is complete.
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(output_dir)


def find_ann_document_folders(search_roots):
    """Given a root, find ANN_Documents directories"""
    ann_folders = []

    for search_root in search_roots:
        ann_folders.extend(
            path
            for path in search_root.rglob("ANN_Documents")
            if path.is_dir()
        )

    return ann_folders


def select_single_path(
    paths,
    issues,
    reviewer_packet,
    issue_if_missing,
    issue_if_duplicate,
    filename_stem = None,
):
    """
    Select and return a file path if exactly one match is found.

    If no matches are found, record as a missing-file and return None. 
    If multiple matches are found, record as a duplicate-file issue
    and return None.
    """
    unique_paths = sorted(set(paths))

    if not unique_paths:
        record_issue(
            issues = issues,
            reviewer_packet = reviewer_packet,
            filename_stem = filename_stem,
            issue=issue_if_missing,
        )

        return None

    if len(unique_paths) > 1:
        record_issue(
            issues = issues,
            reviewer_packet = reviewer_packet,
            filename_stem = filename_stem,
            issue = issue_if_duplicate,
            details=" | ".join(
                str(path) for path in unique_paths
            ),
        )

        return None

    return unique_paths[0]


# ---------------------------------------------------------------------------
# Check that the required output from earlier processes exists 
# ---------------------------------------------------------------------------

if not DOCUMENT_MANIFEST_PATH.is_file():
    raise SystemExit(
        f"{DOCUMENT_MANIFEST_PATH.name} was not found in "
        f"{INTERMEDIATE_DIR}. "
        "Run 01_build_document_manifest.py first."
    )


# ---------------------------------------------------------------------------
# Locate the extracted review data or extract it from the main ZIP
# ---------------------------------------------------------------------------

if EXTRACTED_REVIEW_DIR.is_dir():
    print(
        f"Using existing review directory: "
        f"{EXTRACTED_REVIEW_DIR}"
    )

elif REVIEW_ZIP_PATH.is_file():
    with zipfile.ZipFile(
        REVIEW_ZIP_PATH,
        "r",
    ) as zip_file:
        zip_file.extractall(RAW_DIR)

    print(
        f"Extracted main review ZIP to: "
        f"{EXTRACTED_REVIEW_DIR}"
    )

else:
    raise SystemExit(
        "The review data could not be found. Expected either:\n"
        f"  {EXTRACTED_REVIEW_DIR}\n"
        "or:\n"
        f"  {REVIEW_ZIP_PATH}"
    )


# ---------------------------------------------------------------------------
# Load the document manifest (should be 180 docs) into a Pandas dataframe. 
# ---------------------------------------------------------------------------

documents_df = pd.read_csv(
    DOCUMENT_MANIFEST_PATH,
    dtype = {"filename_stem": "string"},
)

documents_df = documents_df.rename(
    columns = {"reviewer": "reviewer_packet"}
)

documents_df["reviewer_packet"] = (
    documents_df["reviewer_packet"].astype(int)
)


# ---------------------------------------------------------------------------
# Locate the nine directories containing the reviewer packets
# ---------------------------------------------------------------------------

packet_dirs = list(
    EXTRACTED_REVIEW_DIR.rglob(
        "modernisation_reviewer_*_packet"
    )
)

packet_dir_map = {
    get_reviewer_packet(packet_dir): packet_dir
    for packet_dir in packet_dirs
    if get_reviewer_packet(packet_dir) is not None
}


# ---------------------------------------------------------------------------
# Search each of the nine reviewer packets for the required files
# ---------------------------------------------------------------------------

manifest_rows = []
issues = []

expected_packets = sorted(
    documents_df["reviewer_packet"].unique()
)

for reviewer_packet in expected_packets:
    print(
        f"\nScanning reviewer packet "
        f"{reviewer_packet}"
    )

    packet_documents = documents_df.loc[
        documents_df["reviewer_packet"].eq(
            reviewer_packet
        ),
        "filename_stem",
    ].tolist()

    packet_dir = packet_dir_map.get(
        reviewer_packet
    )

    # If the entire packet directory is missing, record every affected document
    # and continue to the next packet.
    if packet_dir is None:
        record_issue(
            issues = issues,
            reviewer_packet = reviewer_packet,
            issue = "missing_packet_directory",
        )

        for filename_stem in packet_documents:
            manifest_rows.append(
                {
                    "filename_stem": filename_stem,
                    "reviewer_packet": reviewer_packet,
                    "annotation_json_path": pd.NA,
                    "modernised_txt_path": pd.NA,
                    "annotations_legend_path": pd.NA,
                    "entities_legend_path": pd.NA,
                    "annotation_json_found": False,
                    "modernised_txt_found": False,
                    "modernised_txt_in_export": False,
                    "annotations_legend_found": False,
                    "entities_legend_found": False,
                }
            )

        continue

    # Search at the top level of each packet 
    search_roots = [packet_dir]

    # Results may also be in zip files so find, extract and add path to search
    # roots if this is the case
    nested_zip_paths = sorted(
        packet_dir.rglob("*.zip")
    )

    for zip_number, zip_path in enumerate(
        nested_zip_paths,
        start=1,
    ):
        zip_output_dir = (
            EXTRACTED_EXPORTS_DIR
            / f"reviewer_packet_{reviewer_packet}"
            / f"{zip_number:02d}_{zip_path.stem}"
        )

        try:
            extract_zip(
                zip_path,
                zip_output_dir,
            )

            search_roots.append(
                zip_output_dir
            )

        except zipfile.BadZipFile:
            record_issue(
                issues = issues,
                reviewer_packet = reviewer_packet,
                issue = "invalid_nested_zip",
                details = str(zip_path),
            )

    # -----------------------------------------------------------------------
    # Locate the base and reviewer-specific annotation-code maps
    # -----------------------------------------------------------------------

    entities_legend_path = select_single_path(
        paths=list(
            packet_dir.rglob(
                ENTITIES_LEGEND_FILENAME
            )
        ),
        issues = issues,
        reviewer_packet = reviewer_packet,
        issue_if_missing = "missing_entities_legend",
        issue_if_duplicate = "duplicate_entities_legend",
    )

    annotations_legend_matches = []

    for search_root in search_roots:
        annotations_legend_matches.extend(
            search_root.rglob(
                ANNOTATIONS_LEGEND_FILENAME
            )
        )

    annotations_legend_path = select_single_path(
        paths = annotations_legend_matches,
        issues = issues,
        reviewer_packet = reviewer_packet,
        issue_if_missing = "missing_annotations_legend",
        issue_if_duplicate = "duplicate_annotations_legend",
    )

    # -----------------------------------------------------------------------
    # Locate the document-level .txt and accompanying .json annotations
    # -----------------------------------------------------------------------

    ann_folders = find_ann_document_folders(
        search_roots
    )

    if not ann_folders:
        record_issue(
            issues = issues,
            reviewer_packet = reviewer_packet,
            issue = "missing_ANN_Documents_directory",
        )

    annotation_json_paths = {}
    exported_txt_paths = {}

    for ann_folder in ann_folders:
        for json_path in ann_folder.glob(
            f"*{ANNOTATION_JSON_SUFFIX}"
        ):
            filename_stem = json_path.name.removesuffix(
                ANNOTATION_JSON_SUFFIX
            )

            annotation_json_paths.setdefault(
                filename_stem,
                [],
            ).append(json_path)

        for txt_path in ann_folder.glob(
            f"*{MODERNISED_TXT_SUFFIX}"
        ):
            filename_stem = txt_path.name.removesuffix(
                MODERNISED_TXT_SUFFIX
            )

            exported_txt_paths.setdefault(
                filename_stem,
                [],
            ).append(txt_path)

    expected_packet_stems = set(
        packet_documents
    )

    # Check that the docs found in the annotation export folder match the 
    # filenames assigned to the packet that are listed in the document manifest
    # Record any unexpected filenames 
    unexpected_annotation_stems = (
        set(annotation_json_paths)
        - expected_packet_stems
    )

    unexpected_txt_stems = (
        set(exported_txt_paths)
        - expected_packet_stems
    )

    for filename_stem in sorted(
        unexpected_annotation_stems
    ):
        record_issue(
            issues = issues,
            reviewer_packet = reviewer_packet,
            filename_stem = filename_stem,
            issue = "unexpected_annotation_json",
        )

    for filename_stem in sorted(
        unexpected_txt_stems
    ):
        record_issue(
            issues = issues,
            reviewer_packet = reviewer_packet,
            filename_stem = filename_stem,
            issue = "unexpected_exported_txt",
        )

    # -----------------------------------------------------------------------
    # For each packet, create a file manifest with one row per document
    # -----------------------------------------------------------------------

    for filename_stem in packet_documents:
        annotation_json_path = select_single_path(
            paths = annotation_json_paths.get(
                filename_stem,
                [],
            ),
            issues = issues,
            reviewer_packet = reviewer_packet,
            filename_stem = filename_stem,
            issue_if_missing = "missing_annotation_json",
            issue_if_duplicate = "duplicate_annotation_json",
        )

        exported_txt_path = select_single_path(
            paths = exported_txt_paths.get(
                filename_stem,
                [],
            ),
            issues = issues,
            reviewer_packet = reviewer_packet,
            filename_stem = filename_stem,
            issue_if_missing = "missing_exported_txt",
            issue_if_duplicate = "duplicate_exported_txt",
        )

        # If the annotation export lacks its TXT, look for the ordinary
        # modernised TXT supplied in the reviewer packet as a fallback
        if exported_txt_path is None:
            ordinary_txt_matches = list(
                packet_dir.rglob(
                    f"{filename_stem}"
                    f"{MODERNISED_TXT_SUFFIX}"
                )
            )

            modernised_txt_path = select_single_path(
                paths = ordinary_txt_matches,
                issues = issues,
                reviewer_packet = reviewer_packet,
                filename_stem = filename_stem,
                issue_if_missing = "missing_modernised_txt",
                issue_if_duplicate = "duplicate_modernised_txt",
            )

        else:
            modernised_txt_path = exported_txt_path

        manifest_rows.append(
            {
                "filename_stem": filename_stem,
                "reviewer_packet": reviewer_packet,
                "annotation_json_path": relative_path(
                    annotation_json_path
                ),
                "modernised_txt_path": relative_path(
                    modernised_txt_path
                ),
                "annotations_legend_path": relative_path(
                    annotations_legend_path
                ),
                "entities_legend_path": relative_path(
                    entities_legend_path
                ),
                "annotation_json_found": (
                    annotation_json_path is not None
                ),
                "modernised_txt_found": (
                    modernised_txt_path is not None
                ),
                "modernised_txt_in_export": (
                    exported_txt_path is not None
                ),
                "annotations_legend_found": (
                    annotations_legend_path is not None
                ),
                "entities_legend_found": (
                    entities_legend_path is not None
                ),
            }
        )


# ---------------------------------------------------------------------------
# Save the file manifest and the report with any issues found across the reviewer packets
# ---------------------------------------------------------------------------

INTERMEDIATE_DIR.mkdir(
    parents = True,
    exist_ok = True,
)

file_manifest_df = (
    pd.DataFrame(manifest_rows)
    .sort_values(
        ["reviewer_packet", "filename_stem"]
    )
    .reset_index(drop = True)
)

file_manifest_df.to_csv(
    FILE_MANIFEST_PATH,
    index = False,
    encoding = "utf-8-sig",
)

issue_columns = [
    "reviewer_packet",
    "filename_stem",
    "issue",
    "details",
]

issues_df = pd.DataFrame(
    issues,
    columns = issue_columns,
)

issues_df.to_csv(
    ISSUE_REPORT_PATH,
    index = False,
    encoding = "utf-8-sig",
)


# ---------------------------------------------------------------------------
# Quality and sense checks 
# ---------------------------------------------------------------------------

print(
    f"\nDocuments in file manifest: "
    f"{len(file_manifest_df)}"
)

print(
    "\nDocuments per reviewer packet:"
)

print(
    file_manifest_df["reviewer_packet"]
    .value_counts()
    .sort_index()
    .to_string()
)

if issues_df.empty:
    print(
        "\nNo file-manifest issues were found."
    )

else:
    print(
        f"\nFile-manifest issues found: "
        f"{len(issues_df)}"
    )

    print("\nIssues by type:")

    print(
        issues_df["issue"]
        .value_counts()
        .to_string()
    )

    print("\nIssues by reviewer packet:")

    print(
        issues_df["reviewer_packet"]
        .value_counts()
        .sort_index()
        .to_string()
    )

print(
    f"\nSaved file manifest to: "
    f"{FILE_MANIFEST_PATH}"
)

print(
    f"Saved issue report to: "
    f"{ISSUE_REPORT_PATH}"
)