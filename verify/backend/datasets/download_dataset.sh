#!/usr/bin/env bash
# download_dataset.sh
# Reads dataset_information.csv and downloads each dataset from Google Drive
# into verify/backend/datasets/<dataset_name>/
#
# Usage:
#   ./download_dataset.sh                   # download all datasets
#   ./download_dataset.sh SROIE2019         # download a specific dataset by name
#
# Requirements: gdown  (pip install gdown)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="$SCRIPT_DIR/dataset_information.csv"
DATASETS_DIR="$SCRIPT_DIR"
TARGET_NAME="${1:-}"  # optional: only download this dataset

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
if ! command -v gdown &>/dev/null; then
    echo "[ERROR] 'gdown' is not installed. Run: pip install gdown"
    exit 1
fi

if [[ ! -f "$CSV_FILE" ]]; then
    echo "[ERROR] CSV file not found: $CSV_FILE"
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: extract a bare ID or URL that gdown can handle
# ---------------------------------------------------------------------------
parse_drive_link() {
    local url="$1"

    # Folder: .../folders/<ID>
    if [[ "$url" =~ /folders/([A-Za-z0-9_-]+) ]]; then
        echo "folder:${BASH_REMATCH[1]}"
        return
    fi

    # File view: /file/d/<ID>/view  or  /file/d/<ID>
    if [[ "$url" =~ /file/d/([A-Za-z0-9_-]+) ]]; then
        echo "file:${BASH_REMATCH[1]}"
        return
    fi

    # Direct open/uc link: ?id=<ID>  or  open?id=<ID>
    if [[ "$url" =~ [?&]id=([A-Za-z0-9_-]+) ]]; then
        echo "file:${BASH_REMATCH[1]}"
        return
    fi

    echo "unknown:"
}

# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------
# Skip header line (dataset_name,google_drive_link,description)
tail -n +2 "$CSV_FILE" | while IFS=',' read -r name link description; do
    # Trim whitespace
    name="${name// /}"
    link="${link// /}"

    # Skip blank rows or comment rows
    [[ -z "$name" || "$name" == \#* ]] && continue

    # If a specific dataset was requested, skip others
    if [[ -n "$TARGET_NAME" && "$name" != "$TARGET_NAME" ]]; then
        continue
    fi

    dest="$DATASETS_DIR/$name"
    mkdir -p "$dest"

    parsed=$(parse_drive_link "$link")
    kind="${parsed%%:*}"
    id="${parsed#*:}"

    if [[ -z "$id" || "$kind" == "unknown" ]]; then
        echo "[SKIP] $name — could not parse Google Drive link: $link"
        continue
    fi

    echo "------------------------------------------------------------"
    echo "[INFO] Downloading: $name"
    echo "       Link  : $link"
    echo "       Dest  : $dest"

    if [[ "$kind" == "folder" ]]; then
        gdown --folder "https://drive.google.com/drive/folders/$id" \
              --output "$dest" \
              --remaining-ok
    else
        # Single file — download into the destination directory
        gdown "https://drive.google.com/uc?id=$id" \
              --output "$dest/" \
              --remaining-ok
    fi

    echo "[DONE] $name"
done

echo "============================================================"
echo "All requested downloads complete."
