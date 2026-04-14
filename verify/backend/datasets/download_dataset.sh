#!/usr/bin/env bash
# download_dataset.sh
# Reads dataset_information.csv and downloads each dataset from Google Drive
# into verify/backend/datasets/<dataset_name>/ using rclone.
#
# Usage:
#   ./download_dataset.sh                   # download all datasets
#   ./download_dataset.sh SROIE2019         # download a specific dataset by name
#   RCLONE_REMOTE=lantern ./download_dataset.sh # use custom rclone remote
#
# Requirements: rclone (https://rclone.org/install.sh)
# Note: Requires an rclone remote configured for Google Drive (default name: 'gdrive')

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="$SCRIPT_DIR/dataset_information.csv"
DATASETS_DIR="$SCRIPT_DIR"
TARGET_NAME="${1:-}"  # optional: only download this dataset

# Remote name can be overridden by environment variable
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
if ! command -v rclone &>/dev/null; then
    echo "[ERROR] 'rclone' is not installed. Run: curl https://rclone.org/install.sh | sudo bash"
    exit 1
fi

if [[ ! -f "$CSV_FILE" ]]; then
    echo "[ERROR] CSV file not found: $CSV_FILE"
    exit 1
fi

# Check if rclone remote exists
if ! rclone listremotes | grep -q "^${RCLONE_REMOTE}:" ; then
    echo "[ERROR] rclone remote '${RCLONE_REMOTE}' not found."
    echo "        Configure it with 'rclone config' or set RCLONE_REMOTE environment variable."
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: extract a bare ID or URL that rclone can use
# ---------------------------------------------------------------------------
parse_drive_link() {
    local url="$1"
    local re_folder='/folders/([A-Za-z0-9_-]+)'
    local re_file='/file/d/([A-Za-z0-9_-]+)'
    local re_id='[?&]id=([A-Za-z0-9_-]+)'

    if [[ "$url" =~ $re_folder ]]; then
        echo "folder:${BASH_REMATCH[1]}"
        return
    fi

    if [[ "$url" =~ $re_file ]]; then
        echo "file:${BASH_REMATCH[1]}"
        return
    fi

    if [[ "$url" =~ $re_id ]]; then
        echo "file:${BASH_REMATCH[1]}"
        return
    fi

    echo "unknown:"
}

# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------
echo "[INFO] Reading datasets from: $CSV_FILE"

# Skip header line (dataset_name,google_drive_link,description)
tail -n +2 "$CSV_FILE" | while IFS=',' read -r name link description; do
    # Trim whitespace
    name="${name// /}"
    link="${link// /}"

    # Skip blank rows or comment rows
    [[ -z "$name" || "$name" == \#* ]] && continue

    echo "[CHECK] $name"

    # If a specific dataset was requested, skip others
    if [[ -n "$TARGET_NAME" && "$name" != "$TARGET_NAME" ]]; then
        echo "        -> skipped (not requested)"
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
    echo "       Kind  : $kind"
    echo "       ID    : $id"
    echo "       Link  : $link"
    echo "       Dest  : $dest"

    # Verify the resolved folder ID before downloading.
    # lsf with --max-depth 1 lists what rclone actually sees at that ID.
    # If this prints root-level Drive files instead of dataset files, the ID is wrong.
    # Use connection string override (remote,root_folder_id=ID:) to hardwire the
    # root at the remote level. The --drive-root-folder-id flag is unreliable when
    # other Drive flags (e.g. --drive-shared-with-me) override it internally.
    SCOPED_REMOTE="${RCLONE_REMOTE},root_folder_id=${id}"

    rclone copy "${SCOPED_REMOTE}:" "$dest" \
           --drive-skip-shortcuts \
           --drive-acknowledge-abuse \
           --progress \
           --fast-list

    echo "[DONE] $name"
done

echo "============================================================"
echo "All requested downloads complete."
