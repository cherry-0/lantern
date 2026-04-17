#!/bin/bash
# Download newer run dirs from Google Drive to verify/outputs/.
# Only downloads files where the Drive version is newer than local.
#
# Usage:
#   ./sync_download.sh    # check + download all
#
# Required env var (can also be set in verify/.env):
#   GDRIVE_FOLDER_ID   Drive folder ID to sync from

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY_DIR="$(dirname "$SCRIPT_DIR")"

# Load verify/.env if present
if [ -f "$VERIFY_DIR/.env" ]; then
  export $(grep -v '^#' "$VERIFY_DIR/.env" | xargs)
fi

python "$SCRIPT_DIR/gdrive_sync.py" download "$@"
