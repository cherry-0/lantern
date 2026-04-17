#!/bin/bash
# Upload verify/outputs/ to Google Drive.
#
# Usage:
#   ./sync_upload.sh      # upload all run dirs (cache_* included)
#
# Required env var (can also be set in verify/.env):
#   GDRIVE_FOLDER_ID   Drive folder ID to sync into

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY_DIR="$(dirname "$SCRIPT_DIR")"

# Load verify/.env if present
if [ -f "$VERIFY_DIR/.env" ]; then
  export $(grep -v '^#' "$VERIFY_DIR/.env" | xargs)
fi

python "$SCRIPT_DIR/gdrive_sync.py" upload "$@"
