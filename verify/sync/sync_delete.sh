#!/bin/bash
# Propagate local deletions to Google Drive.
#
# Compares local verify/outputs/ against the remote and deletes any remote
# directories that no longer exist locally.
#
# Usage:
#   ./sync_delete.sh              # dry-run: show what would be deleted
#   ./sync_delete.sh --execute    # actually delete remote-only dirs
#   ./sync_delete.sh --verbose    # also list local-only (not yet uploaded) dirs
#
# Optional env vars (can also be set in verify/.env):
#   GDRIVE_REMOTE       rclone remote name       (default: lantern)
#   GDRIVE_REMOTE_PATH  path inside the remote   (default: verify/outputs)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY_DIR="$(dirname "$SCRIPT_DIR")"

# Load verify/.env if present
if [ -f "$VERIFY_DIR/.env" ]; then
  export $(grep -v '^#' "$VERIFY_DIR/.env" | xargs)
fi

python "$SCRIPT_DIR/sync_delete.py" "$@"
