#!/bin/bash
# Upload verify/outputs/ to Google Drive.
# Skips cache_* dirs and unchanged run dirs.
#
# Usage:
#   ./sync_upload.sh                        # upload all run dirs
#   ./sync_upload.sh --run <run_dir_name>   # upload one specific run dir
#   ./sync_upload.sh --include-cache        # also sync cache_* dirs
#
# Required env vars (can also be set in verify/.env):
#   GDRIVE_FOLDER_ID        Drive folder ID shared with the service account
#   GDRIVE_SERVICE_ACCOUNT  Path to service_account.json (default: verify/service_account.json)

set -e
cd "$(dirname "$0")"

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

python gdrive_sync.py upload "$@"
