#!/bin/bash
# Download newer run dirs from Google Drive to verify/outputs/.
# Only downloads dirs where Drive version is newer than local manifest record.
#
# Usage:
#   ./sync_download.sh                        # check + download all
#   ./sync_download.sh --run <run_dir_name>   # download one specific run dir
#
# Required env vars (can also be set in verify/.env):
#   GDRIVE_FOLDER_ID        Drive folder ID shared with the service account
#   GDRIVE_SERVICE_ACCOUNT  Path to service_account.json (default: verify/service_account.json)

set -e
cd "$(dirname "$0")"

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

python gdrive_sync.py download "$@"
