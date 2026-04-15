#!/bin/bash
# Compare run directories between local verify/outputs/ and the Google Drive remote.
# Shows counts and lists dirs present on one side but not the other.
#
# Usage:
#   ./see_diff.sh
#
# Requires: rclone configured (same setup as sync_upload/download)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY_DIR="$(dirname "$SCRIPT_DIR")"

# Load verify/.env if present
if [ -f "$VERIFY_DIR/.env" ]; then
  export $(grep -v '^#' "$VERIFY_DIR/.env" | xargs)
fi

REMOTE="${GDRIVE_REMOTE:-lantern}"
REMOTE_PATH="${GDRIVE_REMOTE_PATH:-verify/outputs}"
OUTPUTS_DIR="$VERIFY_DIR/outputs"

echo "Local:  $OUTPUTS_DIR"
echo "Remote: $REMOTE:$REMOTE_PATH"
echo ""

# List local run dirs (exclude cache_*)
LOCAL=$(ls "$OUTPUTS_DIR" | grep -v '^cache_' | sort)
LOCAL_COUNT=$(echo "$LOCAL" | grep -c . || true)

# List remote run dirs (exclude cache_*)
REMOTE_LIST=$(rclone lsf "$REMOTE:$REMOTE_PATH" --dirs-only | sed 's|/$||' | grep -v '^cache_' | sort)
REMOTE_COUNT=$(echo "$REMOTE_LIST" | grep -c . || true)

echo "Local run dirs:  $LOCAL_COUNT"
echo "Remote run dirs: $REMOTE_COUNT"
echo ""

# Dirs only on remote (not local)
ONLY_REMOTE=$(comm -23 <(echo "$REMOTE_LIST") <(echo "$LOCAL"))
ONLY_REMOTE_COUNT=$(echo "$ONLY_REMOTE" | grep -c . || true)

# Dirs only local (not on remote)
ONLY_LOCAL=$(comm -13 <(echo "$REMOTE_LIST") <(echo "$LOCAL"))
ONLY_LOCAL_COUNT=$(echo "$ONLY_LOCAL" | grep -c . || true)

if [ "$ONLY_REMOTE_COUNT" -gt 0 ]; then
  echo "=== Only on remote ($ONLY_REMOTE_COUNT) ==="
  echo "$ONLY_REMOTE"
  echo ""
else
  echo "=== Only on remote: (none) ==="
  echo ""
fi

if [ "$ONLY_LOCAL_COUNT" -gt 0 ]; then
  echo "=== Only local ($ONLY_LOCAL_COUNT) ==="
  echo "$ONLY_LOCAL"
  echo ""
else
  echo "=== Only local: (none) ==="
  echo ""
fi

IN_SYNC=$((LOCAL_COUNT < REMOTE_COUNT ? LOCAL_COUNT : REMOTE_COUNT))
echo "In sync: ~$((LOCAL_COUNT < REMOTE_COUNT ? LOCAL_COUNT : REMOTE_COUNT)) dirs present on both sides"
