#!/bin/bash
set -euo pipefail

# Invalidate all Python cache files under a directory.
# Usage: ./invalidate_cache.sh [directory]

ROOT="${1:-.}"

if [[ ! -d "$ROOT" ]]; then
  echo "Error: not a directory: $ROOT" >&2
  exit 1
fi

echo "Deleting Python cache files under: $ROOT"

find "$ROOT" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
find "$ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} +
