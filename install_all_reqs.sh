#!/usr/bin/env bash
# Install all dependencies for target-apps
set -e

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$REPO_ROOT/target-apps"

# echo "=== clone/server (pip) ==="
# pip install -r "$TARGET/clone/server/requirements.txt"

# echo ""
# echo "=== clone/vectordb (pip) ==="
# pip install -r "$TARGET/clone/vectordb/requirements.txt"

# echo ""
# echo "=== clone/frontend (npm) ==="
# (cd "$TARGET/clone/frontend" && npm install)

echo ""
echo "=== momentag/backend (uv) ==="
(cd "$TARGET/momentag/backend" && uv sync)

echo ""
echo "=== momentag/tag-search (uv) ==="
(cd "$TARGET/momentag/tag-search" && uv sync)

echo ""
echo "=== snapdo (pip) ==="
pip install -r "$TARGET/snapdo/requirements.txt"

echo ""
echo "=== xend/backend (poetry) ==="
(cd "$TARGET/xend/backend" && poetry install)

echo ""
echo "=== xend/gpu-server (pip) ==="
pip install -r "$TARGET/xend/gpu-server/requirements.txt"

echo ""
echo "All dependencies installed."
