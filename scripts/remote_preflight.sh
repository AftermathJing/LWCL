#!/usr/bin/env bash
set -euo pipefail

echo "commit=$(git rev-parse HEAD)"
echo "branch=$(git branch --show-current)"
echo "dirty_files=$(git status --porcelain | wc -l)"
nvidia-smi
conda run --no-capture-output -n LWCL python -m pip check
conda run --no-capture-output -n LWCL python -m pytest
