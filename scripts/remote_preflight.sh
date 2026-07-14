#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
test -x "$CONDA_BIN"

echo "commit=$(git rev-parse HEAD)"
echo "branch=$(git branch --show-current)"
echo "dirty_files=$(git status --porcelain | wc -l)"
nvidia-smi
"$CONDA_BIN" run --no-capture-output -n LWCL python -m pip check
"$CONDA_BIN" run --no-capture-output -n LWCL python -m pytest
