#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
cd "$ROOT"

if "$CONDA" env list | awk '{print $1}' | grep -qx 'LWCL-v2'; then
  "$CONDA" env update -n LWCL-v2 -f environment.yml --prune
else
  "$CONDA" env create -f environment.yml
fi
"$CONDA" run --no-capture-output -n LWCL-v2 python -m pip install -e .
"$CONDA" run --no-capture-output -n LWCL-v2 python -m pip check
