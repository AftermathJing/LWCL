#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
cd "$ROOT"

"$CONDA" run --no-capture-output -n "$ENV_NAME" python -m pytest -q
"$CONDA" run --no-capture-output -n "$ENV_NAME" python -m pip check
"$CONDA" run --no-capture-output -n "$ENV_NAME" python -c \
  'import torch; print({"torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()})'
