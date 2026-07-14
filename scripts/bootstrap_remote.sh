#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-}"
if [[ -z "$CONDA_BIN" ]]; then
  CONDA_BIN="$(command -v conda 2>/dev/null || true)"
fi
for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
  if [[ -z "$CONDA_BIN" && -x "$candidate" ]]; then
    CONDA_BIN="$candidate"
  fi
done
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "conda is required on the remote server" >&2
  exit 1
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx LWCL; then
  "$CONDA_BIN" env update --name LWCL --file environment.yml --prune
else
  "$CONDA_BIN" env create --file environment.yml
fi

"$CONDA_BIN" run --no-capture-output -n LWCL python -m pip install --upgrade pip
"$CONDA_BIN" run --no-capture-output -n LWCL python -m pip install -e '.[llm,dev]'
"$CONDA_BIN" run --no-capture-output -n LWCL python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
