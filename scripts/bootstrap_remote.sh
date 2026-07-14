#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required on the remote server" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx LWCL; then
  conda env update --name LWCL --file environment.yml --prune
else
  conda env create --file environment.yml
fi

conda run --no-capture-output -n LWCL python -m pip install --upgrade pip
conda run --no-capture-output -n LWCL python -m pip install -e '.[llm,dev]'
conda run --no-capture-output -n LWCL python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
