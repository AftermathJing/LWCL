#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
test -x "$CONDA_BIN"

if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(
    Qwen2.5-0.5B-Instruct
    Qwen2.5-1.5B-Instruct
    Qwen2.5-3B-Instruct
    Qwen2.5-7B-Instruct
  )
fi

for model in "${MODELS[@]}"; do
  repo="Qwen/${model}"
  target="$HOME/${model}"
  echo "Downloading ${repo} -> ${target}"
  mkdir -p "$target"
  "$CONDA_BIN" run --no-capture-output -n LWCL huggingface-cli download "$repo" \
    --local-dir "$target" \
    --local-dir-use-symlinks False
done
