#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
GPU="${CUDA_DEVICE:-7}"
OUTPUT="${OUTPUT_DIR:-$ROOT/outputs/widar3_signal_v2_base_seed2025}"
CONFIG="${CONFIG_PATH:-configs/signal_v2_base.yaml}"

mkdir -p "$OUTPUT"
trap 'status=$?; printf "%s\n" "$status" > "$OUTPUT/exit_code"' EXIT
cd "$ROOT"

printf '%q ' env \
  "CUDA_VISIBLE_DEVICES=$GPU" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python -m lwcl_v2.cli.train \
  --config "$CONFIG" \
  --output-dir "$OUTPUT"
printf '\n'

env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python -m lwcl_v2.cli.train \
  --config "$CONFIG" \
  --output-dir "$OUTPUT"
