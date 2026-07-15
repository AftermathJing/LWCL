#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
TARGET="${TARGET_ENVIRONMENT:?TARGET_ENVIRONMENT is required}"
GPU="${CUDA_DEVICE:?CUDA_DEVICE is required}"
MANIFEST="$ROOT/data/splits/widar3_v2_cross_environment/$TARGET/manifest.csv"
OUTPUT="${OUTPUT_DIR:-$ROOT/outputs/widar3_v2_cross_environment/$TARGET}"
CONFIG="configs/widar3_v2_cross_environment.yaml"

mkdir -p "$OUTPUT"
trap 'status=$?; printf "%s\n" "$status" > "$OUTPUT/exit_code"' EXIT
cd "$ROOT"

env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python -m lwcl_v2.cli.train \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT"

env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python -m lwcl_v2.cli.evaluate \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --checkpoint "$OUTPUT/checkpoints/best.pt" \
  --output-dir "$OUTPUT/test" \
  --split test
