#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
GPU="${CUDA_DEVICE:-5}"
CONFIG="${CONFIG_PATH:?set CONFIG_PATH to a Signal-v2 experiment config}"
SEED="${SEED:-2025}"
OUTPUT="${OUTPUT_DIR:?set OUTPUT_DIR for this experiment run}"
MANIFEST="${MANIFEST_PATH:-}"
NUM_WORKERS="${NUM_WORKERS:-2}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/train" "$OUTPUT/eval"

MANIFEST_ARGS=()
if [[ -n "$MANIFEST" ]]; then
  MANIFEST_ARGS=(--manifest "$MANIFEST")
fi

if [[ -f "$OUTPUT/train/checkpoints/best.pt" ]]; then
  printf 'reusing existing best checkpoint: %s\n' "$OUTPUT/train/checkpoints/best.pt"
else
  RESUME_ARGS=()
  if [[ -f "$OUTPUT/train/checkpoints/last.pt" ]]; then
    RESUME_ARGS=(--resume-from "$OUTPUT/train/checkpoints/last.pt")
  elif find "$OUTPUT/train" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    printf 'refusing non-empty training directory without checkpoint: %s\n' "$OUTPUT/train" >&2
    exit 1
  fi
  env CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.train \
    --config "$CONFIG" \
    --seed "$SEED" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$OUTPUT/train" \
    "${MANIFEST_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$OUTPUT/train.log"
fi

for SPLIT in validation test; do
  env CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.evaluate \
    --config "$OUTPUT/train/resolved_config.yaml" \
    --seed "$SEED" \
    --num-workers "$NUM_WORKERS" \
    --checkpoint "$OUTPUT/train/checkpoints/best.pt" \
    --output-dir "$OUTPUT/eval" \
    --split "$SPLIT" \
    --weights raw \
    2>&1 | tee "$OUTPUT/evaluate_$SPLIT.log"
done

printf '0\n' > "$OUTPUT/exit_code"
