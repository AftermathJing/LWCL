#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
GPU="${CUDA_DEVICE:-5}"
CONFIG="${CONFIG_PATH:-configs/signal_v2_base.yaml}"
CHECKPOINT="${CHECKPOINT_PATH:?set CHECKPOINT_PATH to the frozen accepted best.pt}"
OUTPUT="${OUTPUT_DIR:-$ROOT/outputs/subject_error_analysis}"
FOCUS_SUBJECT="${FOCUS_SUBJECT:-user17}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT"

for SPLIT in train validation; do
  env CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.export_predictions \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT" \
    --split "$SPLIT" \
    --weights raw
done

"$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python -m lwcl_v2.cli.analyze_subject_errors \
  --outputs "$OUTPUT/validation_outputs.npz" \
  --focus-subject "$FOCUS_SUBJECT" \
  --reference-outputs "$OUTPUT/train_outputs.npz" \
  --output "$OUTPUT/${FOCUS_SUBJECT}_analysis.json"
