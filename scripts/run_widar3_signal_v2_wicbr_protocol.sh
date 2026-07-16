#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
GPU="${CUDA_DEVICE:-6}"
SEED="${SEED:-2025}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PROTOCOLS="${PROTOCOLS:-cr1 cr2 cr3}"
PROCESSED_MANIFEST="${PROCESSED_MANIFEST:-$ROOT/data/processed/widar3_p1_full/manifest.csv}"
OFFICIAL_SPLIT_DIR="${OFFICIAL_SPLIT_DIR:-/home/wj/LWCL-official-protocol/tmp/widar3_wicbr_official}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT/data/splits/widar3_signal_v2_wicbr_protocol}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/widar3_signal_v2_wicbr_protocol}"
CONFIG="${CONFIG_PATH:-configs/widar3_signal_v2_wicbr_protocol.yaml}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python -m lwcl_v2.cli.build_wicbr_protocol_manifests \
  --processed-manifest "$PROCESSED_MANIFEST" \
  --official-split-dir "$OFFICIAL_SPLIT_DIR" \
  --output-dir "$MANIFEST_ROOT" \
  --protocols $PROTOCOLS

for PROTOCOL in $PROTOCOLS; do
  MANIFEST="$MANIFEST_ROOT/$PROTOCOL/manifest.csv"
  OUTPUT_DIR="$OUTPUT_ROOT/$PROTOCOL"
  mkdir -p "$OUTPUT_DIR"

  env CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.train \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --seed "$SEED" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$OUTPUT_DIR/train" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

  for SPLIT in validation test; do
    env CUDA_VISIBLE_DEVICES="$GPU" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$CONDA" run --no-capture-output -n "$ENV_NAME" \
      python -m lwcl_v2.cli.evaluate \
      --config "$OUTPUT_DIR/train/resolved_config.yaml" \
      --manifest "$MANIFEST" \
      --seed "$SEED" \
      --num-workers "$NUM_WORKERS" \
      --checkpoint "$OUTPUT_DIR/train/checkpoints/best.pt" \
      --output-dir "$OUTPUT_DIR/eval" \
      --split "$SPLIT" \
      --weights raw \
      2>&1 | tee "$OUTPUT_DIR/evaluate_$SPLIT.log"
  done
done
