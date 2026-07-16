#!/usr/bin/env bash
set -euo pipefail

export LD_PRELOAD="${LD_PRELOAD:-/home/wj/miniconda3/envs/LWCL/lib/libstdc++.so.6}"
CONDA_BIN="${CONDA_BIN:-/home/wj/miniconda3/bin/conda}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-auto}"
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/home/wj/LWCL/data/splits/widar3_wicbr_subject_split.csv}"
OFFICIAL_SPLIT_DIR="${OFFICIAL_SPLIT_DIR:-$ROOT_DIR/tmp/widar3_wicbr_official}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/widar3_wicbr_official}"
PROTOCOLS="${PROTOCOLS:-cr1 cr2 cr3 indom cl co}"

if [[ "$GPU_ID" == "auto" ]]; then
  GPU_ID="$($CONDA_BIN run --no-capture-output -n LWCL python "$ROOT_DIR/scripts/select_free_gpu.py")"
fi

echo "[official] using GPU $GPU_ID"
echo "[official] root=$ROOT_DIR"

mkdir -p "$OFFICIAL_SPLIT_DIR" "$OUTPUT_ROOT"

$CONDA_BIN run --no-capture-output -n LWCL python -m lwcl.cli.build_widar3_wicbr_official_splits \
  --input-manifest "$SOURCE_MANIFEST" \
  --output-dir "$OFFICIAL_SPLIT_DIR"

for protocol in $PROTOCOLS; do
  manifest="$OFFICIAL_SPLIT_DIR/$protocol.csv"
  if [[ ! -f "$manifest" ]]; then
    echo "[official] skip $protocol (missing manifest)"
    continue
  fi

  output_dir="$OUTPUT_ROOT/$protocol"
  train_dir="$output_dir/train"
  test_dir="$output_dir/test_eval"
  config_path="$ROOT_DIR/configs/widar3_wicbr_official.yaml"

  echo "[official] start protocol=$protocol manifest=$manifest"

  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  "$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.train_wicbr \
    --config "$config_path" \
    --manifest "$manifest" \
    --output-dir "$train_dir"

  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  "$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.evaluate_wicbr \
    --config "$config_path" \
    --manifest "$manifest" \
    --checkpoint "$train_dir/checkpoints/best.pt" \
    --output-dir "$test_dir" \
    --split test

  echo "[official] done protocol=$protocol"
done

echo "[official] all requested protocols finished"
