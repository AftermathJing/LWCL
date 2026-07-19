#!/usr/bin/env bash
set -euo pipefail

export LD_PRELOAD="${LD_PRELOAD:-/home/wj/miniconda3/envs/LWCL/lib/libstdc++.so.6}"
CONDA_BIN="${CONDA_BIN:-/home/wj/miniconda3/bin/conda}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-auto}"
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONFIG="${CONFIG:-configs/csi_bench_har_wicbr_earlystop.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/csi_bench_har_wicbr_earlystop}"
PROTOCOLS="${PROTOCOLS:-test_id test_cross_device test_cross_env test_cross_user test_cross_user_env}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$GPU_ID" == "auto" ]]; then
  GPU_ID="$($CONDA_BIN run --no-capture-output -n LWCL python scripts/select_free_gpu.py)"
fi
echo "[csi-bench-earlystop] using GPU $GPU_ID"
echo "[csi-bench-earlystop] config=$CONFIG output=$OUTPUT_ROOT"

test -f "$CONFIG"
for protocol in $PROTOCOLS; do
  test -f "data/splits/csi_bench_har_wicbr/$protocol/manifest.csv"
done

RESUME_ARGS=()
if [[ -f "$OUTPUT_ROOT/train_id/checkpoints/last.pt" ]]; then
  echo "[csi-bench-earlystop] resume from $OUTPUT_ROOT/train_id/checkpoints/last.pt"
  RESUME_ARGS=(--resume-from "$OUTPUT_ROOT/train_id/checkpoints/last.pt")
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" \
  "$CONDA_BIN" run --no-capture-output -n LWCL python -u -m lwcl.cli.train_wicbr \
    --config "$CONFIG" \
    --manifest data/splits/csi_bench_har_wicbr/test_id/manifest.csv \
    --output-dir "$OUTPUT_ROOT/train_id" \
    "${RESUME_ARGS[@]}"

test -s "$OUTPUT_ROOT/train_id/checkpoints/best.pt"
test -s "$OUTPUT_ROOT/train_id/checkpoints/last.pt"

for protocol in $PROTOCOLS; do
  echo "[csi-bench-earlystop] evaluating protocol=$protocol"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "$CONDA_BIN" run --no-capture-output -n LWCL python -u -m lwcl.cli.evaluate_wicbr \
      --config "$CONFIG" \
      --manifest "data/splits/csi_bench_har_wicbr/$protocol/manifest.csv" \
      --checkpoint "$OUTPUT_ROOT/train_id/checkpoints/best.pt" \
      --output-dir "$OUTPUT_ROOT/$protocol" \
      --split test
done

echo "[csi-bench-earlystop] finished"
