#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_ROOT="${CSI_BENCH_ROOT:-/home/wj/LWCL/data/raw/CSI-Bench}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT/data/splits/csi_bench_har}"
CONFIG="${CONFIG_PATH:-configs/csi_bench/t00_har_base.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/wj/LWCL-v2.0-staging/outputs/csi_bench_t00_har}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

cd "$ROOT"
python -m lwcl_v2.cli.build_csi_bench_protocols \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$MANIFEST_ROOT"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u -m lwcl_v2.cli.train \
  --config "$CONFIG" \
  --manifest "$MANIFEST_ROOT/test_id/manifest.csv" \
  --output-dir "$OUTPUT_ROOT/train_id"

for protocol in test_id test_cross_device test_cross_env test_cross_user test_cross_user_env; do
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -u -m lwcl_v2.cli.evaluate \
    --config "$CONFIG" \
    --manifest "$MANIFEST_ROOT/$protocol/manifest.csv" \
    --checkpoint "$OUTPUT_ROOT/train_id/checkpoints/best.pt" \
    --output-dir "$OUTPUT_ROOT/$protocol" \
    --split test \
    --weights raw
done
