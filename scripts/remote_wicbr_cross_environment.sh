#!/usr/bin/env bash
set -euo pipefail

export LD_PRELOAD="${LD_PRELOAD:-/home/wj/miniconda3/envs/LWCL/lib/libstdc++.so.6}"
CONDA_BIN="${CONDA_BIN:-/home/wj/miniconda3/bin/conda}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-auto}"

if [[ "$GPU_ID" == "auto" ]]; then
  GPU_ID="$($CONDA_BIN run --no-capture-output -n LWCL python scripts/select_free_gpu.py)"
fi
echo "[cross] using GPU $GPU_ID"

cd "$(dirname "$0")/.."

for fold_dir in data/splits/widar3_cross_environment/*; do
  if [[ ! -d "$fold_dir" ]]; then
    continue
  fi
  fold_name="$(basename "$fold_dir")"
  input_manifest="$fold_dir/manifest.csv"
  wicbr_manifest="$fold_dir/wicbr_manifest.csv"
  if [[ ! -f "$input_manifest" ]]; then
    continue
  fi
  if [[ ! -f "$wicbr_manifest" ]]; then
    "$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.attach_wicbr_paths \
      --input-manifest "$input_manifest" \
      --image-root data/processed/widar3_wicbr \
      --output-manifest "$wicbr_manifest" \
      --phase-source weighted
  fi

  output_root="outputs/widar3_wicbr_cross_environment/$fold_name"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.train_wicbr \
    --config configs/widar3_wicbr_subject.yaml \
    --manifest "$wicbr_manifest" \
    --output-dir "$output_root/train"

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.evaluate_wicbr \
    --config configs/widar3_wicbr_subject.yaml \
    --manifest "$wicbr_manifest" \
    --checkpoint "$output_root/train/checkpoints/best.pt" \
    --output-dir "$output_root/test_eval" \
    --split test
done
