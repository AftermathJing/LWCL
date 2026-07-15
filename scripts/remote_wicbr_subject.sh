#!/usr/bin/env bash
set -euo pipefail

export LD_PRELOAD="${LD_PRELOAD:-/home/wj/miniconda3/envs/LWCL/lib/libstdc++.so.6}"
CONDA_BIN="${CONDA_BIN:-/home/wj/miniconda3/bin/conda}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-auto}"

cd "$(dirname "$0")/.."

if [[ "$GPU_ID" == "auto" ]]; then
  GPU_ID="$($CONDA_BIN run --no-capture-output -n LWCL python scripts/select_free_gpu.py)"
fi
echo "[subject] using GPU $GPU_ID"

if [[ ! -f data/splits/widar3_wicbr_subject_split.csv ]]; then
  "$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.attach_wicbr_paths \
    --input-manifest data/splits/widar3_subject_split.csv \
    --image-root data/processed/widar3_wicbr \
    --output-manifest data/splits/widar3_wicbr_subject_split.csv \
    --phase-source weighted
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" \
"$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.train_wicbr \
  --config configs/widar3_wicbr_subject.yaml \
  --manifest data/splits/widar3_wicbr_subject_split.csv \
  --output-dir outputs/widar3_wicbr_subject

CUDA_VISIBLE_DEVICES="$GPU_ID" \
"$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.evaluate_wicbr \
  --config configs/widar3_wicbr_subject.yaml \
  --manifest data/splits/widar3_wicbr_subject_split.csv \
  --checkpoint outputs/widar3_wicbr_subject/checkpoints/best.pt \
  --output-dir outputs/widar3_wicbr_subject_test \
  --split test
