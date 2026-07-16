#!/usr/bin/env bash
set -euo pipefail

export LD_PRELOAD="${LD_PRELOAD:-/home/wj/miniconda3/envs/LWCL/lib/libstdc++.so.6}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
conda run --no-capture-output -n LWCL python -m lwcl.cli.train_wicbr \
  --config configs/widar3_wicbr_smoke.yaml \
  --output-dir outputs/widar3_wicbr_smoke
