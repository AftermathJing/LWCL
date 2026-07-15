#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
RUN=("$CONDA" run --no-capture-output -n LWCL)
SPLIT_ROOT="data/splits/widar3_cross_environment"
OUTPUT_ROOT="outputs/widar3_cross_environment"
cd "$ROOT"

"${RUN[@]}" python -m lwcl.cli.make_cross_domain_splits \
  --input-manifest data/processed/widar3/manifest.csv \
  --output-dir "$SPLIT_ROOT" \
  --domain-field environment \
  --validation-ratio 0.15 \
  --seed 2025

for manifest in "$SPLIT_ROOT"/*/manifest.csv; do
  target="$(basename "$(dirname "$manifest")")"
  output_dir="$OUTPUT_ROOT/$target"
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to append to existing output directory: $output_dir" >&2
    exit 1
  fi
  echo "Starting held-out environment: $target"
  "${RUN[@]}" python -m lwcl.cli.train \
    --config configs/widar3_hste_classifier.yaml \
    --manifest "$manifest" \
    --output-dir "$output_dir"
  "${RUN[@]}" python -m lwcl.cli.evaluate \
    --config configs/widar3_hste_classifier.yaml \
    --manifest "$manifest" \
    --checkpoint "$output_dir/checkpoints/best.pt" \
    --output-dir "$output_dir/test" \
    --split test
done

echo "All Widar3 cross-environment folds completed"
