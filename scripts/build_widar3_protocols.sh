#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
MANIFEST="${MANIFEST_PATH:-$ROOT/data/processed/widar3_p1_full/manifest.csv}"
OUTPUT="${OUTPUT_ROOT:-$ROOT/data/splits/widar3_v2_protocols}"
SEED="${SEED:-2025}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
for PROTOCOL in id cl co ce cs; do
  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.build_protocol_splits \
    --manifest "$MANIFEST" \
    --output-root "$OUTPUT" \
    --protocol "$PROTOCOL" \
    --seed "$SEED"
done
