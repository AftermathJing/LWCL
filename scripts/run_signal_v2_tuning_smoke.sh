#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
GPU="${CUDA_DEVICE:-7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/wj/LWCL-v2.0-staging/smoke/signal_v2_tuning_checkpoint}"
CONFIG="configs/tuning/signal_v2_tuning_smoke.yaml"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Smoke output already exists: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

run_train() {
  env CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.train "$@"
}

run_train --config "$CONFIG" --output-dir "$OUTPUT_ROOT/initial" --max-steps 3
test -s "$OUTPUT_ROOT/initial/checkpoints/best.pt"
test -s "$OUTPUT_ROOT/initial/checkpoints/step_00000002.pt"
test -s "$OUTPUT_ROOT/initial/checkpoints/last.pt"

run_train \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_ROOT/resume" \
  --resume-from "$OUTPUT_ROOT/initial/checkpoints/last.pt" \
  --max-steps 5

"$CONDA" run --no-capture-output -n "$ENV_NAME" python -c \
  "import torch; p=torch.load('$OUTPUT_ROOT/resume/checkpoints/last.pt', map_location='cpu', weights_only=False); assert p['state']['global_step']==5, p['state']; print(p['state'])"

env CUDA_VISIBLE_DEVICES="$GPU" \
  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python -m lwcl_v2.cli.evaluate \
  --config "$OUTPUT_ROOT/initial/resolved_config.yaml" \
  --checkpoint "$OUTPUT_ROOT/initial/checkpoints/best.pt" \
  --output-dir "$OUTPUT_ROOT/eval" \
  --split validation \
  --weights raw \
  --num-workers 2
test -s "$OUTPUT_ROOT/eval/validation_metrics.json"

set +e
run_train \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_ROOT/emergency" \
  --max-steps 4 \
  --debug-fail-after-step 2
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  echo "Emergency checkpoint smoke unexpectedly succeeded" >&2
  exit 3
fi
test -s "$OUTPUT_ROOT/emergency/checkpoints/emergency_step_00000002.pt"

echo "Signal-v2 tuning smoke passed"

