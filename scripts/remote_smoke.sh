#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
RUN=("$CONDA" run --no-capture-output -n "$ENV_NAME")
cd "$ROOT"

rm -rf data/synthetic_smoke outputs/smoke
"${RUN[@]}" python -m lwcl_v2.cli.generate_synthetic \
  --output-root data/synthetic_smoke --samples-per-split 96

"${RUN[@]}" python -m lwcl_v2.cli.audit_data \
  --manifest data/synthetic_smoke/manifest.csv --check-all-paths

"${RUN[@]}" python -m lwcl_v2.cli.train \
  --config configs/smoke.yaml \
  --output-dir outputs/smoke/initial \
  --max-steps 4

"${RUN[@]}" python -m lwcl_v2.cli.train \
  --config configs/smoke.yaml \
  --output-dir outputs/smoke/resumed \
  --resume-from outputs/smoke/initial/checkpoints/last.pt \
  --max-steps 6

"${RUN[@]}" python -m lwcl_v2.cli.evaluate \
  --config configs/smoke.yaml \
  --checkpoint outputs/smoke/resumed/checkpoints/last.pt \
  --output-dir outputs/smoke/evaluation \
  --split test

set +e
"${RUN[@]}" python -m lwcl_v2.cli.train \
  --config configs/smoke.yaml \
  --output-dir outputs/smoke/emergency \
  --max-steps 4 \
  --debug-fail-after-step 2
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  echo "Expected intentional emergency failure" >&2
  exit 1
fi

test -s outputs/smoke/initial/checkpoints/last.pt
test -s outputs/smoke/resumed/checkpoints/last.pt
test -s outputs/smoke/evaluation/test_metrics.json
find outputs/smoke/emergency/checkpoints -name 'emergency_step_*.pt' -print -quit | grep -q .
echo "LWCL-v2 smoke passed"
