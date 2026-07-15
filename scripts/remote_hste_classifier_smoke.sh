#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
RUN=("$CONDA" run --no-capture-output -n LWCL)
cd "$ROOT"

rm -rf outputs/hste_classifier_smoke

"${RUN[@]}" python -m lwcl.cli.train \
  --config configs/widar3_hste_classifier_smoke.yaml \
  --output-dir outputs/hste_classifier_smoke/initial \
  --max-steps 4

"${RUN[@]}" python -m lwcl.cli.train \
  --config configs/widar3_hste_classifier_smoke.yaml \
  --output-dir outputs/hste_classifier_smoke/resumed \
  --resume-from outputs/hste_classifier_smoke/initial/checkpoints/last.pt \
  --max-steps 6

"${RUN[@]}" python -m lwcl.cli.evaluate \
  --config configs/widar3_hste_classifier_smoke.yaml \
  --checkpoint outputs/hste_classifier_smoke/resumed/checkpoints/last.pt \
  --output-dir outputs/hste_classifier_smoke/evaluation \
  --split test

set +e
"${RUN[@]}" python -m lwcl.cli.train \
  --config configs/widar3_hste_classifier_smoke.yaml \
  --output-dir outputs/hste_classifier_smoke/emergency \
  --max-steps 4 \
  --debug-fail-after-step 2
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  echo "expected intentional emergency smoke failure" >&2
  exit 1
fi

test -f outputs/hste_classifier_smoke/initial/checkpoints/last.pt
test -f outputs/hste_classifier_smoke/resumed/checkpoints/last.pt
test -f outputs/hste_classifier_smoke/evaluation/test_metrics.json
find outputs/hste_classifier_smoke/emergency/checkpoints -name 'emergency_step_*.pt' -print -quit | grep -q .
echo "HSTE classifier smoke passed"
