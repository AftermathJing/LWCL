#!/usr/bin/env bash
set -euo pipefail

test -f pyproject.toml
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
test -x "$CONDA_BIN"

rm -rf data/processed/synthetic outputs/smoke

"$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.generate_synthetic \
  --output-dir data/processed/synthetic \
  --samples-per-label 12 \
  --seed 2025

"$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.make_splits \
  --input-manifest data/processed/synthetic/manifest.csv \
  --output-manifest data/processed/synthetic/manifest_split.csv \
  --group-field subject \
  --seed 2025

if "$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.train \
  --config configs/smoke.yaml \
  --output-dir outputs/smoke/emergency \
  --max-steps 3 \
  --debug-fail-after-step 2; then
  echo "emergency checkpoint smoke was expected to fail" >&2
  exit 1
fi
test -n "$(find outputs/smoke/emergency/checkpoints -name 'emergency_step_*.pt' -print -quit)"

"$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.train \
  --config configs/smoke.yaml \
  --output-dir outputs/smoke/initial \
  --max-steps 4

test -s outputs/smoke/initial/checkpoints/last.pt

"$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.train \
  --config configs/smoke.yaml \
  --output-dir outputs/smoke/resumed \
  --resume-from outputs/smoke/initial/checkpoints/last.pt \
  --max-steps 6

"$CONDA_BIN" run --no-capture-output -n LWCL python -m lwcl.cli.evaluate \
  --config configs/smoke.yaml \
  --checkpoint outputs/smoke/resumed/checkpoints/last.pt \
  --output-dir outputs/smoke/evaluation \
  --split test

test -s outputs/smoke/evaluation/test_metrics.json
"$CONDA_BIN" run -n LWCL python - <<'PY'
import torch
payload = torch.load("outputs/smoke/resumed/checkpoints/last.pt", map_location="cpu", weights_only=False)
assert payload["state"]["global_step"] == 6, payload["state"]
print("resume_global_step", payload["state"]["global_step"])
PY
echo "LWCL remote smoke completed"
