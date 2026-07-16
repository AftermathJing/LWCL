#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
INPUT_ROOT="${WIDAR3_RAW_ROOT:-/home/ccl/data/csi-carat/widar3/widar3g6d/raw}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/home/wj/LWCL/data/raw/widar3_manifest.csv}"
FROZEN_SPLIT="${FROZEN_SPLIT:-$ROOT/data/splits/widar3_v2_subject_split.csv}"
WORKERS="${WORKERS:-8}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_variant() {
  local name="$1"
  local preprocess_config="$2"
  local train_config="$3"
  local processed="$ROOT/data/processed/widar3_tuning_$name"
  local split="$ROOT/data/splits/tuning/widar3_${name}_subject_split.csv"
  local audit="$processed/contract_audit.json"

  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.preprocess_widar3 \
    --config "$preprocess_config" \
    --input-root "$INPUT_ROOT" \
    --source-manifest "$SOURCE_MANIFEST" \
    --output-root "$processed" \
    --workers "$WORKERS"

  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.build_subject_split \
    --processed-manifest "$processed/manifest.csv" \
    --reference-manifest "$FROZEN_SPLIT" \
    --output "$split"

  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.audit_feature_contract \
    --manifest "$split" \
    --config "$train_config" \
    --output "$audit" \
    --strict
}

run_variant phase25 configs/tuning/preprocessing_phase25.yaml configs/tuning/signal_v2_current_plus_phase.yaml
run_variant dfs61 configs/tuning/preprocessing_dfs61.yaml configs/tuning/signal_v2_highres_dfs61.yaml
run_variant dfs121 configs/tuning/preprocessing_dfs121.yaml configs/tuning/signal_v2_highres_dfs121.yaml
