#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
GPU="${CUDA_DEVICE:-6}"
SEED="${SEED:-2025}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/screening}"
CONFIGS="${CONFIGS:-
configs/tuning/signal_v2_tuning_base.yaml
configs/tuning/signal_v2_hp_lr_low.yaml
configs/tuning/signal_v2_hp_lr_high.yaml
configs/tuning/signal_v2_hp_wd_low.yaml
configs/tuning/signal_v2_hp_wd_high.yaml
configs/tuning/signal_v2_hp_regularization_low.yaml
configs/tuning/signal_v2_hp_regularization_high.yaml
configs/tuning/signal_v2_hp_supcon_low.yaml
configs/tuning/signal_v2_hp_supcon_high.yaml
configs/tuning/signal_v2_hp_temperature_low.yaml
configs/tuning/signal_v2_hp_temperature_high.yaml
}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

for CONFIG in $CONFIGS; do
  NAME="$(basename "$CONFIG" .yaml)"
  OUTPUT="$OUTPUT_ROOT/$NAME"
  mkdir -p "$OUTPUT"
  if [[ -f "$OUTPUT/eval/validation_metrics.json" ]]; then
    echo "[tuning] skip completed $NAME"
    continue
  fi
  EXTRA_ARGS=()
  if [[ -n "${MAX_STEPS:-}" ]]; then
    EXTRA_ARGS+=(--max-steps "$MAX_STEPS")
  fi
  env CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.train \
    --config "$CONFIG" \
    --seed "$SEED" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$OUTPUT/train" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUTPUT/train.log"

  env CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.evaluate \
    --config "$OUTPUT/train/resolved_config.yaml" \
    --seed "$SEED" \
    --num-workers "$NUM_WORKERS" \
    --checkpoint "$OUTPUT/train/checkpoints/best.pt" \
    --output-dir "$OUTPUT/eval" \
    --split validation \
    --weights raw \
    2>&1 | tee "$OUTPUT/evaluate_validation.log"
done

