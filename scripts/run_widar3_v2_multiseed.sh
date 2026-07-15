#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
GPU="${CUDA_DEVICE:-7}"
CONFIG="${CONFIG_PATH:-configs/signal_v2_base.yaml}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT/outputs/widar3_signal_v2_multiseed}"
SEEDS_TEXT="${SEEDS:-2026 2027 2028 2029}"

cd "$ROOT"
mkdir -p "$RUNS_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

read -r -a SEED_LIST <<< "$SEEDS_TEXT"
for SEED in "${SEED_LIST[@]}"; do
  RUN_DIR="$RUNS_ROOT/seed_$SEED"
  TRAIN_DIR="$RUN_DIR/train"
  EVAL_DIR="$RUN_DIR/eval"
  mkdir -p "$TRAIN_DIR" "$EVAL_DIR"

  if [[ -f "$TRAIN_DIR/checkpoints/best.pt" ]]; then
    printf 'seed %s already has best.pt; reusing checkpoint\n' "$SEED"
  else
    RESUME_ARGS=()
    if [[ -f "$TRAIN_DIR/checkpoints/last.pt" ]]; then
      RESUME_ARGS=(--resume-from "$TRAIN_DIR/checkpoints/last.pt")
      printf 'seed %s resuming from last.pt\n' "$SEED"
    elif find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      printf 'refusing non-empty training directory without checkpoint: %s\n' "$TRAIN_DIR" >&2
      exit 1
    fi
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$CONDA" run --no-capture-output -n "$ENV_NAME" \
      python -m lwcl_v2.cli.train \
      --config "$CONFIG" \
      --seed "$SEED" \
      --output-dir "$TRAIN_DIR" \
      "${RESUME_ARGS[@]}" \
      2>&1 | tee -a "$RUN_DIR/train.log"
  fi

  for SPLIT in validation test; do
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$CONDA" run --no-capture-output -n "$ENV_NAME" \
      python -m lwcl_v2.cli.evaluate \
      --config "$TRAIN_DIR/resolved_config.yaml" \
      --seed "$SEED" \
      --checkpoint "$TRAIN_DIR/checkpoints/best.pt" \
      --output-dir "$EVAL_DIR" \
      --split "$SPLIT" \
      --weights raw \
      2>&1 | tee "$RUN_DIR/evaluate_$SPLIT.log"
  done
  printf '0\n' > "$RUN_DIR/exit_code"
done

"$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python -m lwcl_v2.cli.summarize_multiseed \
  --runs-root "$RUNS_ROOT" \
  --output "$RUNS_ROOT/multiseed_summary.json"
