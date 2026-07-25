#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wj/miniconda3/envs/LWCL/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT/outputs/widar3_fstc_encoder_ablations_20260725}"
SEEDS_TEXT="${SEEDS:-2025 2026 2027 2028 2029}"
VARIANTS_TEXT="${VARIANTS:-full_control no_feature_encoder no_spatial_encoder no_temporal_encoder}"
GPUS_TEXT="${GPUS:-0 1 2 3 4 5 6}"
NUM_WORKERS="${NUM_WORKERS:-1}"
LD_PRELOAD_PATH="${LD_PRELOAD_PATH:-/home/wj/miniconda3/envs/LWCL/lib/libstdc++.so.6}"

cd "$ROOT"
mkdir -p "$RUNS_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

read -r -a SEEDS <<< "$SEEDS_TEXT"
read -r -a VARIANTS <<< "$VARIANTS_TEXT"
read -r -a GPUS <<< "$GPUS_TEXT"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  printf 'at least one GPU is required\n' >&2
  exit 1
fi

JOBS=()
for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    JOBS+=("$variant:$seed")
  done
done

run_job() {
  local variant="$1"
  local seed="$2"
  local gpu="$3"
  local config="configs/ablations/fstc_encoder/${variant}.yaml"
  local run_dir="$RUNS_ROOT/$variant/seed_$seed"
  local train_dir="$run_dir/train"
  local eval_dir="$run_dir/eval"
  mkdir -p "$train_dir" "$eval_dir"

  printf '[job-start] variant=%s seed=%s gpu=%s time=%s\n' \
    "$variant" "$seed" "$gpu" "$(date --iso-8601=seconds)"
  if [[ ! -f "$train_dir/checkpoints/best.pt" ]]; then
    local resume_args=()
    if [[ -f "$train_dir/checkpoints/last.pt" ]]; then
      resume_args=(--resume-from "$train_dir/checkpoints/last.pt")
      printf '[resume] %s\n' "$train_dir/checkpoints/last.pt"
    elif find "$train_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      printf 'refusing non-empty directory without checkpoint: %s\n' "$train_dir" >&2
      return 1
    fi
    env CUDA_VISIBLE_DEVICES="$gpu" \
      LD_PRELOAD="$LD_PRELOAD_PATH" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$PYTHON_BIN" -u -m lwcl_v2.ablations.train \
      --config "$config" \
      --seed "$seed" \
      --num-workers "$NUM_WORKERS" \
      --output-dir "$train_dir" \
      "${resume_args[@]}" \
      2>&1 | tee -a "$run_dir/train.log"
  else
    printf '[reuse-best] %s\n' "$train_dir/checkpoints/best.pt"
  fi

  for split in validation test; do
    env CUDA_VISIBLE_DEVICES="$gpu" \
      LD_PRELOAD="$LD_PRELOAD_PATH" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$PYTHON_BIN" -u -m lwcl_v2.ablations.evaluate \
      --config "$train_dir/resolved_config.yaml" \
      --seed "$seed" \
      --num-workers "$NUM_WORKERS" \
      --checkpoint "$train_dir/checkpoints/best.pt" \
      --output-dir "$eval_dir" \
      --split "$split" \
      --weights raw \
      2>&1 | tee "$run_dir/evaluate_$split.log"
  done
  printf '0\n' > "$run_dir/exit_code"
  printf '[job-end] variant=%s seed=%s gpu=%s time=%s\n' \
    "$variant" "$seed" "$gpu" "$(date --iso-8601=seconds)"
}

run_lane() {
  local lane="$1"
  local gpu="${GPUS[$lane]}"
  local status=0
  local index
  for ((index=lane; index<${#JOBS[@]}; index+=${#GPUS[@]})); do
    local job="${JOBS[$index]}"
    local variant="${job%%:*}"
    local seed="${job##*:}"
    if ! run_job "$variant" "$seed" "$gpu"; then
      printf '[job-failed] variant=%s seed=%s gpu=%s\n' "$variant" "$seed" "$gpu" >&2
      status=1
    fi
  done
  return "$status"
}

pids=()
for lane in "${!GPUS[@]}"; do
  run_lane "$lane" > "$RUNS_ROOT/lane_gpu_${GPUS[$lane]}.log" 2>&1 &
  pids+=("$!")
  printf 'lane gpu=%s pid=%s\n' "${GPUS[$lane]}" "$!"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  for variant in "${VARIANTS[@]}"; do
    "$PYTHON_BIN" -m lwcl_v2.cli.summarize_multiseed \
      --runs-root "$RUNS_ROOT/$variant" \
      --output "$RUNS_ROOT/$variant/multiseed_summary.json"
  done
fi
exit "$status"
