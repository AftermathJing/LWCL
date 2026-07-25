#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wj/miniconda3/envs/LWCL/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT/outputs/widar3_fstc_encoder_crossdomain_testacc_20260725}"
CLCO_ROOT="${CLCO_ROOT:-/home/wj/LWCL-v2.0-staging/data/splits/widar3_v2_protocols_testacc}"
CE_ROOT="${CE_ROOT:-/home/wj/LWCL-v2.0-staging/data/splits/widar3_signal_v2_wicbr_protocol}"
SEED="${SEED:-2025}"
VARIANTS_TEXT="${VARIANTS:-full_control no_feature_encoder no_spatial_encoder no_temporal_encoder}"
GPUS_TEXT="${GPUS:-0 1 2 3 4 5 6}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LD_PRELOAD_PATH="${LD_PRELOAD_PATH:-/home/wj/miniconda3/envs/LWCL/lib/libstdc++.so.6}"

cd "$ROOT"
mkdir -p "$RUNS_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec 9>"$RUNS_ROOT/.launcher.lock"
if ! flock -n 9; then
  printf 'another cross-domain ablation launcher is already running for %s\n' "$RUNS_ROOT" >&2
  exit 1
fi

read -r -a VARIANTS <<< "$VARIANTS_TEXT"
read -r -a GPUS <<< "$GPUS_TEXT"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  printf 'at least one GPU is required\n' >&2
  exit 1
fi

JOBS=()
for variant in "${VARIANTS[@]}"; do
  for protocol in CL CO; do
    for manifest in "$CLCO_ROOT/$protocol"/test_*/manifest.csv; do
      [[ -f "$manifest" ]] || continue
      fold="$(basename "$(dirname "$manifest")")"
      JOBS+=("$variant|$protocol|$fold|$manifest")
    done
  done
  for manifest in "$CE_ROOT"/cr*/manifest.csv; do
    [[ -f "$manifest" ]] || continue
    fold="$(basename "$(dirname "$manifest")")"
    JOBS+=("$variant|CE|$fold|$manifest")
  done
done

expected_jobs=$((${#VARIANTS[@]} * 13))
if [[ "${#JOBS[@]}" -ne "$expected_jobs" ]]; then
  printf 'expected %s jobs but discovered %s\n' "$expected_jobs" "${#JOBS[@]}" >&2
  exit 1
fi
printf '%s\n' "${JOBS[@]}" > "$RUNS_ROOT/job_manifest.txt"

run_job() {
  local variant="$1"
  local protocol="$2"
  local fold="$3"
  local manifest="$4"
  local gpu="$5"
  local config="configs/ablations/fstc_encoder/${variant}_testacc.yaml"
  local run_dir="$RUNS_ROOT/$variant/$protocol/$fold"
  local train_dir="$run_dir/train"
  local eval_dir="$run_dir/test_eval"
  mkdir -p "$train_dir" "$eval_dir"

  printf '[job-start] variant=%s protocol=%s fold=%s gpu=%s time=%s\n' \
    "$variant" "$protocol" "$fold" "$gpu" "$(date --iso-8601=seconds)"
  if [[ ! -f "$train_dir/runtime_report.json" ]]; then
    local resume_args=()
    if [[ -f "$train_dir/checkpoints/last.pt" ]]; then
      resume_args=(--resume-from "$train_dir/checkpoints/last.pt")
      printf '[resume] %s\n' "$train_dir/checkpoints/last.pt"
    elif find "$train_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      printf 'refusing non-empty directory without resumable checkpoint: %s\n' "$train_dir" >&2
      return 1
    fi
    env CUDA_VISIBLE_DEVICES="$gpu" \
      LD_PRELOAD="$LD_PRELOAD_PATH" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$PYTHON_BIN" -u -m lwcl_v2.ablations.train \
      --config "$config" \
      --manifest "$manifest" \
      --seed "$SEED" \
      --num-workers "$NUM_WORKERS" \
      --output-dir "$train_dir" \
      "${resume_args[@]}" \
      2>&1 | tee -a "$run_dir/train.log"
  else
    printf '[reuse-train] %s\n' "$train_dir"
  fi

  if [[ ! -f "$train_dir/checkpoints/best.pt" ]]; then
    printf 'completed training has no best checkpoint: %s\n' "$train_dir" >&2
    return 1
  fi
  env CUDA_VISIBLE_DEVICES="$gpu" \
    LD_PRELOAD="$LD_PRELOAD_PATH" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$PYTHON_BIN" -u -m lwcl_v2.ablations.evaluate \
    --config "$train_dir/resolved_config.yaml" \
    --manifest "$manifest" \
    --seed "$SEED" \
    --num-workers "$NUM_WORKERS" \
    --checkpoint "$train_dir/checkpoints/best.pt" \
    --output-dir "$eval_dir" \
    --split test \
    --weights raw \
    2>&1 | tee "$run_dir/evaluate_test.log"
  printf '0\n' > "$run_dir/exit_code"
  printf '[job-end] variant=%s protocol=%s fold=%s gpu=%s time=%s\n' \
    "$variant" "$protocol" "$fold" "$gpu" "$(date --iso-8601=seconds)"
}

run_lane() {
  local lane="$1"
  local gpu="${GPUS[$lane]}"
  local status=0
  local index
  for ((index=lane; index<${#JOBS[@]}; index+=${#GPUS[@]})); do
    IFS='|' read -r variant protocol fold manifest <<< "${JOBS[$index]}"
    if ! run_job "$variant" "$protocol" "$fold" "$manifest" "$gpu"; then
      printf '[job-failed] variant=%s protocol=%s fold=%s gpu=%s\n' \
        "$variant" "$protocol" "$fold" "$gpu" >&2
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
printf '%s\n' "$status" > "$RUNS_ROOT/launcher_exit_code"
exit "$status"
