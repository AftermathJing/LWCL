#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT/outputs/widar3_fstc_encoder_ablations_20260725}"
SEEDS_TEXT="${SEEDS:-2025 2026 2027 2028 2029}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LD_PRELOAD_PATH="${LD_PRELOAD_PATH:-/home/wj/miniconda3/envs/LWCL/lib/libstdc++.so.6}"
VARIANTS_TEXT="${VARIANTS:-full_control no_feature_encoder no_spatial_encoder no_temporal_encoder}"
GPUS_TEXT="${GPUS:-0 1 2 3}"

cd "$ROOT"
mkdir -p "$RUNS_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

read -r -a VARIANTS <<< "$VARIANTS_TEXT"
read -r -a GPUS <<< "$GPUS_TEXT"
if [[ "${#GPUS[@]}" -lt "${#VARIANTS[@]}" ]]; then
  printf 'need at least one GPU entry per concurrent variant lane\n' >&2
  exit 1
fi

run_variant() {
  local variant="$1"
  local gpu="$2"
  local config="configs/ablations/fstc_encoder/${variant}.yaml"
  local variant_root="$RUNS_ROOT/$variant"
  local seed
  mkdir -p "$variant_root"
  read -r -a seed_list <<< "$SEEDS_TEXT"
  for seed in "${seed_list[@]}"; do
    local run_dir="$variant_root/seed_$seed"
    local train_dir="$run_dir/train"
    local eval_dir="$run_dir/eval"
    mkdir -p "$train_dir" "$eval_dir"
    if [[ ! -f "$train_dir/checkpoints/best.pt" ]]; then
      local resume_args=()
      if [[ -f "$train_dir/checkpoints/last.pt" ]]; then
        resume_args=(--resume-from "$train_dir/checkpoints/last.pt")
      elif find "$train_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        printf 'refusing non-empty directory without checkpoint: %s\n' "$train_dir" >&2
        return 1
      fi
      env CUDA_VISIBLE_DEVICES="$gpu" \
        LD_PRELOAD="$LD_PRELOAD_PATH" \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        "$CONDA" run --no-capture-output -n "$ENV_NAME" \
        python -m lwcl_v2.ablations.train \
        --config "$config" \
        --seed "$seed" \
        --num-workers "$NUM_WORKERS" \
        --output-dir "$train_dir" \
        "${resume_args[@]}" \
        2>&1 | tee -a "$run_dir/train.log"
    fi
    for split in validation test; do
      env CUDA_VISIBLE_DEVICES="$gpu" \
        LD_PRELOAD="$LD_PRELOAD_PATH" \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        "$CONDA" run --no-capture-output -n "$ENV_NAME" \
        python -m lwcl_v2.ablations.evaluate \
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
  done
  "$CONDA" run --no-capture-output -n "$ENV_NAME" \
    python -m lwcl_v2.cli.summarize_multiseed \
    --runs-root "$variant_root" \
    --output "$variant_root/multiseed_summary.json"
}

pids=()
for index in "${!VARIANTS[@]}"; do
  run_variant "${VARIANTS[$index]}" "${GPUS[$index]}" \
    > "$RUNS_ROOT/${VARIANTS[$index]}_lane.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
