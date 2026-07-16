#!/usr/bin/env bash
set -euo pipefail

export LD_PRELOAD="${LD_PRELOAD:-/home/wj/miniconda3/envs/LWCL/lib/libstdc++.so.6}"
CONDA_BIN="${CONDA_BIN:-/home/wj/miniconda3/bin/conda}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-auto}"
POLL_SECONDS="${POLL_SECONDS:-120}"

if [[ "$GPU_ID" == "auto" ]]; then
  GPU_ID="$($CONDA_BIN run --no-capture-output -n LWCL python scripts/select_free_gpu.py)"
fi

cd "$(dirname "$0")/.."
mkdir -p logs docs

TARGET_MANIFEST="data/splits/widar3_wicbr_subject_split.csv"
PREPROCESS_PATTERN="python -m lwcl.cli.prepare_wicbr --input-manifest data/splits/widar3_subject_split.csv"
LOG_FILE="logs/wicbr_pipeline_after_preprocess.log"

echo "[$(date '+%F %T')] watcher start gpu=$GPU_ID" | tee -a "$LOG_FILE"

while [[ ! -f "$TARGET_MANIFEST" ]]; do
  if pgrep -f "$PREPROCESS_PATTERN" >/dev/null 2>&1; then
    phase_count=$(find data/processed/widar3_wicbr/phase_weighted -type f 2>/dev/null | wc -l || true)
    dfs_count=$(find data/processed/widar3_wicbr/dfs -type f 2>/dev/null | wc -l || true)
    echo "[$(date '+%F %T')] waiting preprocess phase=$phase_count dfs=$dfs_count" | tee -a "$LOG_FILE"
    sleep "$POLL_SECONDS"
  else
    echo "[$(date '+%F %T')] preprocess stopped before manifest existed" | tee -a "$LOG_FILE"
    exit 1
  fi
done

echo "[$(date '+%F %T')] preprocess complete, starting subject run" | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/remote_wicbr_subject.sh 2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%F %T')] subject run complete, starting cross-environment run" | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/remote_wicbr_cross_environment.sh 2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%F %T')] generating reports" | tee -a "$LOG_FILE"
$CONDA_BIN run --no-capture-output -n LWCL python scripts/generate_wicbr_reports.py \
  --subject-output outputs/widar3_wicbr_subject \
  --subject-test-output outputs/widar3_wicbr_subject_test \
  --subject-manifest data/splits/widar3_wicbr_subject_split.csv \
  --subject-report docs/WIDAR3_WICBR_SUBJECT_REPORT.md \
  --cross-root outputs/widar3_wicbr_cross_environment \
  --cross-report docs/WIDAR3_WICBR_CROSS_DOMAIN_REPORT.md 2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%F %T')] pipeline complete" | tee -a "$LOG_FILE"
