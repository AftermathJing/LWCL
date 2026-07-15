#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-/home/wj/miniconda3/bin/conda}"
ENV_NAME="${LWCL_V2_ENV:-LWCL-v2}"
MANIFEST="${MANIFEST_PATH:-$ROOT/data/splits/widar3_v2_subject_split.csv}"
OUTPUT="${OUTPUT_DIR:-$ROOT/outputs/audits/widar3_subject_domains}"
FOCUS_SUBJECT="${FOCUS_SUBJECT:-user17}"

cd "$ROOT"
"$CONDA" run --no-capture-output -n "$ENV_NAME" \
  python -m lwcl_v2.cli.audit_subject_domains \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT" \
  --focus-subject "$FOCUS_SUBJECT" \
  --strict-features
