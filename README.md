# LWCL

Reconstruction of the thesis project **LLM-Enabled Wireless Correlation Learning Framework**.

The old undergraduate-project code was treated only as a reference. This repository rebuilds the workflow from the thesis architecture and uses explicit contracts for data, model inputs, training, evaluation, checkpointing and remote execution.

## Scope

Implemented pipeline:

```text
Intel 5300 CSI .dat files
  -> CSI/RSSI parsing
  -> reference-antenna differential CSI
  -> 2-60 Hz filtering
  -> Complex PCA + STFT Doppler features
  -> 49-dimensional per-receiver representation
  -> Channel Attention
  -> hierarchical local/global RoPE encoder
  -> cross-modal Adapter
  -> Tiny smoke backbone or Qwen2.5 + LoRA
  -> gesture classification
```

Canonical tensor shape: `[batch, time, receivers, features_per_receiver]`.

The default paper configuration assumes six receivers, 49 features per receiver and six gesture classes.

## Repository layout

```text
configs/                 smoke and paper-scale configurations
docs/                    architecture, data and remote runbooks
scripts/                 remote environment, preflight and smoke scripts
src/lwcl/data/           Intel 5300 parsing, preprocessing, datasets and splits
src/lwcl/models/         Channel Attention, HSTE, Adapter and backbones
src/lwcl/training/       metrics, checkpoints and trainer
src/lwcl/cli/            preprocessing, split, train and evaluation commands
tests/                   forward/backward, data and checkpoint tests
```

## Execution policy

Training and deployment are not performed on the local Windows workspace. The target is:

```text
wj@10.69.216.119
```

The remote Conda environment name is fixed to `LWCL`.

For a concise implementation-oriented summary of the thesis, read [`docs/THESIS_SPEC.md`](docs/THESIS_SPEC.md). It should be the first reference before reopening the full thesis.

After cloning on the remote server:

```bash
bash scripts/bootstrap_remote.sh
bash scripts/remote_preflight.sh
bash scripts/remote_smoke.sh
```

The smoke route uses a small Transformer so it can prove forward/backward, evaluation, checkpoint save, resume and final test evaluation before loading Qwen. It is not a substitute for the paper experiment.

## Raw-data preprocessing

### Widar3

Build the thesis six-gesture action-group manifest directly from the released filenames:

```bash
conda run --no-capture-output -n LWCL python -m lwcl.cli.build_widar3_manifest \
  --dataset-root data/raw/widar3_raw \
  --output-manifest data/raw/widar3_manifest.csv
```

Create a CSV with at least:

```csv
sample_id,raw_path,label,subject,environment,position,orientation
g0_000,/path/to/sample_directory,0,S01,R1,P1,O1
```

Each sample directory must contain receiver files named like `<prefix>-r1.dat`, `<prefix>-r2.dat`, and so on.

Run on the remote server:

```bash
conda run --no-capture-output -n LWCL python -m lwcl.cli.preprocess \
  --input-manifest data/raw/manifest.csv \
  --output-dir data/processed/widar3 \
  --output-manifest data/processed/widar3/manifest.csv \
  --num-receivers 6 \
  --workers 16 \
  --resume

conda run --no-capture-output -n LWCL python -m lwcl.cli.make_splits \
  --input-manifest data/processed/widar3/manifest.csv \
  --output-manifest data/splits/widar3_subject_split.csv \
  --group-field subject
```

The scanner groups `user-gesture-position-orientation-instance-r1.dat` through `r6.dat` as one sample. The default manifest includes gesture IDs 1-6 used by the thesis and excludes the extra gesture IDs 7-9 present in the full copied archive.

### CSI-Bench

CSI-Bench uses HDF5 amplitude tensors and official metadata/split files rather than Intel 5300 `.dat` receiver groups. Build model-ready per-task manifests with:

```bash
conda run --no-capture-output -n LWCL python -m lwcl.cli.prepare_csi_bench \
  --dataset-root data/raw/CSI-Bench \
  --output-dir data/processed/csi_bench \
  --tasks all
```

The dataset loader converts each referenced `CSI_amps` array to a task-specific `[500,N,F]` tensor. For Localization, the three device blocks are restored as `N=3` instead of being collapsed into one channel; single-device tasks use `N=1`. This avoids duplicating the full release. See [`docs/CSI_BENCH_FORMAT.md`](docs/CSI_BENCH_FORMAT.md).

Audit paths and model-facing shapes with:

```bash
conda run --no-capture-output -n LWCL python -m lwcl.cli.audit_manifest \
  --manifest data/processed/csi_bench/Localization/manifest.csv \
  --split train --max-seq-len 500 --check-all-paths
```

Use `--group-field environment` for strict environment-disjoint evaluation. Random sample splitting is retained only for reproducing the original thesis protocol and should not be used as the primary generalization claim.

## Training

Pure signal baseline without an LLM or Adapter:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/remote_hste_classifier_smoke.sh

CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n LWCL python -m lwcl.cli.train \
  --config configs/widar3_hste_classifier.yaml \
  --output-dir outputs/widar3_hste_classifier
```

This route is exactly `Channel Attention -> HSTE -> masked attention pooling -> classification head`.

The completed subject-disjoint run reached **84.06% test accuracy** and **84.15% macro-F1** without an LLM or Adapter. See [`docs/WIDAR3_HSTE_CLASSIFIER_REPORT.md`](docs/WIDAR3_HSTE_CLASSIFIER_REPORT.md) for the full configuration, per-class metrics, confusion matrix and thesis comparison.

Mandatory smoke:

```bash
bash scripts/remote_smoke.sh
```

Paper route, only after the smoke passes:

```bash
conda run --no-capture-output -n LWCL python -m lwcl.cli.train \
  --config configs/paper_qwen.yaml \
  --output-dir outputs/paper_qwen
```

Resume:

```bash
conda run --no-capture-output -n LWCL python -m lwcl.cli.train \
  --config configs/paper_qwen.yaml \
  --output-dir outputs/paper_qwen_resumed \
  --resume-from outputs/paper_qwen/checkpoints/last.pt
```

Evaluation:

```bash
conda run --no-capture-output -n LWCL python -m lwcl.cli.evaluate \
  --config configs/paper_qwen.yaml \
  --checkpoint outputs/paper_qwen/checkpoints/best.pt \
  --output-dir outputs/paper_qwen_test \
  --split test
```

Metrics include accuracy, macro precision/recall/F1, per-class metrics, confusion matrix and subgroup accuracy by subject, environment, position and orientation.

Single-sample remote inference:

```bash
conda run --no-capture-output -n LWCL python -m lwcl.cli.predict \
  --config configs/paper_qwen.yaml \
  --checkpoint outputs/paper_qwen/checkpoints/best.pt \
  --sample data/processed/widar3/features/example.npz
```

## Checkpoint and observability contract

- JSONL metrics are written to `metrics.jsonl`.
- `last.pt`, numbered step checkpoints and `best.pt` are supported.
- Model, optimizer, scheduler, scaler, global step, config and random states are saved.
- Exceptions trigger an emergency-checkpoint attempt.
- Qwen runs use trainable-only model checkpoints by default to avoid copying frozen 7B weights into every checkpoint.

## Git safety

Data, model weights, checkpoints, outputs, `.env`, credentials, keys, `.idea`, `.venv` and cache folders are ignored. Before publishing, run:

```bash
python scripts/pre_push_audit.py
```

No data or secrets belong in this repository.

## Current validation boundary

The remote environment and real-data preparation are now validated. Widar3 provides 11,371 processed six-receiver samples with a subject-disjoint split, and CSI-Bench provides seven official task manifests with device-aware H5 loading. Real-data tiny-model smokes passed for Widar3, FallDetection and three-device Localization. The LLM-free Channel Attention + HSTE baseline has also completed full training and independent test evaluation at 84.06% accuracy / 84.15% macro-F1.

See [`docs/DATA_PREPARATION_REPORT.md`](docs/DATA_PREPARATION_REPORT.md) for exact counts, shapes and the four unusable Widar3 raw groups, and [`docs/WIDAR3_HSTE_CLASSIFIER_REPORT.md`](docs/WIDAR3_HSTE_CLASSIFIER_REPORT.md) for the pure-signal baseline. The full thesis LWCL result is not considered reproduced until the Qwen/LoRA route is trained and compared under the same controlled split.
