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

After cloning on the remote server:

```bash
bash scripts/bootstrap_remote.sh
bash scripts/remote_preflight.sh
bash scripts/remote_smoke.sh
```

The smoke route uses a small Transformer so it can prove forward/backward, evaluation, checkpoint save, resume and final test evaluation before loading Qwen. It is not a substitute for the paper experiment.

## Raw-data preprocessing

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
  --num-receivers 6

conda run --no-capture-output -n LWCL python -m lwcl.cli.make_splits \
  --input-manifest data/processed/widar3/manifest.csv \
  --output-manifest data/splits/widar3_subject_split.csv \
  --group-field subject
```

Use `--group-field environment` for strict environment-disjoint evaluation. Random sample splitting is retained only for reproducing the original thesis protocol and should not be used as the primary generalization claim.

## Training

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

This reconstruction is code-complete at the repository level, but paper-scale accuracy is not considered restored until the remote environment, real Widar3.0 manifest, full smoke test and controlled reproduction runs have completed.
