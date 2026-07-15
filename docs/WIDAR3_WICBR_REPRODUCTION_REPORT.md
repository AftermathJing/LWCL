# Wi-CBR on LWCL Widar3: Reproduction and Evaluation Report

## 1. Status

The Wi-CBR method-level reproduction is complete on the remote `LWCL` environment. The run includes raw-CSI image reconstruction, a subject-disjoint experiment, four leave-one-environment-out experiments, independent test evaluation, checkpointing, and generated metric reports.

Run date: 2026-07-16 (Asia/Shanghai).

The detailed metric exports are:

- [`WIDAR3_WICBR_SUBJECT_REPORT.md`](WIDAR3_WICBR_SUBJECT_REPORT.md)
- [`WIDAR3_WICBR_CROSS_DOMAIN_REPORT.md`](WIDAR3_WICBR_CROSS_DOMAIN_REPORT.md)

## 2. Fidelity to the official Wi-CBR repository

| Official component | Reproduced component | Status |
| --- | --- | --- |
| phase and DFS two-branch inputs | `WiCBRDataset` phase/DFS image fields | implemented |
| branch-wise `SpatialGate` | `WiCBRNet.phase_gate` and `dfs_gate` | direct structural port |
| two ImageNet-pretrained ResNet-18 branches | two torchvision ResNet-18 feature extractors | implemented |
| 1024-channel DPFusion with four groups and threshold 0.5 | `DPFusion(1024, group_num=4, gate_threshold=0.5)` | direct structural port |
| proxy contrastive loss using classifier weights | `ProxyContrastiveLoss` | direct mathematical port |
| `CE + 0.1 * contrastive` | `beta_1: 0.1` | matched |
| Adam, learning rate `1e-4`, batch size 10, 30 epochs | formal subject configuration | matched |
| StepLR, step size 3, gamma 0.5 | formal subject configuration | matched |

The reproduction intentionally changes the evaluation contract. The official training script evaluates on its test loader while training and selects the best checkpoint by that result. This project uses a separate validation split for checkpoint selection and evaluates the held-out test split once from `best.pt`. This prevents test-set model selection.

## 3. Known adaptation boundaries

This is a method-level reproduction, not a byte-identical execution of every original artifact.

- The official MATLAB phase/DFS pipeline was reconstructed in Python from raw Widar3 Intel 5300 CSI. It follows CSI-ratio/reference denoising, filtering, complex PCA, STFT/DFS construction, phase rendering, and DFS-derived phase saliency, but it is not a byte-for-byte MATLAB execution of the official STIFMM scripts.
- The selected phase input is `phase_weighted`, whose brightness is modulated by the DFS saliency map. Raw phase and DFS weight images are retained for audit.
- The official repository uses seed 888 in `python/Widar/train.py`; the formal LWCL-adapted runs use seed 42.
- The official README targets PyTorch 1.13.1. The verified remote environment uses a modern CUDA-compatible stack listed below.
- Results are from one deterministic split and one training seed. They are not a multi-seed confidence interval.

## 4. Data reconstruction

The subject manifest contains 11,371 usable six-gesture samples. Image reconstruction completed with zero failures:

| Artifact | Count |
| --- | ---: |
| weighted phase images | 11,371 |
| raw phase images | 11,371 |
| DFS images | 11,371 |
| DFS weight images | 11,371 |
| output manifest rows | 11,371 |
| failures | 0 |

Subject-disjoint split:

| Split | Samples | Subjects |
| --- | ---: | --- |
| train | 8,247 | user1, user3, user5, user6, user8, user10, user11, user13, user14, user15, user16 |
| validation | 1,499 | user9, user17 |
| test | 1,625 | user2, user7, user12 |

The three subject sets have no overlap. All referenced phase and DFS files existed at preflight time.

## 5. Formal configuration

```yaml
data:
  num_labels: 6
  image_size: 224
  phase_field: phase_path
  dfs_field: dfs_path
model:
  pretrained: true
  group_num: 4
  gate_threshold: 0.5
training:
  seed: 42
  precision: fp32
  epochs: 30
  batch_size: 10
  eval_batch_size: 10
  num_workers: 4
  learning_rate: 0.0001
  weight_decay: 0.0
  temperature: 0.1
  beta_1: 0.1
  step_lr:
    step_size: 3
    gamma: 0.5
  eval_every_steps: 100
  save_every_steps: 100
  early_stopping_patience: 10
  early_stopping_min_delta: 0.0005
```

## 6. Remote environment

- Host: `wj@10.69.216.119`
- Conda environment: `LWCL`
- GPU used for formal cross-environment runs: NVIDIA A100-PCIE-40GB
- Python: 3.11.15
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- NumPy: 2.4.4
- SciPy: 1.17.1
- Run-recorded base commit: `099ec8e371182786688720eed88c76f8e91f16fa`

The run metadata records a dirty worktree because the Wi-CBR implementation had not yet been committed when the formal experiments started. The delivered repository commit contains the code and reports used by this run; data, outputs, logs, and checkpoints remain untracked.

Manifest hashes:

| Protocol | SHA-256 |
| --- | --- |
| subject-disjoint | `3a4f0a6404becc66f50fc36157d7cd2c5f234e53f2cb18a20060d8b06e60b839` |
| target 20181130 | `98a74e69989be6e96c038edd7eaba5f6fe766da0deca334262ccf45eaad701d3` |
| target 20181204 | `ff37ec6195a4aaf230ece3f30b251076636bb2bdb09a3c235bf1cbd9a07d1679` |
| target 20181209 | `34b5b6409e0ec43fbe18374fc4a8a54ed780f50108ed64ce5f5a5afbe8edecd6` |
| target 20181211 | `f510ae6767bffc2bd6e1233156a80eaa34f715b1bd0d5eb3ed17f4ca2c79ed48` |

## 7. Subject-disjoint result

The best checkpoint was selected by validation macro-F1 at step 1,400.

| Split | Accuracy | Macro-F1 | Samples |
| --- | ---: | ---: | ---: |
| validation | 62.84% | 62.46% | 1,499 |
| test | 63.57% | 64.68% | 1,625 |

Subject-level test accuracy varies substantially: user12 reaches 85.33%, user7 reaches 47.20%, and user2 reaches 31.20%. This is evidence that the aggregate result does not remove all subject/domain sensitivity.

## 8. Cross-environment result

Each fold holds out one environment from both training and validation. The target environment is only used for final test evaluation.

| Target environment | Best step | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 | Test samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20181130 | 2,400 | 96.13% | 96.05% | 70.64% | 70.57% | 6,750 |
| 20181204 | 5,700 | 97.37% | 97.37% | 87.20% | 87.21% | 750 |
| 20181209 | 4,000 | 96.20% | 96.20% | 92.22% | 93.06% | 874 |
| 20181211 | 3,200 | 96.75% | 96.73% | 78.31% | 78.26% | 2,997 |

Aggregate:

| Aggregation | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| equal-fold mean | 82.09% | 82.28% |
| pooled by test samples | 75.41% | 75.42% |

The difference between equal-fold and pooled metrics is caused by unequal target-domain sizes. The 20181130 fold contains 6,750 of the 11,371 test assignments and is also the hardest target, so the pooled result is lower and should be reported alongside the equal-fold mean.

## 9. Exact commands

Run from `/home/wj/LWCL` with `LD_PRELOAD=/home/wj/miniconda3/envs/LWCL/lib/libstdc++.so.6` when required by the remote Conda runtime.

```bash
conda run --no-capture-output -n LWCL python -m lwcl.cli.prepare_wicbr \
  --input-manifest data/splits/widar3_subject_split.csv \
  --output-dir data/processed/widar3_wicbr \
  --output-manifest data/splits/widar3_wicbr_subject_split.csv \
  --phase-source weighted \
  --workers 8 \
  --resume

bash scripts/remote_wicbr_subject.sh
bash scripts/remote_wicbr_cross_environment.sh

conda run --no-capture-output -n LWCL python scripts/generate_wicbr_reports.py \
  --subject-output outputs/widar3_wicbr_subject \
  --subject-test-output outputs/widar3_wicbr_subject_test \
  --subject-manifest data/splits/widar3_wicbr_subject_split.csv \
  --subject-report docs/WIDAR3_WICBR_SUBJECT_REPORT.md \
  --cross-root outputs/widar3_wicbr_cross_environment \
  --cross-report docs/WIDAR3_WICBR_CROSS_DOMAIN_REPORT.md
```

## 10. Artifact contract

The following runtime artifacts are intentionally excluded from Git:

- `data/processed/widar3_wicbr/`
- `data/splits/`
- `outputs/widar3_wicbr_subject/`
- `outputs/widar3_wicbr_subject_test/`
- `outputs/widar3_wicbr_cross_environment/`
- `logs/`
- all `*.pt` checkpoints

The repository tracks the implementation, configuration, launch scripts, tests, and Markdown reports. The remote outputs retain `best.pt`, `last.pt`, resolved configuration, run metadata, JSONL training metrics, and final test metrics for every formal run.
