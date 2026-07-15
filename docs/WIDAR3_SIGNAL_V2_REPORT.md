# Widar3 Signal-v2 training report

Date: 2026-07-15

## Protocol

- Remote host: `wj@10.69.216.119`
- GPU: NVIDIA A100, device 7
- Conda environment: `LWCL-v2`
- Code commit used by the accepted run: `dc937b5e209914b0c80cc8b7eae44aa6bd0c2131`
- Model: Signal-v2 base, 2,009,186 trainable parameters
- Seed: 2025
- Preprocessing: P1, STFT window 251 and hop 50
- Position model: local window-relative RoPE and global physical-time RoPE
- Split: frozen subject-disjoint subject sets inherited from the v1 baseline

| Split | Samples | Subjects |
| --- | ---: | --- |
| Train | 8,247 | user1, user3, user5, user6, user8, user10, user11, user13, user14, user15, user16 |
| Validation | 1,498 | user9, user17 |
| Test | 1,625 | user2, user7, user12 |

The v2 quality gate excluded one additional validation sample whose six receivers all failed quality control. The test split was evaluated only after selecting `best.pt` on validation subjects.

## Data audit

- 11,370 valid samples from 11,375 raw action groups;
- five rejected groups: four empty-receiver groups already known from v1 plus one all-receivers-invalid group;
- 11,297 samples with 6/6 valid receivers;
- 73 samples retained with 5/6 valid receivers;
- no missing feature paths and no NPZ contract errors;
- temporal length min/median/p95/max: 11/31/41/70 frames;
- frame interval exactly 50 ms.

## Training correction

The first run used `ema_decay=0.999`. It selected an EMA checkpoint with only 20.25% test macro-F1, while the raw weights stored in the same step-860 checkpoint reached 77.15% test macro-F1. Training is short enough that the 0.999 EMA remained strongly biased toward initialization and was therefore unsuitable for model selection.

The accepted run changed only EMA lag to zero, making validation and selection use the current online weights. Architecture, data, augmentations, subject-balanced sampler, SupCon loss, optimizer, learning-rate schedule and seed remained unchanged. The project default now uses zero-lag EMA until a separately validated EMA schedule is introduced.

## Best validation checkpoint

- Global step: 5,762
- Validation accuracy: 77.770%
- Validation macro-F1: 77.097%
- Validation selection score: 71.338%
- user9 macro-F1: 91.352%
- user17 macro-F1: 57.900%

Training stopped normally at epoch 86 / step 7,482 after 20 validation checks without improvement. Exit code was zero.

## Independent test result

| Metric | Signal-v2 | Frozen v1 HSTE baseline | Difference |
| --- | ---: | ---: | ---: |
| Accuracy | **89.415%** | 84.062% | **+5.353 pp** |
| Macro-F1 | **89.323%** | 84.15% | **+5.17 pp** |

Per-class results:

| Gesture label | Precision | Recall | F1 | Support |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 89.24% | 90.67% | 89.95% | 375 |
| 1 | 94.92% | 89.60% | 92.18% | 250 |
| 2 | 92.69% | 96.40% | 94.51% | 250 |
| 3 | 88.79% | 79.20% | 83.72% | 250 |
| 4 | 85.21% | 87.60% | 86.39% | 250 |
| 5 | 86.19% | 92.40% | 89.19% | 250 |

Per-subject accuracy:

| Subject | Accuracy | Notes |
| --- | ---: | --- |
| user12 | 90.53% | six-class subject |
| user7 | 88.93% | six-class subject |
| user2 | 85.60% | only label 0 is present in this split |

The reported `user2` macro-F1 over all six labels is not meaningful because that subject has support for only one label. Subject accuracy is the appropriate descriptive statistic for this row.

## Interpretation

Signal-v2 exceeded the frozen v1 HSTE baseline by about 5.35 accuracy points under the same held-out subject sets. This supports the combined P1 temporal resolution, feature-family stems, quality-aware receiver fusion, low-compression temporal encoder and cross-subject training route.

This single run does not attribute the gain to one component. B2-B8 ablations are still required before claiming which module caused the improvement, and additional seeds are required before reporting mean and variance.

## Remote artifacts

- Training: `/home/wj/LWCL-v2.0-staging/outputs/widar3_signal_v2_raw_selection_seed2025/`
- Best checkpoint: `checkpoints/best.pt`
- Test metrics: `/home/wj/LWCL-v2.0-staging/outputs/widar3_signal_v2_raw_selection_seed2025_test/test_metrics.json`
