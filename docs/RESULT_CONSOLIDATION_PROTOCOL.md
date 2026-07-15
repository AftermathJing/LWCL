# Signal-v2 result consolidation protocol

The project is now in result consolidation rather than architecture expansion. The immediate questions are whether the gain is stable, which components cause it, and why the hardest held-out subjects remain difficult.

## Multi-seed strict cross-subject evaluation

The frozen 2025 run remains the first accepted result. New runs use seeds 2026-2029 with the same data, split, model and optimizer configuration. EMA stays disabled and checkpoint selection uses raw online weights.

Remote launch:

```bash
SEEDS="2026 2027 2028 2029" CUDA_DEVICE=7 \
  bash scripts/run_widar3_v2_multiseed.sh
```

`NUM_WORKERS` defaults to 2 in the launcher because the shared server can be CPU/I/O saturated even when GPU memory is free. This runtime setting does not alter batches, augmentations or model optimization.

Each seed writes separate training and evaluation artifacts:

```text
outputs/widar3_signal_v2_multiseed/
  seed_2026/
    train/
    eval/validation_metrics.json
    eval/test_metrics.json
  multiseed_summary.json
```

The summary contains accuracy and macro-F1 mean/std, worst-subject macro-F1, per-class F1, and per-subject accuracy/macro-F1. Seed 2025 should be copied or linked into the same layout only after its accepted checkpoint and resolved config have been verified.

## Subject/domain audit

Run the audit against the frozen strict subject split:

```bash
bash scripts/audit_widar3_subject_domains.sh
```

The audit writes:

- `subject_domain_audit.json`: complete machine-readable report and focused user17 section;
- `subject_summary.csv`: sequence length, receiver count and receiver-quality summary per subject;
- `subject_x_collection_date.csv`, `subject_x_environment.csv`, `subject_x_position.csv`, `subject_x_orientation.csv`: contingency tables used to inspect subject/session/domain confounding.

The audit loads processed feature files by default so receiver-quality statistics are not inferred from filenames. `--strict-features` makes missing or unreadable feature files fail the run.

## Unified evaluation protocols

Generate new manifests from the full processed P1 manifest:

```bash
bash scripts/build_widar3_protocols.sh
```

The generated protocols are deliberately stored under `data/splits/widar3_v2_protocols/` and do not replace the archived cross-environment snapshot protocol.

| Protocol | Test split | Validation split | Purpose |
| --- | --- | --- | --- |
| ID | one deterministic sample fold | next deterministic sample fold | same-domain reference |
| CL | one held-out position | another held-out position | cross-location |
| CO | one held-out orientation | another held-out orientation | cross-orientation |
| CE | one held-out environment | another held-out environment | cross-environment |
| CS | 2-3 held-out subjects, each subject tested once | two other held-out subjects | strict cross-subject |

Every generated fold is checked for six-label coverage. CL/CO/CE also enforce domain-disjoint train, validation and test partitions. CS enforces subject-disjoint train, validation and test partitions.

## Core ablation order

The first formal ablations remain:

1. STFT hop 126 / 50 / 25 with the same model;
2. HSTE window/stride 5/2, 5/1 and 7/3;
3. full quality-aware receiver fusion, receiver mask only, masked mean;
4. unified 49-dimensional projection versus separate feature-family stems;
5. remove first difference, temporal convolution, or both;
6. absolute sinusoidal plus RoPE, RoPE-only, and no positional encoding.

Proxy contrastive loss, CSI-ratio phase, higher-resolution DFS and multi-scale HSTE remain later experiments. They should not be mixed into the stability or attribution runs.

The one-change-at-a-time configs are stored in `configs/ablations/`. Run one with:

```bash
CONFIG_PATH=configs/ablations/a4_flat49.yaml \
OUTPUT_DIR=outputs/ablations/a4_flat49_seed2025 \
SEED=2025 CUDA_DEVICE=5 \
  bash scripts/run_widar3_v2_experiment.sh
```

The hop-126 and hop-25 configs remain data-gated until their own P0/P2 processed manifests are generated and audited. They must not point at P1 features while merely changing the YAML hop value.

## Frozen-checkpoint subject error analysis

Per-sample predictions and embeddings can be exported without retraining:

```bash
CHECKPOINT_PATH=/path/to/frozen/best.pt \
OUTPUT_DIR=outputs/subject_error_analysis \
FOCUS_SUBJECT=user17 CUDA_DEVICE=5 \
  bash scripts/export_widar3_subject_analysis.sh
```

The output contains train/validation probabilities, embeddings, sequence lengths, valid-receiver counts and mean receiver-quality vectors. The final subject report includes confusion matrix, per-class F1, position/orientation/length/receiver-count groups, ECE, NLL, error confidence and same-gesture cosine distance to each training subject.
