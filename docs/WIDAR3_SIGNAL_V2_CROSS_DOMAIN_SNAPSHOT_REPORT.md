# Widar3 Signal-v2 cross-environment snapshot report

Date: 2026-07-15

## Scope

This report summarizes the current leave-one-environment-out snapshot for Signal-v2. It reuses the frozen four-fold protocol from the archived HSTE cross-domain report and evaluates the best checkpoint available at snapshot time for each fold.

It is a **best-so-far snapshot**, not a final sealed report after all folds naturally early-stop.

## Protocol

- Model: Signal-v2 base, 2,009,186 trainable parameters
- Remote host: `wj@10.69.216.119`
- Environment: `LWCL-v2`
- Split source: archived v1 `widar3_cross_environment` manifests
- Feature source: v2 P1 processed manifest with the same sample identities wherever available
- Selection rule: source-domain validation macro-F1 only
- Precision: bf16

The v2 quality gate removed one additional sample, `20181130_user17_g4_p1_o2_i5`, because all six receivers failed quality control. Therefore every fold contains 11,370 rather than 11,371 valid samples.

## Fold-level snapshot results

| Target environment | Test samples | Accuracy | Macro-F1 | Archived HSTE accuracy | Archived HSTE macro-F1 | Accuracy delta | Macro-F1 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20181130 | 6,749 | **85.54%** | **85.48%** | 76.50% | 76.46% | **+9.04 pp** | **+9.02 pp** |
| 20181204 | 750 | **92.27%** | **92.28%** | 85.60% | 85.61% | **+6.67 pp** | **+6.67 pp** |
| 20181209 | 874 | **95.08%** | **95.28%** | 95.31% | 95.35% | **-0.23 pp** | **-0.07 pp** |
| 20181211 | 2,997 | **86.29%** | **86.25%** | 81.85% | 81.84% | **+4.44 pp** | **+4.41 pp** |

Equal-weight environment average:

- Accuracy: **89.79%**
- Macro-F1: **89.82%**

Archived equal-weight baseline:

- Accuracy: 84.82%
- Macro-F1: 84.81%

Equal-weight gain:

- Accuracy: **+4.97 pp**
- Macro-F1: **+5.01 pp**

Merged out-of-domain snapshot across all 11,370 samples:

- Accuracy: **86.91%**
- Macro-F1: **86.93%**

Archived merged baseline:

- Accuracy: 79.96%
- Macro-F1: 79.92%

Merged gain:

- Accuracy: **+6.95 pp**
- Macro-F1: **+7.01 pp**

## Notes on subject metrics

- `20181204` test contains only `user1`, so its subject metric is well-defined but not multi-subject.
- `20181209` test contains `user2` and `user6`; `user2` only contributes one gesture label in this split, so its six-label macro-F1 is not meaningful. Accuracy remains the safer descriptive statistic for that row.
- `20181130` remains the hardest environment overall, and the weakest held-out subject inside this fold is still `user17`.

## Interpretation

Relative to the archived Channel Attention + HSTE route, the current Signal-v2 front end improves three of the four held-out environments and strongly improves the hardest weighted environment, `20181130`. The only fold not clearly improved at snapshot time is `20181209`, where the archived baseline was already exceptionally strong.

This indicates that the combination of:

- P1 temporal resolution,
- feature-family stems,
- quality-aware receiver fusion,
- low-compression local-global temporal encoding,
- and cross-subject training

transfers well to the cross-environment setting rather than only helping the fixed subject-disjoint split.

## Snapshot artifacts

- Fold outputs: `/home/wj/LWCL-v2.0-staging/outputs/widar3_v2_cross_environment/<target>/`
- Snapshot checkpoints: `snapshot_eval/checkpoints/best_snapshot.pt`
- Snapshot test metrics: `snapshot_eval/test/test_metrics.json`
