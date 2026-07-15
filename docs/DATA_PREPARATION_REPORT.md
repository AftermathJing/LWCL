# Real-data preparation report

Date: 2026-07-15

Remote host: `wj@10.69.216.119`

Environment: `LWCL`

## Widar3

Source: `/home/wj/LWCL/data/raw/widar3_raw`

The scanner found 11,375 complete six-receiver action groups for gesture IDs 1-6. The full archive also contains gesture IDs 7-9, which are excluded from the thesis mainline manifest.

The thesis preprocessing pipeline was run with 24 workers and produced:

- 11,371 model-ready NPZ samples;
- canonical shape `[T,6,49]`, with variable `T` preserved;
- zero missing paths in the processed and split manifests;
- four excluded action groups whose raw receiver file contains no CSI records.

Excluded raw groups:

| Sample ID | Empty receiver |
| --- | --- |
| `20181209_user6_g3_p1_o1_i5` | `r5` |
| `20181211_user8_g1_p1_o1_i1` | `r5` |
| `20181211_user8_g3_p3_o3_i5` | `r2` |
| `20181211_user9_g1_p1_o1_i1` | `r1` |

Two additional low-rank receiver groups were recovered by zero-padding missing differential-CSI PCA components. No CSI values are fabricated when a receiver has no records.

Subject-disjoint split:

| Split | Samples | Subjects |
| --- | ---: | --- |
| train | 8,247 | user1, user3, user5, user6, user8, user10, user11, user13, user14, user15, user16 |
| validation | 1,499 | user9, user17 |
| test | 1,625 | user2, user7, user12 |

A real-data tiny-model run completed four training steps successfully with this split.

## CSI-Bench

Source: `/home/wj/LWCL/data/raw/CSI-Bench`

The data adapter follows the official H5 metadata/split organization but restores the physical device axis when multiple devices are concatenated along the frequency dimension.

| Task | Samples | Train | Validation | Test | Model-facing shape |
| --- | ---: | ---: | ---: | ---: | --- |
| BreathingDetection | 100,000 | 70,000 | 15,000 | 15,000 | `[500,1,416]` |
| FallDetection | 6,700 | 4,690 | 1,005 | 1,005 | `[500,1,232]` |
| HumanActivityRecognition | 55,684 | 21,473 | 4,601 | 29,610 | `[500,1,232]` |
| HumanIdentification | 55,684 | 21,473 | 4,601 | 29,610 | `[500,1,232]` |
| Localization | 10,732 | 7,512 | 1,610 | 1,610 | `[500,3,832]` |
| MotionSourceRecognition | 60,858 | 42,599 | 9,128 | 9,131 | `[500,1,232]` |
| ProximityRecognition | 55,684 | 21,473 | 4,601 | 29,610 | `[500,1,232]` |

All 345,342 task-view rows are represented by manifests. The three multitask manifests intentionally reference the same 55,684 physical H5 recordings with different label views. Every generated `feature_path` was checked and exists.

Real-data tiny-model smoke runs completed for:

- FallDetection single-device input `[500,1,232]`;
- Localization three-device input `[500,3,832]`.

## Validation boundary

The real-data preparation layer is ready for model training. This does not yet reproduce the thesis accuracy: paper-scale Qwen/LoRA training, controlled evaluation and baseline comparison remain separate experiment stages.
