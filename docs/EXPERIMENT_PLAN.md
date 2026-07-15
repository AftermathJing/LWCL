# B0-B9 controlled experiment plan

| ID | Change from previous stage | Config or runner | Admission question |
| --- | --- | --- | --- |
| B0 | Frozen 251/126, HSTE 4/2 baseline | archived v1 report/checkpoint | Preserve 84.06% reference |
| B1 | Only HSTE window 3/1 | `b1_legacy_hste_3_1.yaml` with archived v1 runner | Was second-stage compression the main loss? |
| B2 | P1 251/50, flat 49-D encoder, minimal receiver mean | `b2_time_resolution.yaml` | Does temporal density help by itself? |
| B3 | Median-valid alignment, receiver QC/mask | `b3_receiver_qc.yaml` | Do truncated receivers explain failures? |
| B4 | Three physical feature-family stems | `b4_feature_stems.yaml` | Does heterogeneous encoding help? |
| B5 | Mean/max/std/quality-attention fusion | `b5_receiver_fusion.yaml` | Does relational receiver fusion help? |
| B6 | First difference and 3/5 temporal convolution | `b6_temporal_stem.yaml` | Does explicit local motion help? |
| B7 | Attentive mean/std/max pooling | `b7_statistics_pooling.yaml` | Does variation-aware pooling help? |
| B8 | Subject-balanced sampler, SupCon, safe augmentations | `b8_cross_subject.yaml` | Does the difficult subject improve without average collapse? |
| B9 | 9/4 long branch cross-attending to 5/2 short tokens | `b9_multiscale.yaml` | Does multiscale context add value after input fixes? |

Every stage uses the same subject-disjoint split and checkpoint selection rule unless the row explicitly belongs to the frozen v1 runner. A stage is admitted only after save/eval/resume/emergency smoke and after reporting both macro-F1 and worst-subject macro-F1.

Additional orthogonal ablations:

- position A/B/C: absolute+RoPE, RoPE-only, RoPE-only+normalized phase;
- activation A/B/C: GELU/GELU, SiLU/GELU, SiLU/SwiGLU;
- preprocessing P0/P1/P2: hop 126/50/25 with window fixed at 251.
