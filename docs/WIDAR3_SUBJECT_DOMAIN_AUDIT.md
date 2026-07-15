# Widar3 subject/domain audit

Audit date: 2026-07-15

This report audits the frozen P1 subject-disjoint manifest and all processed feature archives. It addresses whether the difficult-subject result, especially user17, can be explained by collection-domain or signal-quality differences.

## Audit coverage

- Samples: 11,370
- Subjects: 16
- Feature archives read: 11,370
- Feature read/contract errors: 0
- Focus subject: user17
- Machine-readable output: `/home/wj/LWCL-v2.0-staging/outputs/audits/widar3_subject_domains/subject_domain_audit.json`

## Subject and environment are not independently sampled

Every subject belongs to exactly one collection environment/date:

- 20181130: user5, user10-user17
- 20181204: user1
- 20181209: user2, user6
- 20181211: user3, user7-user9

Therefore a cross-subject experiment also contains collection-session/environment structure. This does not invalidate the strict subject-disjoint split, but it means the measured shift cannot be described as pure subject style alone.

User17 is not uniquely tied to an unseen environment: eight other subjects share 20181130. This permits a within-environment comparison and shows that user17 still has several distinctive signal statistics.

## user17 data profile

user17 contributes 749 retained validation samples:

- Position counts: 149/150/150/150/150
- Orientation counts: 150/149/150/150/150
- Gesture counts: 125/125/125/124/125/125
- Valid receivers: 739 samples with 6 receivers, 10 samples with 5 receivers
- Median sequence length: 40 STFT frames
- Sequence length p05/p95: 28/50 frames

The class, position and orientation distributions are almost perfectly balanced. The low user17 result is therefore unlikely to be caused by a simple label or pose-count imbalance.

## user17 versus same-environment peers

Among the nine subjects collected in 20181130:

| Statistic | user17 | Within-environment rank | Interpretation |
| --- | ---: | ---: | --- |
| Median sequence length | 40 frames | highest, 9/9 | actions or retained recordings are substantially longer |
| Median packet retention | 0.9865 | lowest, 1/9 | more packet loss than same-session peers |
| Median timestamp health | 0.9947 | lowest, 1/9 | more irregular timing/gaps |
| Median motion energy | 0.0528 | 7/9 | not an extreme low-energy subject |
| Median SNR proxy | 1.6430 | highest, 9/9 | low performance is not explained by weak signal amplitude |

Across all 16 subjects, user17 also has the longest median sequence, the lowest median timestamp health and the highest median SNR proxy. Receiver availability is not exceptional: its mean valid-receiver count is 5.9866, with only ten 5-receiver samples.

## Current conclusion

The data audit does not support the hypothesis that user17 fails mainly because receivers are missing or because its class/position/orientation distribution is unbalanced. The stronger candidates are:

1. a duration/action-speed shift, visible in the 40-frame median versus 30-32 frames for same-environment peers;
2. timestamp and packet-retention irregularity;
3. a subject-specific representation shift despite strong SNR.

This supports keeping temporal resolution and explicit time information as core method claims. It also motivates error analysis by gesture and sequence length before adding a larger temporal model.

## Unified protocol manifests

The new protocol generator completed successfully on all 11,370 samples:

- ID: 5 folds, each 6,820 train / 2,275 validation / 2,275 test samples;
- CL: 5 leave-one-position-out folds;
- CO: 5 leave-one-orientation-out folds;
- CE: 4 leave-one-environment-out folds;
- CS: 6 subject-disjoint folds, with every subject entering test exactly once.

user17 appears in CS fold 03 together with user1 and user9. All generated folds passed six-label coverage and the protocol-specific leakage checks.

## Remaining analysis requirement

This report is data-side only. The next user17 analysis still needs per-sample model outputs to report:

- per-gesture F1 and confusion matrix;
- performance by position, orientation and sequence-length bin;
- error confidence, ECE and NLL;
- embedding distance to training subjects within the same gesture.

Those outputs should be generated from the frozen accepted checkpoint, without retraining or changing the model.
