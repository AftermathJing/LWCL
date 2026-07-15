# LWCL-v2 validation report

Date: 2026-07-15

## Scope

This report covers engineering validation only. It does not claim gesture-recognition performance or authorize a full B2-B9 training run.

## Remote environment

- Host: `wj@10.69.216.119`
- Conda environment: `LWCL-v2`
- Validation GPU: device 6
- Staging path: `/home/wj/LWCL-v2.0-staging`
- Formal deployment remains pending a dedicated GitHub repository and remote `git clone`.

## Model and activation policy

The base model has 2,009,186 trainable parameters. The default activation policy is:

- SiLU in RSSI, Doppler, differential-CSI, receiver and temporal-convolution front ends;
- GELU with tanh approximation in local/global Transformer FFNs and the classifier;
- softmax for competitive weighting and sigmoid for independent reliability gates;
- no activation after the final six-class logit projection;
- parameter-matched SwiGLU is an optional activation-C ablation, not the base default.

## CUDA and training smoke

Remote tests: `10 passed` after the real-data differential-CSI regression was added.

The CUDA bf16 regression covers the mask value used by attention and pooling. All masked logits use `-1e4`, avoiding the former bf16 overflow caused by `torch.finfo(torch.bfloat16).min`.

The complete smoke test passed all mandatory branches:

1. four optimizer steps with finite CE, cross-subject SupCon loss and gradients;
2. evaluation and checkpoint save on steps 2 and 4;
3. nontrivial `best.pt`, `last.pt` and numbered checkpoints (about 31 MiB each);
4. resume from global step 4 and continue through step 6;
5. independent subject-disjoint test evaluation;
6. intentional exception at step 2;
7. creation of `emergency_step_00000002.pt`;
8. independent validation evaluation loaded from the emergency checkpoint.

The synthetic smoke metrics are intentionally not treated as model performance. Six training steps are only sufficient to exercise the runtime paths.

## Real Widar3 P1 pilot

The pilot used 97 deterministically spaced rows from the 11,375-row raw manifest. It covered all six gesture labels, 16 subjects and four collection dates.

The first run failed for all 97 samples because differential CSI multiplied a flattened `[T, streams*30]` tensor by a `[T,30]` reference. The fix restores `[T,streams,30]`, broadcasts the reference only across streams, removes the reference stream, and then flattens the remaining streams. A shape and finiteness regression test now protects this path.

Post-fix P1 result:

| Check | Result |
| --- | ---: |
| Valid samples | 97/97 |
| Missing feature files | 0 |
| Data-contract errors | 0 |
| Time steps, min / median / p95 / max | 17 / 31 / 40.2 / 49 |
| Frame delta, min / median / max | 50 / 50 / 50 ms |
| Samples with 6 valid receivers | 97 |
| Samples with 5 valid receivers | 0 |

Valid-receiver quality summaries:

| Field | p05 | median | p95 |
| --- | ---: | ---: | ---: |
| Length ratio | 0.9913 | 1.0000 | 1.0080 |
| Packet retention | 0.9751 | 0.9939 | 0.9994 |
| Timestamp health | 0.9958 | 0.9989 | 1.0000 |
| Motion energy | 0.0358 | 0.0538 | 0.0795 |
| SNR proxy | 0.9796 | 1.5692 | 2.0143 |

The median of 31 frames confirms the intended P1 temporal density. This pilot does not yet validate the real 5/6-receiver retention path because every sampled group passed with all six receivers. That path is covered synthetically, but a targeted real-data anomaly audit is still required before full preprocessing.

## Current gate

The code, bf16 runtime, checkpoint lifecycle and normal real-data P1 path are validated. Full Widar3 preprocessing and formal B2-B9 training remain gated on:

1. a targeted scan that identifies and processes real groups with one invalid receiver;
2. construction and audit of the final subject-disjoint split manifest;
3. formal GitHub publication and a clean remote clone.
