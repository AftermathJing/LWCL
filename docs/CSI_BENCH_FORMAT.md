# CSI-Bench data contract

## Authoritative sources

- Paper: https://arxiv.org/abs/2505.21866
- Official code: https://github.com/guozhen-jenn-zhu/CSI-Bench-Real-WiFi-Sensing-Benchmark

The implementation follows the executable official H5 loader when the paper and older comments differ. In particular, released `CSI_amps` arrays are frequency-first (`[F,T,1]`), while some comments describe them as time-first.

## Released layout

Single-task datasets live directly under the dataset root:

- `FallDetection`
- `BreathingDetection`
- `Localization`
- `MotionSourceRecognition`

The shared multitask recordings live under `Multitask/sub_Human_h5`, with three independent metadata/split views:

- `HumanActivityRecognition`
- `HumanIdentification`
- `ProximityRecognition`

Each task contains:

- `metadata/sample_metadata.csv`
- `metadata/label_mapping.json`
- `splits/*.json`
- H5 recordings referenced by the metadata `file_path` column

## Observed H5 shapes

| Task/source | Typical raw `CSI_amps` shape |
| --- | --- |
| FallDetection | `[232,500,1]` |
| MotionSourceRecognition | `[232,500,1]` |
| Localization | `[168,500,1]` (`3 x 56`) or `[2496,500,1]` (`3 x 832`) |
| BreathingDetection | `[208,300,1]` and related variants |
| Multitask smart-device recordings | `[56,500,1]` and other device-dependent widths |

The metadata `num_sub` field is the authoritative frequency-axis hint. CSI-Bench intentionally mixes hardware with different subcarrier counts, so a fixed raw width must not be assumed. Localization recordings contain a device triplet in the metadata and concatenate the three device blocks along the H5 frequency axis.

## LWCL mapping

The official baseline loader uses amplitude data, sample-level normalization, and a single-channel leading crop/zero-pad to `500 x 232`. That is retained as a baseline reference, but it collapses the physical device structure. LWCL instead restores the explicit device axis before task-specific padding:

```text
H5 CSI_amps [sum(F_device),T,1]
  -> frequency/time axis resolution from num_sub
  -> [T,sum(F_device)]
  -> sample z-score
  -> split contiguous device blocks
  -> crop/pad [500,N,F_device]
```

Current resolved task shapes are inferred from metadata. Examples:

- FallDetection and MotionSourceRecognition: `[500,1,232]`
- Localization: `[500,3,832]`, retaining all three devices
- BreathingDetection: `[500,1,416]` for the widest released device
- Multitask tasks: `[500,1,232]`

Generated manifests use:

- `feature_format=h5`
- `data_key=CSI_amps`
- `device_count`, `target_receivers` and `target_features`
- zero-based integer `label` from the official label mapping
- `label_name` with the original semantic label
- canonical `split` values `train`, `validation`, `test`
- `benchmark_splits` containing all official split memberships

Direct H5 loading is deliberate: materializing 352,050 samples as float32 NPZ files would duplicate the dataset and can exceed the raw release size substantially without changing the model input.
