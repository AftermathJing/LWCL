# Data format

## Raw manifest

CSV columns:

- `sample_id`: stable unique identifier;
- `raw_path`: directory containing `<sample-prefix>-r1.dat` through receiver N;
- `label`: zero-based class integer;
- `subject`, `environment`, `position`, `orientation`: required grouping metadata.

## Processed sample

Each `.npz` contains:

- `features`: float32 array `[T,N,F]`;
- `metadata`: JSON string when produced by the raw preprocessor.

The paper configuration uses `N=6` and `F=49`:

- four RSSI features;
- 25 aggregated Doppler features;
- 20 real-valued differential-CSI PCA features.

If a valid receiver contains fewer than ten differential-CSI PCA directions, missing components are zero-padded so the feature contract remains fixed. Action groups containing a receiver file with no CSI records are excluded from the processed manifest and recorded in the preprocessing failure CSV; CSI values are never fabricated for an empty receiver file.

## Processed manifest

CSV columns include `sample_id`, `feature_path`, `label`, grouping metadata, and `split`.

Subject-disjoint splitting is the default recommendation. Environment-disjoint splitting is required for a strict cross-environment result. Random sample splitting may be used only to reproduce the original thesis protocol and must be labelled as such.

## CSI-Bench

CSI-Bench is not an Intel 5300 receiver-file dataset. Its released samples are HDF5 files with a `CSI_amps` dataset, normally stored as `[total_subcarriers, time, 1]`. The model adapter converts each sample to device-aware `[time, devices, features_per_device]` by:

1. resolving the frequency axis with the metadata `num_sub` field;
2. using amplitude only;
3. applying sample-level z-score normalization before padding;
4. splitting concatenated multi-device frequency blocks using task metadata;
5. leading crop/zero-padding to a task-specific fixed shape;
6. preserving the official `train_id`, `val_id`, `test_id` and additional benchmark split memberships.

The generated per-task manifests reference the original H5 files directly. This follows the official loader and avoids duplicating the 77 GB release into a much larger NPZ cache. See `docs/CSI_BENCH_FORMAT.md` for task-specific details.
