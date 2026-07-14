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

## Processed manifest

CSV columns include `sample_id`, `feature_path`, `label`, grouping metadata, and `split`.

Subject-disjoint splitting is the default recommendation. Environment-disjoint splitting is required for a strict cross-environment result. Random sample splitting may be used only to reproduce the original thesis protocol and must be labelled as such.
