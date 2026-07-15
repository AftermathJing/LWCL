# LWCL Signal Encoder v2

LWCL-v2.0 is a quality-aware multi-receiver hierarchical spatio-temporal framework for cross-subject wireless gesture recognition. It intentionally contains no LLM, Adapter, LoRA, tokenization or language-space alignment.

```text
multi-receiver CSI/RSSI
  -> receiver quality control and timestamp alignment
  -> RSSI / Doppler / differential-CSI feature-family stems
  -> shared receiver encoder
  -> quality-aware receiver fusion
  -> temporal stem with first differences
  -> local-global temporal encoder
  -> attentive statistics pooling
  -> gesture classifier
```

The primary configuration is `configs/signal_v2_base.yaml`. `configs/experiments/` holds one-change-at-a-time B0-B9 ablations; B9 is an explicitly later multiscale extension.

## Execution policy

Training and preprocessing run only on `wj@10.69.216.119` in the `LWCL-v2` Conda environment. Local Windows is used for source editing and static inspection only.

## Data contract

Each processed NPZ contains:

- `rssi`: `[T,6,4]`
- `doppler`: `[T,6,25]`
- `differential_csi`: `[T,6,10,2]`
- `time_mask`: `[T]`
- `frame_times_ms`: `[T]`, the physical center time of every STFT frame
- `receiver_mask`: `[6]`
- `receiver_quality`: `[6,5]` containing length ratio, packet retention, timestamp health, motion energy and SNR proxy

The manifest remains subject-disjoint. Train-only augmentations and subject-balanced batches never alter validation or test samples.

## Initial commands

```bash
conda env create -f environment.yml
conda run --no-capture-output -n LWCL-v2 pytest -q

conda run --no-capture-output -n LWCL-v2 python -m lwcl_v2.cli.preprocess_widar3 \
  --config configs/preprocessing_p1.yaml \
  --input-root data/raw/widar3_raw \
  --source-manifest data/raw/widar3_manifest.csv \
  --output-root data/processed/widar3_v2

CUDA_VISIBLE_DEVICES=6 bash scripts/remote_smoke.sh
```

Long training is not authorized until the preprocessing audit and save/eval/resume/emergency smoke all pass.

Current engineering validation, including the CUDA bf16 smoke and the 97-sample real Widar3 P1 pilot, is recorded in [docs/VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md).

The first full subject-disjoint Widar3 result is documented in [docs/WIDAR3_SIGNAL_V2_REPORT.md](docs/WIDAR3_SIGNAL_V2_REPORT.md): 89.42% test accuracy and 89.32% macro-F1 for Signal-v2 base.
