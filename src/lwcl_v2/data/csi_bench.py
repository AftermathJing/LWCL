from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HAR_PROTOCOLS = (
    "test_id",
    "test_cross_device",
    "test_cross_env",
    "test_cross_user",
    "test_cross_user_env",
    "test_instance",
)


def resolve_csi_bench_path(dataset_root: str | Path, task_dir: str | Path, file_path: str) -> Path:
    root = Path(dataset_root).resolve()
    original = Path(file_path)
    # CSI-Bench metadata stores paths relative to the task metadata
    # directory, using ../../sub_Human_h5/... form. Resolve against
    # <task_dir>/metadata, then try broader fallbacks.
    task = Path(task_dir).resolve()
    metadata_candidate = (task / "metadata" / original).resolve()
    if metadata_candidate.exists():
        return metadata_candidate
    candidates = [
        original if original.is_absolute() else None,
        task / original,
        root / original,
        root / "Multitask" / original,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.resolve().exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve CSI-Bench file {file_path!r} under {root}")


def _orient_frequency_time(values: np.ndarray, num_subcarriers: int | None) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        array = np.abs(array)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected CSI_amps to become 2D after squeeze, got {array.shape}")
    expected = int(num_subcarriers) if num_subcarriers not in (None, 0) else None
    if expected is not None and array.shape[0] == expected:
        return array.T.astype(np.float32, copy=False)
    if expected is not None and array.shape[1] == expected:
        return array.astype(np.float32, copy=False)
    # Released CSI-Bench H5 files are conventionally [F,T,1].
    return (array.T if array.shape[0] <= array.shape[1] else array).astype(np.float32, copy=False)


def resample_frequency(values: np.ndarray, bins: int) -> np.ndarray:
    """Interpolate [T,N,F] to a hardware-width invariant [T,N,bins] tensor."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [T,N,F], got {values.shape}")
    if values.shape[-1] == bins:
        return values
    source = np.linspace(0.0, 1.0, values.shape[-1], dtype=np.float32)
    target = np.linspace(0.0, 1.0, bins, dtype=np.float32)
    output = np.empty((*values.shape[:2], bins), dtype=np.float32)
    for t in range(values.shape[0]):
        for receiver in range(values.shape[1]):
            output[t, receiver] = np.interp(target, source, values[t, receiver])
    return output


def load_csi_bench_amplitude(
    path: str | Path,
    *,
    num_subcarriers: int | None = None,
    num_devices: int = 1,
    amplitude_bins: int = 64,
    frame_interval_ms: float = 10.0,
) -> dict[str, np.ndarray]:
    """Load one CSI-Bench H5 sample as T00-compatible amplitude features.

    The benchmark stores ``CSI_amps`` as a frequency/time plane. For the
    multi-device tasks, device planes are concatenated along frequency and are
    split back into an explicit receiver axis before interpolation.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CSI-Bench amplitude loading requires h5py") from exc
    with h5py.File(path, "r") as handle:
        key = next((name for name in ("CSI_amps", "csi", "CSI") if name in handle), None)
        if key is None:
            raise KeyError(f"No CSI amplitude dataset in {path}; keys={list(handle.keys())}")
        values = _orient_frequency_time(np.asarray(handle[key]), num_subcarriers)
    devices = int(num_devices)
    if devices < 1 or values.shape[1] % devices:
        raise ValueError(f"Frequency width {values.shape[1]} is incompatible with num_devices={devices}")
    values = values.reshape(values.shape[0], devices, values.shape[1] // devices)
    mean = values.mean(axis=(0, 2), keepdims=True)
    std = values.std(axis=(0, 2), keepdims=True)
    values = (values - mean) / np.maximum(std, 1e-6)
    values = resample_frequency(values, int(amplitude_bins))
    length = values.shape[0]
    receiver_mask = np.ones(devices, dtype=bool)
    energy = np.sqrt(np.mean(values * values, axis=(0, 2)))
    variation = np.mean(np.abs(np.diff(values, axis=0)), axis=(0, 2)) if length > 1 else np.ones(devices)
    snr_proxy = np.clip(energy / np.maximum(variation, 1e-6), 0.0, 20.0)
    receiver_quality = np.stack(
        [np.ones(devices), np.zeros(devices), np.zeros(devices), energy, snr_proxy], axis=-1
    ).astype(np.float32)
    return {
        "amplitude": values.astype(np.float32),
        "time_mask": np.ones(length, dtype=bool),
        "frame_times_ms": np.arange(length, dtype=np.float32) * float(frame_interval_ms),
        "receiver_mask": receiver_mask,
        "receiver_quality": receiver_quality,
    }


def _load_json_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of sample IDs in {path}")
    return {str(value) for value in payload}


def build_har_protocol_manifests(
    dataset_root: str | Path,
    output_root: str | Path,
    protocols: Iterable[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Create independent train/validation/test manifests for official HAR protocols."""
    root = Path(dataset_root).resolve()
    task_dir = root / "Multitask" / "HumanActivityRecognition"
    if not task_dir.exists():
        task_dir = root / "HumanActivityRecognition"
    metadata_dir = task_dir / "metadata"
    split_dir = task_dir / "splits"
    with (metadata_dir / "label_mapping.json").open("r", encoding="utf-8") as handle:
        label_mapping = json.load(handle)["label_to_idx"]
    with (metadata_dir / "sample_metadata.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    by_id = {str(row.get("id") or row.get("sample_id")): row for row in source_rows}
    selected = tuple(protocols or HAR_PROTOCOLS)
    summaries: dict[str, dict[str, int]] = {}
    for protocol in selected:
        if protocol not in HAR_PROTOCOLS:
            raise ValueError(f"Unsupported HAR protocol {protocol}; choose from {HAR_PROTOCOLS}")
        train_ids = _load_json_ids(split_dir / "train_id.json")
        validation_ids = _load_json_ids(split_dir / "val_id.json")
        test_ids = _load_json_ids(split_dir / f"{protocol}.json")
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise ValueError(f"Official split overlap detected for {protocol}")
        rows: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for split_name, ids in (("train", train_ids), ("validation", validation_ids), ("test", test_ids)):
            for sample_id in sorted(ids):
                if sample_id not in by_id:
                    raise KeyError(f"Split {protocol} references unknown sample {sample_id}")
                source = by_id[sample_id]
                label_name = str(source["label"])
                user = str(source.get("user") or source.get("subject") or "unknown")
                feature_path = resolve_csi_bench_path(root, task_dir, source["file_path"])
                row = dict(source)
                row.update(
                    {
                        "sample_id": sample_id,
                        "feature_path": str(feature_path),
                        "feature_format": "h5",
                        "data_key": "CSI_amps",
                        "feature_mode": "csi_bench_amplitude",
                        "label": int(label_mapping[label_name]),
                        "label_name": label_name,
                        "split": split_name,
                        "benchmark_split": protocol,
                        "subject": user,
                        "user": user,
                        "source_subject": str(source.get("subject", "")),
                        "device_count": 1,
                    }
                )
                rows.append(row)
                counts[split_name] += 1
        target = Path(output_root).resolve() / protocol
        target.mkdir(parents=True, exist_ok=True)
        manifest = target / "manifest.csv"
        fields: list[str] = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        summaries[protocol] = {"samples": len(rows), **dict(counts)}
    return summaries
