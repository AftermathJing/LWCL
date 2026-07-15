from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CANONICAL_SPLITS = {
    "train_id": "train",
    "val_id": "validation",
    "test_id": "test",
}

TASK_DEVICE_COUNTS = {
    "Localization": 3,
}

TASK_DEFAULT_FEATURES = {
    "FallDetection": 232,
    "MotionSourceRecognition": 232,
}


def discover_csi_bench_tasks(dataset_root: str | Path) -> dict[str, Path]:
    root = Path(dataset_root).resolve()
    tasks: dict[str, Path] = {}
    for candidate in root.iterdir():
        if candidate.is_dir() and (candidate / "metadata" / "sample_metadata.csv").exists():
            tasks[candidate.name] = candidate
    multitask_root = root / "Multitask"
    if multitask_root.is_dir():
        for candidate in multitask_root.iterdir():
            if candidate.is_dir() and (candidate / "metadata" / "sample_metadata.csv").exists():
                tasks[candidate.name] = candidate
    return dict(sorted(tasks.items()))


def resolve_csi_bench_path(dataset_root: Path, task_dir: Path, file_path: str) -> Path:
    original = Path(file_path)
    candidates = [
        original if original.is_absolute() else None,
        task_dir / original,
        task_dir / "metadata" / original,
        dataset_root / original,
        dataset_root / "tasks" / task_dir.name / original,
    ]
    for candidate in candidates:
        if candidate is not None:
            resolved = candidate.resolve()
            if resolved.exists():
                return resolved
    raise FileNotFoundError(f"Cannot resolve CSI-Bench file {file_path!r} for task {task_dir.name}")


def standardize_csi_bench_array(
    values: np.ndarray,
    num_subcarriers: int | None = None,
    num_devices: int = 1,
    target_time: int = 500,
    target_receivers: int = 1,
    target_features: int = 232,
) -> np.ndarray:
    """Return device-aware LWCL layout ``[T,N,F]``.

    Released H5 files store ``CSI_amps`` as ``[F,T,1]`` even though older
    comments describe ``[T,F,1]``. Metadata ``num_sub`` is therefore used as
    the authoritative frequency-axis hint. Multi-device localization samples
    concatenate three devices along the frequency axis, so the concatenation
    is split back into an explicit receiver/device axis before padding.
    """

    array = np.asarray(values)
    if np.iscomplexobj(array):
        array = np.abs(array)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D CSI plane after squeezing, got {array.shape}")

    expected = int(num_subcarriers) if num_subcarriers not in (None, 0) else None
    if expected is not None and array.shape[0] == expected:
        time_features = array.T
    elif expected is not None and array.shape[1] == expected:
        time_features = array
    elif array.shape[0] <= target_features and array.shape[1] > array.shape[0]:
        time_features = array.T
    elif array.shape[1] <= target_features and array.shape[0] > array.shape[1]:
        time_features = array
    else:
        # The released benchmark's canonical on-disk layout is [F,T,1].
        time_features = array.T

    time_features = np.asarray(time_features, dtype=np.float32)
    mean = float(time_features.mean())
    std = float(time_features.std(ddof=1)) if time_features.size > 1 else 0.0
    normalized = (time_features - mean) / max(std, 1e-8)

    num_devices = int(num_devices)
    if num_devices < 1:
        raise ValueError(f"num_devices must be positive, got {num_devices}")
    if normalized.shape[1] % num_devices:
        raise ValueError(
            f"Frequency width {normalized.shape[1]} is not divisible by num_devices={num_devices}"
        )
    per_device_features = normalized.shape[1] // num_devices
    normalized = normalized.reshape(normalized.shape[0], num_devices, per_device_features)
    if target_receivers < num_devices:
        raise ValueError(f"target_receivers={target_receivers} cannot hold {num_devices} devices")

    output = np.zeros((target_time, target_receivers, target_features), dtype=np.float32)
    copy_time = min(target_time, normalized.shape[0])
    copy_receivers = min(target_receivers, normalized.shape[1])
    copy_features = min(target_features, normalized.shape[2])
    output[:copy_time, :copy_receivers, :copy_features] = normalized[
        :copy_time, :copy_receivers, :copy_features
    ]
    return output


def load_csi_bench_h5(
    path: str | Path,
    data_key: str = "CSI_amps",
    num_subcarriers: int | None = None,
    num_devices: int = 1,
    target_time: int = 500,
    target_receivers: int = 1,
    target_features: int = 232,
) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise ImportError("CSI-Bench loading requires h5py; reinstall the LWCL environment") from exc

    with h5py.File(path, "r") as handle:
        key = next((candidate for candidate in (data_key, "CSI_amps", "csi", "CSI") if candidate in handle), None)
        if key is None:
            raise KeyError(f"No CSI dataset found in {path}; available keys={list(handle.keys())}")
        values = np.asarray(handle[key])
    return standardize_csi_bench_array(
        values,
        num_subcarriers=num_subcarriers,
        num_devices=num_devices,
        target_time=target_time,
        target_receivers=target_receivers,
        target_features=target_features,
    )


def _load_split_memberships(task_dir: Path) -> dict[str, set[str]]:
    memberships: dict[str, set[str]] = {}
    for split_path in sorted((task_dir / "splits").glob("*.json")):
        with split_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            memberships[split_path.stem] = {str(value) for value in payload}
    return memberships


def build_csi_bench_manifests(
    dataset_root: str | Path,
    output_root: str | Path,
    tasks: Iterable[str] | None = None,
    target_time: int = 500,
    target_receivers: int | None = None,
    target_features: int | None = None,
) -> dict[str, dict[str, int]]:
    root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    discovered = discover_csi_bench_tasks(root)
    selected = list(discovered) if tasks is None else list(tasks)
    unknown = sorted(set(selected) - set(discovered))
    if unknown:
        raise ValueError(f"Unknown CSI-Bench tasks: {unknown}; available={list(discovered)}")

    summaries: dict[str, dict[str, int]] = {}
    for task_name in selected:
        task_dir = discovered[task_name]
        metadata_path = task_dir / "metadata" / "sample_metadata.csv"
        mapping_path = task_dir / "metadata" / "label_mapping.json"
        with mapping_path.open("r", encoding="utf-8") as handle:
            label_mapping = json.load(handle)["label_to_idx"]
        memberships = _load_split_memberships(task_dir)
        task_output = output_root / task_name
        manifest_path = task_output / "manifest.csv"
        rows: list[dict[str, Any]] = []
        split_counts: Counter[str] = Counter()

        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        device_count = TASK_DEVICE_COUNTS.get(task_name, 1)
        observed_per_device_features = [
            int(float(row["num_sub"])) // device_count
            for row in source_rows
            if row.get("num_sub") and int(float(row["num_sub"])) % device_count == 0
        ]
        resolved_target_receivers = target_receivers or device_count
        resolved_target_features = target_features or max(
            observed_per_device_features or [TASK_DEFAULT_FEATURES.get(task_name, 232)]
        )

        for source_row in source_rows:
            sample_id = str(source_row.get("id") or source_row.get("sample_id"))
            source_label = str(source_row["label"])
            if source_label not in label_mapping:
                raise KeyError(f"Label {source_label!r} is absent from {mapping_path}")
            source_path = resolve_csi_bench_path(root, task_dir, source_row["file_path"])
            matching_splits = [name for name, ids in memberships.items() if sample_id in ids]
            canonical_split = ""
            for official_name, resolved_name in CANONICAL_SPLITS.items():
                if official_name in matching_splits:
                    canonical_split = resolved_name
                    break
            if not canonical_split and any(name.startswith("test_") for name in matching_splits):
                canonical_split = "test"
            split_counts[canonical_split or "unassigned"] += 1

            normalized_row = dict(source_row)
            if "Difficulty" in normalized_row and "difficulty" not in normalized_row:
                normalized_row["difficulty"] = normalized_row.pop("Difficulty")
            normalized_row.pop("id", None)
            normalized_row.pop("file_path", None)
            normalized_row.pop("label", None)
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "feature_path": Path(os.path.relpath(source_path, manifest_path.parent)).as_posix(),
                "feature_format": "h5",
                "data_key": "CSI_amps",
                "label": int(label_mapping[source_label]),
                "label_name": source_label,
                "task": task_name,
                "split": canonical_split,
                "benchmark_splits": ";".join(sorted(matching_splits)),
                "device_count": device_count,
                "target_time": target_time,
                "target_receivers": resolved_target_receivers,
                "target_features": resolved_target_features,
                **normalized_row,
            }
            rows.append(row)

        if not rows:
            raise RuntimeError(f"No metadata rows found for CSI-Bench task {task_name}")
        task_output.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        summaries[task_name] = {
            "samples": len(rows),
            "num_receivers": resolved_target_receivers,
            "input_features": resolved_target_features,
            **dict(split_counts),
        }
    return summaries
