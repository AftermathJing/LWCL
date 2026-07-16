from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from lwcl_v2.config import load_config


BASE_KEYS = (
    "rssi",
    "doppler",
    "differential_csi",
    "time_mask",
    "frame_times_ms",
    "receiver_mask",
    "receiver_quality",
)


def _resolve_feature(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def _update_array_hash(digest: Any, key: str, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    digest.update(key.encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))


def build_contract_audit(
    manifest_path: str | Path,
    config_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    config = load_config(config_path)
    data = config["data"]
    feature_mode = str(data.get("feature_mode", "current"))
    requires_phase = feature_mode in {"phase_dfs", "current_plus_phase"}
    expected_receivers = int(data.get("num_receivers", 6))
    expected_doppler = int(data.get("doppler_bins", 25))
    expected_differential = int(data.get("differential_components", 10))
    expected_phase = int(data.get("phase_subcarriers", 30))

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    unique: dict[str, dict[str, str]] = {}
    duplicate_rows = 0
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in unique:
            duplicate_rows += 1
            continue
        unique[sample_id] = row
    selected = [unique[key] for key in sorted(unique)]
    if limit is not None:
        selected = selected[:limit]

    digest = hashlib.sha256()
    errors: list[dict[str, str]] = []
    nonfinite: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    valid_receiver_counts: Counter[str] = Counter()
    phase_pairs: Counter[str] = Counter()
    lengths: list[int] = []
    shape_examples: dict[str, list[int]] = {}
    processed = 0

    for row in selected:
        sample_id = row["sample_id"]
        labels[str(row["label"])] += 1
        path = _resolve_feature(manifest, row["feature_path"])
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = list(BASE_KEYS)
                if requires_phase:
                    required.extend(("csi_ratio_phase", "csi_ratio_pair"))
                missing = [key for key in required if key not in archive.files]
                if missing:
                    raise ValueError(f"missing keys: {missing}")
                arrays = {key: archive[key] for key in required}
                t = int(arrays["rssi"].shape[0])
                expected_shapes = {
                    "rssi": (t, expected_receivers, 4),
                    "doppler": (t, expected_receivers, expected_doppler),
                    "differential_csi": (
                        t,
                        expected_receivers,
                        expected_differential,
                        2,
                    ),
                    "time_mask": (t,),
                    "frame_times_ms": (t,),
                    "receiver_mask": (expected_receivers,),
                }
                if requires_phase:
                    expected_shapes.update(
                        {
                            "csi_ratio_phase": (
                                t,
                                expected_receivers,
                                expected_phase,
                                2,
                            ),
                            "csi_ratio_pair": (expected_receivers, 2),
                        }
                    )
                for key, expected in expected_shapes.items():
                    if arrays[key].shape != expected:
                        raise ValueError(
                            f"{key} shape {arrays[key].shape} != expected {expected}"
                        )
                if arrays["receiver_quality"].shape[0] != expected_receivers:
                    raise ValueError(
                        f"receiver_quality shape {arrays['receiver_quality'].shape}"
                    )
                for key, value in arrays.items():
                    shape_examples.setdefault(key, list(value.shape))
                    if np.issubdtype(value.dtype, np.number):
                        count = int(value.size - np.isfinite(value).sum())
                        nonfinite[key] += count
                    _update_array_hash(digest, key, value)
                digest.update(sample_id.encode("utf-8"))
                lengths.append(t)
                valid_receivers = int(arrays["receiver_mask"].astype(bool).sum())
                valid_receiver_counts[str(valid_receivers)] += 1
                if requires_phase:
                    for receiver, pair in enumerate(arrays["csi_ratio_pair"]):
                        if arrays["receiver_mask"][receiver]:
                            phase_pairs[f"{int(pair[0])}->{int(pair[1])}"] += 1
                processed += 1
        except Exception as error:
            errors.append(
                {
                    "sample_id": sample_id,
                    "path": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    length_array = np.asarray(lengths, dtype=np.float64)
    valid_total = sum(int(count) * value for count, value in valid_receiver_counts.items())
    receiver_total = processed * expected_receivers
    report = {
        "manifest": str(manifest),
        "config": str(Path(config_path).resolve()),
        "feature_mode": feature_mode,
        "manifest_rows": len(rows),
        "unique_manifest_samples": len(unique),
        "duplicate_manifest_rows": duplicate_rows,
        "audited_samples": processed,
        "failed_samples": len(errors),
        "errors": errors[:100],
        "class_distribution": dict(sorted(labels.items())),
        "shape_examples": shape_examples,
        "time_steps": {
            "min": int(length_array.min()) if length_array.size else None,
            "median": float(np.median(length_array)) if length_array.size else None,
            "p95": float(np.percentile(length_array, 95)) if length_array.size else None,
            "max": int(length_array.max()) if length_array.size else None,
        },
        "nonfinite": dict(sorted(nonfinite.items())),
        "valid_receivers": {
            "counts": dict(sorted(valid_receiver_counts.items())),
            "valid_rate": float(valid_total / receiver_total) if receiver_total else None,
        },
        "phase_pairs": dict(sorted(phase_pairs.items())),
        "contract_sha256": digest.hexdigest(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Signal-v2 current/phase/high-resolution feature contracts"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = build_contract_audit(args.manifest, args.config, limit=args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if args.strict and (
        report["failed_samples"]
        or any(int(value) for value in report["nonfinite"].values())
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
