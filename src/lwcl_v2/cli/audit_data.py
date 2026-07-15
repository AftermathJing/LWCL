from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from lwcl_v2.data.quality import QUALITY_FIELDS


REQUIRED_FIELDS = (
    "rssi",
    "doppler",
    "differential_csi",
    "time_mask",
    "frame_times_ms",
    "receiver_mask",
    "receiver_quality",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LWCL-v2 feature lengths and receiver quality")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--check-all-paths", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lengths = []
    valid_receivers = []
    missing = []
    splits = Counter()
    subjects = {}
    frame_deltas = []
    quality_values = []
    contract_errors = []
    for row in rows:
        split = row.get("split") or "unspecified"
        splits[split] += 1
        subjects.setdefault(split, set()).add(row["subject"])
        path = Path(row["feature_path"])
        if not path.is_absolute():
            path = manifest.parent / path
        if not path.exists():
            missing.append(str(path))
            continue
        if args.check_all_paths or len(lengths) < 1000:
            try:
                with np.load(path, allow_pickle=False) as archive:
                    missing_fields = [field for field in REQUIRED_FIELDS if field not in archive]
                    if missing_fields:
                        raise ValueError(f"missing fields: {missing_fields}")
                    time_mask = archive["time_mask"].astype(bool)
                    receiver_mask = archive["receiver_mask"].astype(bool)
                    time_steps = int(time_mask.sum())
                    receiver_count = int(receiver_mask.sum())
                    expected_shapes = {
                        "rssi": (len(time_mask), len(receiver_mask), 4),
                        "doppler": (len(time_mask), len(receiver_mask), 25),
                        "differential_csi": (len(time_mask), len(receiver_mask), 10, 2),
                        "receiver_quality": (len(receiver_mask), len(QUALITY_FIELDS)),
                    }
                    for field, expected in expected_shapes.items():
                        if archive[field].shape != expected:
                            raise ValueError(f"{field} shape {archive[field].shape}, expected {expected}")
                    for field in REQUIRED_FIELDS:
                        if not np.isfinite(archive[field]).all():
                            raise ValueError(f"{field} contains non-finite values")
                    frame_times = archive["frame_times_ms"][time_mask]
                    if frame_times.size > 1:
                        deltas = np.diff(frame_times)
                        if np.any(deltas <= 0):
                            raise ValueError("frame_times_ms is not strictly increasing")
                        frame_deltas.extend(deltas.tolist())
                    lengths.append(time_steps)
                    valid_receivers.append(receiver_count)
                    quality_values.extend(archive["receiver_quality"][receiver_mask].tolist())
            except Exception as error:
                contract_errors.append({"sample_id": row.get("sample_id", ""), "error": str(error)})
    overlap = {}
    for left in subjects:
        for right in subjects:
            if left < right:
                overlap[f"{left}:{right}"] = sorted(subjects[left] & subjects[right])
    quality_array = np.asarray(quality_values, dtype=np.float64)
    quality_summary = {}
    if quality_array.size:
        for index, field in enumerate(QUALITY_FIELDS):
            quality_summary[field] = {
                "median": float(np.median(quality_array[:, index])),
                "p05": float(np.percentile(quality_array[:, index], 5)),
                "p95": float(np.percentile(quality_array[:, index], 95)),
            }
    report = {
        "rows": len(rows),
        "splits": dict(splits),
        "subjects": {split: sorted(values) for split, values in subjects.items()},
        "subject_overlap": overlap,
        "missing_paths": len(missing),
        "time_steps": {
            "min": int(np.min(lengths)) if lengths else None,
            "median": float(np.median(lengths)) if lengths else None,
            "p95": float(np.percentile(lengths, 95)) if lengths else None,
            "max": int(np.max(lengths)) if lengths else None,
        },
        "valid_receivers": dict(Counter(valid_receivers)),
        "frame_delta_ms": {
            "median": float(np.median(frame_deltas)) if frame_deltas else None,
            "min": float(np.min(frame_deltas)) if frame_deltas else None,
            "max": float(np.max(frame_deltas)) if frame_deltas else None,
        },
        "receiver_quality": quality_summary,
        "contract_errors": {
            "count": len(contract_errors),
            "examples": contract_errors[:10],
        },
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if missing or contract_errors or any(overlap.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
