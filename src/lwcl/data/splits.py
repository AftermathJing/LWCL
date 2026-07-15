from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from pathlib import Path


def create_group_splits(
    input_manifest: str | Path,
    output_manifest: str | Path,
    group_field: str,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 2025,
) -> dict[str, int]:
    input_manifest = Path(input_manifest).resolve()
    output_manifest = Path(output_manifest).resolve()
    with input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Input manifest is empty")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = row["sample_id"] if group_field == "sample" else row[group_field]
        grouped[key].append(row)
    groups = list(grouped)
    random.Random(seed).shuffle(groups)
    group_count = len(groups)
    if group_count < 3:
        raise ValueError(f"At least three groups are required for train/validation/test, found {group_count}")
    train_group_count = max(1, min(group_count - 2, round(group_count * train_ratio)))
    validation_group_count = max(
        1,
        min(group_count - train_group_count - 1, round(group_count * validation_ratio)),
    )
    assignments: dict[str, str] = {}
    counts = {"train": 0, "validation": 0, "test": 0}
    for index, group in enumerate(groups):
        group_size = len(grouped[group])
        if index < train_group_count:
            split = "train"
        elif index < train_group_count + validation_group_count:
            split = "validation"
        else:
            split = "test"
        assignments[group] = split
        counts[split] += group_size
    for row in rows:
        key = row["sample_id"] if group_field == "sample" else row[group_field]
        row["split"] = assignments[key]
        for path_field in ("feature_path", "raw_path"):
            value = row.get(path_field)
            if value and not Path(value).is_absolute():
                absolute_path = (input_manifest.parent / value).resolve()
                row[path_field] = Path(os.path.relpath(absolute_path, output_manifest.parent)).as_posix()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if "split" not in fieldnames:
        fieldnames.append("split")
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return counts
