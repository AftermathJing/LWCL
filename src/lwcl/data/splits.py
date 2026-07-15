from __future__ import annotations

import csv
import hashlib
import json
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


def create_leave_one_domain_out_splits(
    input_manifest: str | Path,
    output_dir: str | Path,
    domain_field: str = "environment",
    validation_ratio: float = 0.15,
    seed: int = 2025,
) -> dict[str, dict[str, object]]:
    """Create one strict held-out-test manifest for every observed domain."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between zero and one")
    input_manifest = Path(input_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    with input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Input manifest is empty")
    if domain_field not in rows[0]:
        raise ValueError(f"Missing domain field {domain_field!r}")
    domains = sorted({row[domain_field] for row in rows})
    if len(domains) < 2:
        raise ValueError("At least two domains are required")

    summary: dict[str, dict[str, object]] = {}
    for target_domain in domains:
        fold_dir = output_dir / target_domain
        output_manifest = fold_dir / "manifest.csv"
        assignments: dict[int, str] = {
            index: "test" for index, row in enumerate(rows) if row[domain_field] == target_domain
        }
        strata: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            if row[domain_field] != target_domain:
                strata[(row[domain_field], row["label"])].append(index)
        for (source_domain, label), indexes in sorted(strata.items()):
            digest = hashlib.sha256(
                f"{seed}:{target_domain}:{source_domain}:{label}".encode("utf-8")
            ).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            rng.shuffle(indexes)
            validation_count = (
                0
                if len(indexes) == 1
                else max(1, min(len(indexes) - 1, round(len(indexes) * validation_ratio)))
            )
            validation_indexes = set(indexes[:validation_count])
            for index in indexes:
                assignments[index] = "validation" if index in validation_indexes else "train"

        output_rows: list[dict[str, str]] = []
        split_counts = {"train": 0, "validation": 0, "test": 0}
        split_domains: dict[str, set[str]] = defaultdict(set)
        split_subjects: dict[str, set[str]] = defaultdict(set)
        split_labels: dict[str, set[str]] = defaultdict(set)
        for index, source_row in enumerate(rows):
            row = dict(source_row)
            split = assignments[index]
            row["split"] = split
            for path_field in ("feature_path", "raw_path"):
                value = row.get(path_field)
                if value and not Path(value).is_absolute():
                    absolute_path = (input_manifest.parent / value).resolve()
                    row[path_field] = Path(os.path.relpath(absolute_path, fold_dir)).as_posix()
            output_rows.append(row)
            split_counts[split] += 1
            split_domains[split].add(row[domain_field])
            if row.get("subject"):
                split_subjects[split].add(row["subject"])
            split_labels[split].add(row["label"])

        if split_domains["test"] != {target_domain}:
            raise RuntimeError(f"Target-domain leakage in fold {target_domain}")
        if target_domain in split_domains["train"] or target_domain in split_domains["validation"]:
            raise RuntimeError(f"Target domain leaked into source splits for fold {target_domain}")
        if not split_counts["train"] or not split_counts["validation"] or not split_counts["test"]:
            raise RuntimeError(f"Fold {target_domain} contains an empty split")

        fold_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = list(output_rows[0])
        if "split" not in fieldnames:
            fieldnames.append("split")
        with output_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        fold_summary: dict[str, object] = {
            "manifest": str(output_manifest),
            "target_domain": target_domain,
            "counts": split_counts,
            "domains": {split: sorted(values) for split, values in split_domains.items()},
            "subjects": {split: sorted(values) for split, values in split_subjects.items()},
            "labels": {split: sorted(values) for split, values in split_labels.items()},
        }
        (fold_dir / "summary.json").write_text(
            json.dumps(fold_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary[target_domain] = fold_summary
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
