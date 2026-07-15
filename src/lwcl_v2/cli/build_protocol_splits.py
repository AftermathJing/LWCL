from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROTOCOL_FIELDS = {"cl": "position", "co": "orientation", "ce": "environment"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _label_set(rows: list[dict[str, str]]) -> set[str]:
    return {row["label"] for row in rows}


def _balanced_chunks(values: list[str], maximum_size: int) -> list[list[str]]:
    fold_count = math.ceil(len(values) / maximum_size)
    base, remainder = divmod(len(values), fold_count)
    sizes = [base + (index < remainder) for index in range(fold_count)]
    output = []
    offset = 0
    for size in sizes:
        output.append(values[offset : offset + size])
        offset += size
    return output


def _subject_test_folds(
    rows: list[dict[str, str]], seed: int, maximum_size: int
) -> list[list[str]]:
    subjects = sorted({row["subject"] for row in rows})
    all_labels = _label_set(rows)
    for attempt in range(1000):
        candidate = list(subjects)
        random.Random(seed + attempt).shuffle(candidate)
        folds = _balanced_chunks(candidate, maximum_size)
        if all(
            _label_set([row for row in rows if row["subject"] in fold]) == all_labels
            for fold in folds
        ):
            return folds
    raise ValueError(
        "Could not construct class-complete subject test folds; inspect per-subject label coverage"
    )


def _choose_validation_subjects(
    rows: list[dict[str, str]], excluded: set[str], count: int, seed: int
) -> list[str]:
    candidates = sorted({row["subject"] for row in rows} - excluded)
    all_labels = _label_set(rows)
    combinations = list(itertools.combinations(candidates, count))
    random.Random(seed).shuffle(combinations)
    for subjects in combinations:
        selected = [row for row in rows if row["subject"] in subjects]
        if _label_set(selected) == all_labels:
            return list(subjects)
    raise ValueError(f"No class-complete validation subject set of size {count}")


def _assign_id(rows: list[dict[str, str]], seed: int, fold_count: int) -> list[tuple[str, dict[str, str]]]:
    strata: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    stratum_fields = ("subject", "label", "environment", "position", "orientation")
    for row in rows:
        strata[tuple(row.get(field, "") for field in stratum_fields)].append(row)
    bucket_by_id = {}
    for stratum_rows in strata.values():
        ordered = sorted(
            stratum_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['sample_id']}".encode("utf-8")
            ).digest(),
        )
        for index, row in enumerate(ordered):
            bucket_by_id[row["sample_id"]] = index % fold_count
    output = []
    for fold in range(fold_count):
        split_by_id = {}
        for row in rows:
            bucket = bucket_by_id[row["sample_id"]]
            if bucket == fold:
                split = "test"
            elif bucket == (fold + 1) % fold_count:
                split = "validation"
            else:
                split = "train"
            split_by_id[row["sample_id"]] = split
        output.append((f"fold_{fold:02d}", split_by_id))
    return output


def _assign_leave_one_domain(
    rows: list[dict[str, str]], field: str
) -> list[tuple[str, dict[str, str]]]:
    domains = sorted({row[field] for row in rows})
    if len(domains) < 3:
        raise ValueError(f"Protocol field {field!r} needs at least three domains, found {domains}")
    output = []
    for index, target in enumerate(domains):
        validation_domain = domains[(index + 1) % len(domains)]
        split_by_id = {
            row["sample_id"]: (
                "test"
                if row[field] == target
                else "validation"
                if row[field] == validation_domain
                else "train"
            )
            for row in rows
        }
        output.append((f"test_{_safe_name(target)}", split_by_id))
    return output


def _assign_cross_subject(
    rows: list[dict[str, str]], seed: int, test_subjects_per_fold: int, validation_subjects: int
) -> list[tuple[str, dict[str, str]]]:
    test_folds = _subject_test_folds(rows, seed, test_subjects_per_fold)
    output = []
    for index, test_subjects in enumerate(test_folds):
        test_set = set(test_subjects)
        validation = set(_choose_validation_subjects(rows, test_set, validation_subjects, seed + index))
        split_by_id = {
            row["sample_id"]: (
                "test"
                if row["subject"] in test_set
                else "validation"
                if row["subject"] in validation
                else "train"
            )
            for row in rows
        }
        output.append((f"fold_{index:02d}", split_by_id))
    return output


def _summary(rows: list[dict[str, str]], protocol: str) -> dict[str, Any]:
    counts = Counter(row["split"] for row in rows)
    subjects: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, set[str]] = defaultdict(set)
    domains: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in ("environment", "position", "orientation")
    }
    for row in rows:
        split = row["split"]
        subjects[split].add(row["subject"])
        labels[split].add(row["label"])
        for field in domains:
            if field in row:
                domains[field][split].add(row[field])
    subject_overlap = {
        f"{left}:{right}": sorted(subjects[left] & subjects[right])
        for left, right in itertools.combinations(("train", "validation", "test"), 2)
    }
    expected_labels = _label_set(rows)
    missing_labels = {
        split: sorted(expected_labels - labels[split]) for split in ("train", "validation", "test")
    }
    return {
        "protocol": protocol.upper(),
        "counts": dict(counts),
        "subjects": {split: sorted(values) for split, values in subjects.items()},
        "subject_overlap": subject_overlap,
        "labels": {split: sorted(values) for split, values in labels.items()},
        "missing_labels": missing_labels,
        "domains": {
            field: {split: sorted(values) for split, values in split_values.items()}
            for field, split_values in domains.items()
        },
    }


def build_protocol_splits(
    manifest: Path,
    output_root: Path,
    protocol: str,
    seed: int = 2025,
    id_folds: int = 5,
    test_subjects_per_fold: int = 3,
    validation_subjects: int = 2,
) -> dict[str, Any]:
    source_rows = _read_rows(manifest)
    if not source_rows:
        raise ValueError(f"Empty manifest: {manifest}")
    required = {"sample_id", "feature_path", "subject", "label"}
    missing = required - set(source_rows[0])
    if missing:
        raise ValueError(f"Manifest is missing required fields: {sorted(missing)}")
    protocol = protocol.lower()
    if protocol == "id":
        assignments = _assign_id(source_rows, seed, id_folds)
    elif protocol in PROTOCOL_FIELDS:
        field = PROTOCOL_FIELDS[protocol]
        if field not in source_rows[0]:
            raise ValueError(f"Protocol {protocol.upper()} requires manifest field {field!r}")
        assignments = _assign_leave_one_domain(source_rows, field)
    elif protocol == "cs":
        assignments = _assign_cross_subject(
            source_rows, seed, test_subjects_per_fold, validation_subjects
        )
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    reports = {}
    for fold_name, split_by_id in assignments:
        fold_dir = output_root / protocol.upper() / fold_name
        output_manifest = fold_dir / "manifest.csv"
        rows = []
        for source in source_rows:
            row = dict(source)
            row["split"] = split_by_id[row["sample_id"]]
            feature_path = Path(row["feature_path"])
            if not feature_path.is_absolute():
                feature_path = manifest.parent / feature_path
            row["feature_path"] = Path(os.path.relpath(feature_path.resolve(), fold_dir)).as_posix()
            rows.append(row)
        rows.sort(key=lambda row: row["sample_id"])
        report = _summary(rows, protocol)
        if any(report["missing_labels"].values()):
            raise ValueError(f"Fold {fold_name} has incomplete label coverage: {report['missing_labels']}")
        if protocol == "cs" and any(report["subject_overlap"].values()):
            raise ValueError(f"CS subject leakage in {fold_name}: {report['subject_overlap']}")
        if protocol in PROTOCOL_FIELDS:
            field = PROTOCOL_FIELDS[protocol]
            split_domains = report["domains"][field]
            if set(split_domains["test"]) & (
                set(split_domains["train"]) | set(split_domains["validation"])
            ):
                raise ValueError(f"Target {field} leaked in {fold_name}")
            if set(split_domains["validation"]) & set(split_domains["train"]):
                raise ValueError(f"Validation {field} leaked into training in {fold_name}")
        fold_dir.mkdir(parents=True, exist_ok=True)
        with output_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (fold_dir / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        reports[fold_name] = report
    protocol_root = output_root / protocol.upper()
    protocol_root.mkdir(parents=True, exist_ok=True)
    (protocol_root / "summary.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified ID/CL/CO/CE/CS Widar3 protocols")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol", choices=("id", "cl", "co", "ce", "cs"), required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--id-folds", type=int, default=5)
    parser.add_argument("--test-subjects-per-fold", type=int, default=3)
    parser.add_argument("--validation-subjects", type=int, default=2)
    args = parser.parse_args()
    reports = build_protocol_splits(
        Path(args.manifest).resolve(),
        Path(args.output_root).resolve(),
        args.protocol,
        seed=args.seed,
        id_folds=args.id_folds,
        test_subjects_per_fold=args.test_subjects_per_fold,
        validation_subjects=args.validation_subjects,
    )
    print(json.dumps({"protocol": args.protocol.upper(), "folds": len(reports)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
