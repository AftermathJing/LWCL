from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer frozen cross-environment splits to v2 features")
    parser.add_argument("--processed-manifest", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    processed_manifest = Path(args.processed_manifest).resolve()
    reference_root = Path(args.reference_root).resolve()
    output_root = Path(args.output_root).resolve()
    processed = read_rows(processed_manifest)
    processed_by_id = {row["sample_id"]: row for row in processed}
    if len(processed_by_id) != len(processed):
        raise ValueError("Processed manifest contains duplicate sample_id values")

    summary = {}
    for reference_manifest in sorted(reference_root.glob("*/manifest.csv")):
        target = reference_manifest.parent.name
        reference = read_rows(reference_manifest)
        split_by_id = {row["sample_id"]: row["split"] for row in reference}
        if len(split_by_id) != len(reference):
            raise ValueError(f"Duplicate sample_id values in {reference_manifest}")
        unexpected = sorted(set(processed_by_id) - set(split_by_id))
        if unexpected:
            raise ValueError(f"Processed samples absent from fold {target}: {unexpected[:10]}")

        fold_dir = output_root / target
        output_manifest = fold_dir / "manifest.csv"
        rows = []
        split_counts = Counter()
        split_domains: dict[str, set[str]] = defaultdict(set)
        split_subjects: dict[str, set[str]] = defaultdict(set)
        split_labels: dict[str, set[str]] = defaultdict(set)
        for sample_id, source in processed_by_id.items():
            row = dict(source)
            split = split_by_id[sample_id]
            row["split"] = split
            feature_path = Path(row["feature_path"])
            if not feature_path.is_absolute():
                feature_path = processed_manifest.parent / feature_path
            row["feature_path"] = Path(os.path.relpath(feature_path.resolve(), fold_dir)).as_posix()
            rows.append(row)
            split_counts[split] += 1
            split_domains[split].add(row["environment"])
            split_subjects[split].add(row["subject"])
            split_labels[split].add(row["label"])

        if split_domains["test"] != {target}:
            raise ValueError(f"Fold {target} does not isolate its target environment")
        if target in split_domains["train"] or target in split_domains["validation"]:
            raise ValueError(f"Target environment leaked into source data for fold {target}")
        if split_subjects["test"] & (split_subjects["train"] | split_subjects["validation"]):
            raise ValueError(f"Target subjects leaked into source data for fold {target}")
        if any(values != {str(label) for label in range(6)} for values in split_labels.values()):
            raise ValueError(f"Fold {target} does not cover all six labels in every split")

        rows.sort(key=lambda row: row["sample_id"])
        fold_dir.mkdir(parents=True, exist_ok=True)
        with output_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        missing_reference = sorted(set(split_by_id) - set(processed_by_id))
        fold_summary = {
            "target_environment": target,
            "counts": dict(split_counts),
            "domains": {key: sorted(value) for key, value in split_domains.items()},
            "subjects": {key: sorted(value) for key, value in split_subjects.items()},
            "labels": {key: sorted(value) for key, value in split_labels.items()},
            "missing_reference_samples": missing_reference,
        }
        (fold_dir / "summary.json").write_text(
            json.dumps(fold_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        summary[target] = fold_summary

    if not summary:
        raise ValueError(f"No reference fold manifests found under {reference_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
