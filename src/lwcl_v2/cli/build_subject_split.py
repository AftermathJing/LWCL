from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer a frozen subject split to LWCL-v2 features")
    parser.add_argument("--processed-manifest", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-missing-reference", action="store_true")
    args = parser.parse_args()

    processed_manifest = Path(args.processed_manifest).resolve()
    reference_manifest = Path(args.reference_manifest).resolve()
    output = Path(args.output).resolve()
    processed = read_rows(processed_manifest)
    reference = read_rows(reference_manifest)
    split_by_sample = {row["sample_id"]: row["split"] for row in reference}
    if len(split_by_sample) != len(reference):
        raise ValueError("Reference manifest contains duplicate sample_id values")

    processed_ids = {row["sample_id"] for row in processed}
    reference_ids = set(split_by_sample)
    missing_reference = sorted(reference_ids - processed_ids)
    unexpected = sorted(processed_ids - reference_ids)
    if unexpected or (missing_reference and not args.allow_missing_reference):
        raise ValueError(
            f"Manifest sample mismatch: missing={missing_reference[:10]}, unexpected={unexpected[:10]}"
        )

    rows = []
    subjects_by_split: dict[str, set[str]] = {}
    for row in processed:
        result = dict(row)
        split = split_by_sample[row["sample_id"]]
        result["split"] = split
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = processed_manifest.parent / feature_path
        result["feature_path"] = os.path.relpath(feature_path.resolve(), output.parent)
        rows.append(result)
        subjects_by_split.setdefault(split, set()).add(row["subject"])

    split_names = sorted(subjects_by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = subjects_by_split[left] & subjects_by_split[right]
            if overlap:
                raise ValueError(f"Subject leakage between {left} and {right}: {sorted(overlap)}")

    rows.sort(key=lambda row: row["sample_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        {
            "rows": len(rows),
            "missing_reference_samples": missing_reference,
            "splits": {split: sum(row["split"] == split for row in rows) for split in split_names},
            "subjects": {split: sorted(values) for split, values in subjects_by_split.items()},
        }
    )


if __name__ == "__main__":
    main()
