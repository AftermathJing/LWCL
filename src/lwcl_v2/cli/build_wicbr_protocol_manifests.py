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


def build_protocol_manifest(
    processed_manifest: Path,
    official_manifest: Path,
    output_manifest: Path,
    duplicate_test_as_validation: bool = True,
    allow_missing_reference: bool = True,
) -> dict[str, object]:
    processed_rows = read_rows(processed_manifest)
    official_rows = read_rows(official_manifest)
    processed_by_id = {row["sample_id"]: row for row in processed_rows}
    if len(processed_by_id) != len(processed_rows):
        raise ValueError(f"Processed manifest contains duplicate sample_id values: {processed_manifest}")

    rows: list[dict[str, str]] = []
    missing_reference: list[str] = []
    counts = Counter()
    envs: dict[str, set[str]] = defaultdict(set)
    subjects: dict[str, set[str]] = defaultdict(set)

    for official in official_rows:
        sample_id = official["sample_id"]
        processed = processed_by_id.get(sample_id)
        if processed is None:
            missing_reference.append(sample_id)
            if not allow_missing_reference:
                raise ValueError(f"Reference sample missing from processed manifest: {sample_id}")
            continue

        feature_path = Path(processed["feature_path"])
        if not feature_path.is_absolute():
            feature_path = processed_manifest.parent / feature_path
        rel_feature = Path(os.path.relpath(feature_path.resolve(), output_manifest.parent)).as_posix()

        base = dict(official)
        for key, value in processed.items():
            if key not in base:
                base[key] = value
        base["feature_path"] = rel_feature
        base["feature_format"] = processed.get("feature_format", "npz-v2")
        base["time_steps"] = processed.get("time_steps", "")
        base["valid_receivers"] = processed.get("valid_receivers", "")
        base["preprocessing_profile"] = processed.get("preprocessing_profile", "")
        base["official_protocol"] = official_manifest.stem
        base["official_split"] = official["split"]

        split = official["split"]
        if split == "train":
            row = dict(base)
            row["split"] = "train"
            rows.append(row)
            counts["train"] += 1
            envs["train"].add(row.get("environment", ""))
            subjects["train"].add(row.get("subject", ""))
        elif split == "test":
            if duplicate_test_as_validation:
                validation = dict(base)
                validation["split"] = "validation"
                rows.append(validation)
                counts["validation"] += 1
                envs["validation"].add(validation.get("environment", ""))
                subjects["validation"].add(validation.get("subject", ""))
            test = dict(base)
            test["split"] = "test"
            rows.append(test)
            counts["test"] += 1
            envs["test"].add(test.get("environment", ""))
            subjects["test"].add(test.get("subject", ""))
        else:
            raise ValueError(f"Unsupported official split value {split!r} in {official_manifest}")

    rows.sort(key=lambda row: (row["split"], row["sample_id"]))
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": official_manifest.stem,
        "rows": len(rows),
        "counts": dict(counts),
        "environments": {key: sorted(value) for key, value in envs.items()},
        "subjects": {key: sorted(value) for key, value in subjects.items()},
        "missing_reference_samples": missing_reference,
    }
    (output_manifest.parent / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map Wi-CBR official Widar3 manifests onto Signal-v2 processed features"
    )
    parser.add_argument("--processed-manifest", required=True)
    parser.add_argument("--official-split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocols", nargs="+", default=["cr1", "cr2", "cr3"])
    parser.add_argument("--no-duplicate-test-as-validation", action="store_true")
    parser.add_argument("--strict-missing-reference", action="store_true")
    args = parser.parse_args()

    processed_manifest = Path(args.processed_manifest).resolve()
    official_dir = Path(args.official_split_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    all_summaries = {}
    for protocol in args.protocols:
        official_manifest = official_dir / f"{protocol}.csv"
        if not official_manifest.exists():
            raise FileNotFoundError(official_manifest)
        output_manifest = output_dir / protocol / "manifest.csv"
        all_summaries[protocol] = build_protocol_manifest(
            processed_manifest=processed_manifest,
            official_manifest=official_manifest,
            output_manifest=output_manifest,
            duplicate_test_as_validation=not args.no_duplicate_test_as_validation,
            allow_missing_reference=not args.strict_missing_reference,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(all_summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
