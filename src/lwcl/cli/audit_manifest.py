from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from lwcl.data.dataset import SignalDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit manifest paths and model-facing sample shapes")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-seq-len", type=int, default=500)
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--check-all-paths", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing: list[str] = []
    if args.check_all_paths:
        for row in rows:
            path = Path(row["feature_path"])
            if not path.is_absolute():
                path = manifest.parent / path
            if not path.exists():
                missing.append(row["sample_id"])

    dataset = SignalDataset(manifest, args.split, max_seq_len=args.max_seq_len, training=False)
    shapes: dict[str, int] = {}
    labels: list[int] = []
    sample_ids: list[str] = []
    for index in range(min(args.sample_limit, len(dataset))):
        item = dataset[index]
        shape = "x".join(str(int(value)) for value in item["features"].shape)
        shapes[shape] = shapes.get(shape, 0) + 1
        labels.append(int(item["label"]))
        sample_ids.append(item["sample_id"])
    result = {
        "manifest": str(manifest.resolve()),
        "rows": len(rows),
        "split": args.split,
        "split_rows": len(dataset),
        "checked_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "shapes": shapes,
        "labels": labels,
        "missing_paths": len(missing),
        "missing_examples": missing[:10],
    }
    print(json.dumps(result, ensure_ascii=False))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
