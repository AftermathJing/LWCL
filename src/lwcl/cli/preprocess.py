from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from lwcl.data.preprocessing import CSIToFeatureProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Intel 5300 CSI sample directories to LWCL feature archives")
    parser.add_argument("--input-manifest", required=True, help="CSV containing sample_id, raw_path, label and metadata")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--num-receivers", type=int, default=6)
    parser.add_argument("--sample-rate", type=int, default=1000)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    input_manifest = Path(args.input_manifest)
    with input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    processor = CSIToFeatureProcessor(sample_rate=args.sample_rate)
    output_dir = Path(args.output_dir)
    output_rows: list[dict[str, str]] = []
    failures: list[tuple[str, str]] = []
    for row in rows:
        sample_id = row["sample_id"]
        raw_path = Path(row["raw_path"])
        if not raw_path.is_absolute():
            raw_path = input_manifest.parent / raw_path
        output_path = output_dir / "features" / f"{sample_id}.npz"
        try:
            processor.save_sample(raw_path, output_path, args.num_receivers, row)
        except Exception as exc:
            failures.append((sample_id, str(exc)))
            if not args.continue_on_error:
                raise
            continue
        output_row = dict(row)
        output_row.pop("raw_path", None)
        output_row["feature_path"] = Path(
            os.path.relpath(output_path, Path(args.output_manifest).parent)
        ).as_posix()
        output_rows.append(output_row)
    if not output_rows:
        raise RuntimeError("No samples were successfully processed")
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"processed={len(output_rows)} failed={len(failures)} manifest={output_manifest}")
    for sample_id, error in failures:
        print(f"FAILED {sample_id}: {error}")


if __name__ == "__main__":
    main()
