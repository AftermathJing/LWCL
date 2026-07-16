"""Preprocess all CSI-Bench HAR samples into Signal-v2 NPZ feature files.

Usage:
  python -m lwcl_v2.cli.preprocess_csi_bench_har \\
      --manifest data/splits/csi_bench_har/test_id/manifest.csv \\
      --output-root data/processed/csi_bench_har/test_id \\
      --workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv

import numpy as np

from lwcl_v2.data.csi_bench import load_csi_bench_amplitude


def _process_one(row_data: dict) -> tuple[str, bool, str]:
    sample_id = str(row_data["sample_id"])
    feature_path = str(row_data["feature_path"])
    output_path = str(row_data["output_npz"])
    try:
        result = load_csi_bench_amplitude(
            feature_path,
            num_subcarriers=int(float(row_data.get("num_sub", 0) or 0)) or None,
            num_devices=int(row_data.get("device_count", 1) or 1),
        )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **result)
        return (sample_id, True, "")
    except Exception as exc:
        return (sample_id, False, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess CSI-Bench HAR samples into NPZ feature files")
    parser.add_argument("--manifest", required=True, help="Path to HAR manifest CSV")
    parser.add_argument("--output-root", required=True, help="Directory to write NPZ files")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--processed-manifest", default=None, help="Write a new manifest CSV pointing to the generated NPZ files")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    tasks: list[dict] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        rel = f"{sample_id}.npz"
        row["output_npz"] = str(output_root / rel)

    print(f"{len(rows)} samples to process with {args.workers} workers", flush=True)
    success = 0
    failed = 0
    completed = 0
    results: dict[str, str] = {}  # sample_id -> rel_npz_path

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one, row): row for row in rows}
        for future in as_completed(futures):
            completed += 1
            sample_id, ok, error = future.result()
            if ok:
                success += 1
            else:
                failed += 1
                print(f"  FAILED {sample_id}: {error}", file=sys.stderr, flush=True)
            if completed % 1000 == 0:
                print(f"  progress: {completed}/{len(rows)}  ok={success}  fail={failed}", flush=True)

    if args.processed_manifest:
        pm = Path(args.processed_manifest)
        pm.parent.mkdir(parents=True, exist_ok=True)
        with open(pm, "w", encoding="utf-8", newline="") as out:
            new_rows = []
            for row in rows:
                sid = str(row["sample_id"])
                if sid in results:
                    nr = dict(row)
                    nr["feature_path"] = str(output_root / results[sid])
                    nr["feature_format"] = "npz"
                    new_rows.append(nr)
            if new_rows:
                fieldnames = list(new_rows[0].keys())
                writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(new_rows)
        print(f"Wrote processed manifest ({len(new_rows)} rows) to {pm}", flush=True)

    print(f"Done: {success} ok, {failed} failed out of {len(rows)}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
            if ok:
                results[sample_id] = f"{sample_id}.npz"
