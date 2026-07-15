from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from lwcl.data.preprocessing import CSIToFeatureProcessor


_WORKER_PROCESSOR: CSIToFeatureProcessor | None = None


def _initialize_worker(sample_rate: int) -> None:
    global _WORKER_PROCESSOR
    _WORKER_PROCESSOR = CSIToFeatureProcessor(sample_rate=sample_rate)


def _process_job(job: dict[str, Any]) -> tuple[int, dict[str, str] | None, str | None, bool]:
    global _WORKER_PROCESSOR
    if _WORKER_PROCESSOR is None:
        _WORKER_PROCESSOR = CSIToFeatureProcessor(sample_rate=int(job["sample_rate"]))
    row = dict(job["row"])
    sample_id = row["sample_id"]
    raw_path = Path(row["raw_path"])
    if not raw_path.is_absolute():
        raw_path = Path(job["input_manifest_parent"]) / raw_path
    output_path = Path(job["output_dir"]) / "features" / f"{sample_id}.npz"
    resumed = False
    try:
        if bool(job["resume"]) and output_path.exists():
            with np.load(output_path, allow_pickle=False) as archive:
                features = archive["features"]
                if features.ndim != 3:
                    raise ValueError(f"Invalid existing feature shape {features.shape}")
            resumed = True
        else:
            _WORKER_PROCESSOR.save_sample(raw_path, output_path, int(job["num_receivers"]), row)
    except Exception as exc:
        return int(job["index"]), None, str(exc), resumed
    row.pop("raw_path", None)
    row["feature_path"] = Path(
        os.path.relpath(output_path, Path(job["output_manifest_parent"]))
    ).as_posix()
    row["feature_format"] = "npz"
    return int(job["index"]), row, None, resumed


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Intel 5300 CSI sample directories to LWCL feature archives")
    parser.add_argument("--input-manifest", required=True, help="CSV containing sample_id, raw_path, label and metadata")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--num-receivers", type=int, default=6)
    parser.add_argument("--sample-rate", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    input_manifest = Path(args.input_manifest)
    with input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_dir = Path(args.output_dir)
    output_manifest = Path(args.output_manifest)
    jobs = [
        {
            "index": index,
            "row": row,
            "input_manifest_parent": str(input_manifest.parent.resolve()),
            "output_dir": str(output_dir.resolve()),
            "output_manifest_parent": str(output_manifest.parent.resolve()),
            "num_receivers": args.num_receivers,
            "sample_rate": args.sample_rate,
            "resume": args.resume,
        }
        for index, row in enumerate(rows)
    ]
    indexed_rows: dict[int, dict[str, str]] = {}
    failures: list[tuple[str, str]] = []
    resumed_count = 0
    completed = 0

    if args.workers <= 1:
        _initialize_worker(args.sample_rate)
        results = (_process_job(job) for job in jobs)
        for index, output_row, error, resumed in results:
            completed += 1
            resumed_count += int(resumed)
            if error is not None:
                failures.append((rows[index]["sample_id"], error))
                if not args.continue_on_error:
                    raise RuntimeError(f"FAILED {rows[index]['sample_id']}: {error}")
            elif output_row is not None:
                indexed_rows[index] = output_row
            if completed % max(args.progress_every, 1) == 0:
                print(f"progress={completed}/{len(rows)} failed={len(failures)} resumed={resumed_count}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_worker,
            initargs=(args.sample_rate,),
        ) as executor:
            future_map = {executor.submit(_process_job, job): job for job in jobs}
            for future in as_completed(future_map):
                index, output_row, error, resumed = future.result()
                completed += 1
                resumed_count += int(resumed)
                if error is not None:
                    failures.append((rows[index]["sample_id"], error))
                    if not args.continue_on_error:
                        for pending in future_map:
                            pending.cancel()
                        raise RuntimeError(f"FAILED {rows[index]['sample_id']}: {error}")
                elif output_row is not None:
                    indexed_rows[index] = output_row
                if completed % max(args.progress_every, 1) == 0:
                    print(f"progress={completed}/{len(rows)} failed={len(failures)} resumed={resumed_count}", flush=True)

    output_rows = [indexed_rows[index] for index in sorted(indexed_rows)]
    if not output_rows:
        raise RuntimeError("No samples were successfully processed")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    failures_path = output_manifest.with_suffix(".failures.csv")
    if failures:
        with failures_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "error"])
            writer.writerows(failures)
    first_path = Path(output_rows[0]["feature_path"])
    if not first_path.is_absolute():
        first_path = output_manifest.parent / first_path
    with np.load(first_path, allow_pickle=False) as archive:
        first_shape = tuple(int(value) for value in archive["features"].shape)
    print(
        f"processed={len(output_rows)} failed={len(failures)} resumed={resumed_count} "
        f"first_shape={first_shape} manifest={output_manifest}"
    )
    for sample_id, error in failures:
        print(f"FAILED {sample_id}: {error}")


if __name__ == "__main__":
    main()
