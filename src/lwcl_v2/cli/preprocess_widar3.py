from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from lwcl_v2.config import load_config
from lwcl_v2.data.preprocessing import QualityAwareCSIProcessor


SAMPLE_PATTERN = re.compile(
    r"(?P<date>\d+)_(?P<subject>user\d+)_g(?P<gesture>\d+)_p(?P<position>\d+)_o(?P<orientation>\d+)_i(?P<instance>\d+)"
)


def receiver_paths(input_root: Path, sample_id: str, num_receivers: int) -> list[Path]:
    match = SAMPLE_PATTERN.fullmatch(sample_id)
    if not match:
        raise ValueError(f"Unsupported Widar3 sample_id: {sample_id}")
    fields = match.groupdict()
    directory = input_root / fields["date"] / fields["subject"]
    stem = "-".join(
        (
            fields["subject"],
            fields["gesture"],
            fields["position"],
            fields["orientation"],
            fields["instance"],
        )
    )
    paths = [directory / f"{stem}-r{receiver}.dat" for receiver in range(1, num_receivers + 1)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing receiver files: {missing}")
    return paths


def process_one(task: tuple[dict[str, Any], dict[str, str], str, str]) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    config, row, input_root_text, output_root_text = task
    processor = QualityAwareCSIProcessor(config)
    input_root = Path(input_root_text)
    output_root = Path(output_root_text)
    sample_id = row["sample_id"]
    output_path = output_root / "features" / f"{sample_id}.npz"
    try:
        paths = receiver_paths(input_root, sample_id, processor.num_receivers)
        try:
            if output_path.exists():
                with np.load(output_path, allow_pickle=False) as archive:
                    _ = archive["time_mask"].shape, archive["receiver_mask"].shape
        except (OSError, ValueError, EOFError, KeyError):
            output_path.unlink(missing_ok=True)
        if not output_path.exists():
            processor.save_group(paths, output_path, metadata=row)
        with np.load(output_path, allow_pickle=False) as archive:
            time_steps = int(archive["time_mask"].sum())
            valid_receivers = int(archive["receiver_mask"].sum())
        result = dict(row)
        result.update(
            {
                "feature_path": str(Path("features") / output_path.name),
                "feature_format": "npz-v2",
                "time_steps": str(time_steps),
                "valid_receivers": str(valid_receivers),
                "preprocessing_profile": str(config.get("preprocessing", {}).get("profile", "custom")),
            }
        )
        return result, None
    except Exception as error:
        return None, {"sample_id": sample_id, "error_type": type(error).__name__, "error": str(error)}


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-aware Widar3 preprocessing for LWCL-v2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    config.setdefault("preprocessing", {})["profile"] = Path(args.config).stem.replace("preprocessing_", "").upper()
    with Path(args.source_manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]
    tasks = [(config, row, args.input_root, args.output_root) for row in rows]
    completed = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            result, failure = future.result()
            if result:
                completed.append(result)
            if failure:
                failures.append(failure)
            if index % 100 == 0 or index == len(futures):
                print(f"processed={index}/{len(futures)} valid={len(completed)} failures={len(failures)}", flush=True)
    completed.sort(key=lambda row: row["sample_id"])
    failures.sort(key=lambda row: row["sample_id"])
    output_root = Path(args.output_root)
    write_csv(output_root / "manifest.csv", completed)
    write_csv(output_root / "manifest.failures.csv", failures)
    print({"valid": len(completed), "failures": len(failures)})


if __name__ == "__main__":
    main()
