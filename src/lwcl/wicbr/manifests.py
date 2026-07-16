from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def attach_wicbr_paths(
    input_manifest: str | Path,
    image_root: str | Path,
    output_manifest: str | Path,
    *,
    phase_source: str = "weighted",
    require_all: bool = True,
) -> dict[str, Any]:
    input_manifest = Path(input_manifest).resolve()
    image_root = Path(image_root).resolve()
    output_manifest = Path(output_manifest).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    phase_dir = image_root / ("phase_weighted" if phase_source == "weighted" else "phase_raw")
    phase_raw_dir = image_root / "phase_raw"
    phase_weighted_dir = image_root / "phase_weighted"
    dfs_dir = image_root / "dfs"
    dfs_weight_dir = image_root / "dfs_weight"

    with input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        phase_raw_path = phase_raw_dir / f"{sample_id}.jpg"
        phase_weighted_path = phase_weighted_dir / f"{sample_id}.jpg"
        phase_path = phase_dir / f"{sample_id}.jpg"
        dfs_path = dfs_dir / f"{sample_id}.jpg"
        dfs_weight_path = dfs_weight_dir / f"{sample_id}.jpg"
        missing = [
            str(path)
            for path in (phase_raw_path, phase_weighted_path, phase_path, dfs_path, dfs_weight_path)
            if not path.exists()
        ]
        if missing and require_all:
            failures.append({"sample_id": sample_id, "error": "missing image files", "missing": ";".join(missing)})
            continue
        output_row = dict(row)
        output_row["phase_raw_path"] = _relative(phase_raw_path, output_manifest.parent)
        output_row["phase_weighted_path"] = _relative(phase_weighted_path, output_manifest.parent)
        output_row["phase_path"] = _relative(phase_path, output_manifest.parent)
        output_row["dfs_path"] = _relative(dfs_path, output_manifest.parent)
        output_row["dfs_weight_path"] = _relative(dfs_weight_path, output_manifest.parent)
        output_rows.append(output_row)

    fieldnames = list(output_rows[0].keys()) if output_rows else [
        "sample_id",
        "phase_raw_path",
        "phase_weighted_path",
        "phase_path",
        "dfs_path",
        "dfs_weight_path",
    ]
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    failures_path = output_manifest.with_suffix(".failures.csv")
    with failures_path.open("w", encoding="utf-8", newline="") as handle:
        if failures:
            writer = csv.DictWriter(handle, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)
        else:
            handle.write("sample_id,error\n")

    summary = {
        "input_manifest": str(input_manifest),
        "image_root": str(image_root),
        "output_manifest": str(output_manifest),
        "phase_source": phase_source,
        "rows_written": len(output_rows),
        "failures": len(failures),
    }
    output_manifest.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
