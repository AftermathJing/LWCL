from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from lwcl_v2.config import load_config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_tuning_runs(
    run_root: str | Path,
    *,
    candidate_ids: set[str] | None = None,
    strict_validation_only: bool = False,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for candidate_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = candidate_dir / "train" / "resolved_config.yaml"
        metrics_path = candidate_dir / "eval" / "validation_metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            missing.append(candidate_dir.name)
            continue
        config = load_config(config_path)
        candidate_id = str(config.get("experiment", {}).get("id", candidate_dir.name))
        if candidate_ids is not None and candidate_id not in candidate_ids:
            continue
        test_artifacts = sorted(str(path) for path in candidate_dir.rglob("test_metrics.json"))
        if strict_validation_only and test_artifacts:
            raise ValueError(f"Screening run {candidate_id} contains test artifacts: {test_artifacts}")
        metrics = _read_json(metrics_path)
        parameter_report = _read_json(candidate_dir / "train" / "parameter_report.json")
        metadata = _read_json(candidate_dir / "train" / "run_metadata.json")
        runtime_path = candidate_dir / "train" / "runtime_report.json"
        runtime = _read_json(runtime_path) if runtime_path.exists() else {}
        user17 = metrics.get("subjects", {}).get("user17", {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_dir": str(candidate_dir),
                "config": str(config_path),
                "checkpoint": str(candidate_dir / "train" / "checkpoints" / "best.pt"),
                "config_hash": metadata.get("config_hash"),
                "manifest_sha256": metadata.get("manifest_sha256"),
                "seed": metadata.get("seed"),
                "parameters": parameter_report["total"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "user17_macro_f1": user17.get("macro_f1"),
                "user17_accuracy": user17.get("accuracy"),
                "best_epoch": metrics.get("epoch"),
                "best_global_step": metrics.get("global_step"),
                "peak_memory_bytes": runtime.get("peak_memory_bytes"),
                "train_seconds": runtime.get("train_seconds"),
                "test_artifacts": test_artifacts,
            }
        )
    rows.sort(key=lambda row: (-float(row["macro_f1"]), row["candidate_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {"run_root": str(root), "count": len(rows), "missing": missing, "ranking": rows}


def write_summary(report: dict[str, Any], output_json: Path, output_csv: Path | None = None) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_csv is None:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = report["ranking"]
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return
    fields = [key for key in rows[0] if key != "test_artifacts"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank validation-only Signal-v2 tuning runs")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--candidate-ids", nargs="*")
    parser.add_argument("--require-count", type=int)
    parser.add_argument("--strict-validation-only", action="store_true")
    parser.add_argument("--freeze-top", type=int, default=0)
    parser.add_argument("--freeze-output")
    args = parser.parse_args()
    report = summarize_tuning_runs(
        args.run_root,
        candidate_ids=set(args.candidate_ids) if args.candidate_ids else None,
        strict_validation_only=args.strict_validation_only,
    )
    if args.require_count is not None and report["count"] != args.require_count:
        raise ValueError(f"Expected {args.require_count} completed runs, got {report['count']}")
    write_summary(
        report,
        Path(args.output_json),
        Path(args.output_csv) if args.output_csv else None,
    )
    if args.freeze_top:
        if not args.freeze_output:
            raise ValueError("--freeze-output is required with --freeze-top")
        frozen = {
            "selection_metric": "source_validation_macro_f1",
            "count": args.freeze_top,
            "candidates": report["ranking"][: args.freeze_top],
        }
        Path(args.freeze_output).write_text(
            json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
