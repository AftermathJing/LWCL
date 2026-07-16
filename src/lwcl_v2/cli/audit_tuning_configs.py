from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lwcl_v2.config import config_hash, load_config
from lwcl_v2.models import SignalV2Classifier


def audit_configs(paths: list[str | Path], typical_frames: int = 31) -> dict[str, Any]:
    rows = []
    ids = set()
    errors = []
    for value in paths:
        path = Path(value).resolve()
        config = load_config(path)
        candidate_id = str(config.get("experiment", {}).get("id", path.stem))
        if candidate_id in ids:
            errors.append(f"duplicate candidate id: {candidate_id}")
        ids.add(candidate_id)
        training = config["training"]
        selection = training.get("selection", {})
        model = SignalV2Classifier(config)
        parameters = model.parameter_report()
        hste = config["model"]["hste"]
        window = int(hste.get("window_size", 5))
        stride = int(hste.get("window_stride", 2))
        tokens = 1 if typical_frames < window else 1 + (typical_frames - window) // stride
        row = {
            "candidate_id": candidate_id,
            "path": str(path),
            "config_hash": config_hash(config),
            "feature_mode": str(config["data"].get("feature_mode", "current")),
            "doppler_bins": int(config["data"].get("doppler_bins", 25)),
            "parameters": parameters["total"],
            "parameter_delta": parameters["total"] - 2_009_186,
            "parameter_ratio": parameters["total"] / 2_009_186,
            "typical_global_tokens": tokens,
            "use_ema": bool(training.get("use_ema", False)),
            "selection_macro_f1_weight": float(selection.get("macro_f1_weight", 0.7)),
            "selection_worst_subject_weight": float(
                selection.get("worst_subject_macro_f1_weight", 0.3)
            ),
        }
        rows.append(row)
        if row["parameters"] > 4_000_000:
            errors.append(f"{candidate_id} exceeds 4M parameters: {row['parameters']}")
        if row["use_ema"]:
            errors.append(f"{candidate_id} unexpectedly enables EMA")
        if row["selection_macro_f1_weight"] != 1.0 or row[
            "selection_worst_subject_weight"
        ] != 0.0:
            errors.append(f"{candidate_id} does not use validation macro-F1-only selection")
    return {"count": len(rows), "errors": errors, "candidates": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen Signal-v2 tuning matrix")
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-count", type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = audit_configs(args.configs)
    if args.require_count is not None and report["count"] != args.require_count:
        report["errors"].append(
            f"expected {args.require_count} candidates, got {report['count']}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if args.strict and report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
