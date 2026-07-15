from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from lwcl_v2.config import load_config


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _find_one(run_dir: Path, relative_candidates: tuple[str, ...], filename: str) -> Path | None:
    for candidate in relative_candidates:
        path = run_dir / candidate
        if path.exists():
            return path
    matches = sorted(run_dir.rglob(filename))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {filename} under {run_dir}: {matches}")
    return matches[0] if matches else None


def _load_metrics(run_dir: Path, split: str) -> dict[str, Any] | None:
    filename = f"{split}_metrics.json"
    path = _find_one(
        run_dir,
        (filename, f"eval/{filename}", f"evaluation/{filename}", f"{split}/{filename}"),
        filename,
    )
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_seed(run_dir: Path) -> int | None:
    config_path = _find_one(
        run_dir,
        ("resolved_config.yaml", "train/resolved_config.yaml"),
        "resolved_config.yaml",
    )
    if config_path is not None:
        return int(load_config(config_path)["training"].get("seed", 2025))
    metadata_path = _find_one(
        run_dir,
        ("run_metadata.json", "train/run_metadata.json"),
        "run_metadata.json",
    )
    if metadata_path is not None:
        seed = json.loads(metadata_path.read_text(encoding="utf-8")).get("seed")
        return int(seed) if seed is not None else None
    return None


def _load_best_state(run_dir: Path) -> dict[str, Any] | None:
    checkpoint = _find_one(
        run_dir,
        ("checkpoints/best.pt", "train/checkpoints/best.pt"),
        "best.pt",
    )
    if checkpoint is not None:
        try:
            import torch

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload.get("state", {})
            return {
                "checkpoint": str(checkpoint),
                "epoch": state.get("epoch"),
                "global_step": state.get("global_step"),
                "selection_score": state.get("best_selection_score"),
                "macro_f1": state.get("best_macro_f1"),
                "worst_subject_macro_f1": state.get("best_worst_subject_macro_f1"),
            }
        except Exception as error:
            return {"checkpoint": str(checkpoint), "error": f"{type(error).__name__}: {error}"}
    metrics_log = _find_one(run_dir, ("metrics.jsonl", "train/metrics.jsonl"), "metrics.jsonl")
    if metrics_log is None:
        return None
    evaluations = []
    for line in metrics_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("event") == "evaluation":
            evaluations.append(payload)
    if not evaluations:
        return None
    best = max(evaluations, key=lambda item: float(item.get("selection_score", -1.0)))
    return {
        "checkpoint": None,
        "epoch": best.get("epoch"),
        "global_step": best.get("global_step"),
        "selection_score": best.get("selection_score"),
        "macro_f1": best.get("macro_f1"),
        "worst_subject_macro_f1": best.get("worst_subject_macro_f1"),
    }


def _aggregate_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    scalar_names = ("accuracy", "macro_f1", "worst_subject_macro_f1", "selection_score", "loss")
    aggregate: dict[str, Any] = {
        name: _summary([float(record[name]) for record in records if record.get(name) is not None])
        for name in scalar_names
    }
    class_count = max(len(record.get("per_class_f1", [])) for record in records)
    aggregate["per_class_f1"] = {
        str(index): _summary(
            [float(record["per_class_f1"][index]) for record in records if len(record.get("per_class_f1", [])) > index]
        )
        for index in range(class_count)
    }
    subjects = sorted(
        {subject for record in records for subject in record.get("subjects", {})}
    )
    aggregate["per_subject"] = {
        subject: {
            "runs": sum(subject in record.get("subjects", {}) for record in records),
            "accuracy": _summary(
                [
                    float(record["subjects"][subject]["accuracy"])
                    for record in records
                    if subject in record.get("subjects", {})
                ]
            ),
            "macro_f1": _summary(
                [
                    float(record["subjects"][subject]["macro_f1"])
                    for record in records
                    if subject in record.get("subjects", {})
                ]
            ),
        }
        for subject in subjects
    }
    observed_subject_scores = [
        (float(metrics["macro_f1"]), subject)
        for record in records
        for subject, metrics in record.get("subjects", {}).items()
    ]
    if observed_subject_scores:
        score, subject = min(observed_subject_scores)
        aggregate["worst_observed_subject"] = {"subject": subject, "macro_f1": score}
    return aggregate


def summarize_runs(run_dirs: list[Path]) -> dict[str, Any]:
    runs = []
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        validation = _load_metrics(run_dir, "validation")
        test = _load_metrics(run_dir, "test")
        if validation is None and test is None:
            raise ValueError(f"No validation/test metrics found under {run_dir}")
        runs.append(
            {
                "run_dir": str(run_dir),
                "seed": _load_seed(run_dir),
                "best_checkpoint": _load_best_state(run_dir),
                "validation": validation,
                "test": test,
            }
        )
    known_seeds = [run["seed"] for run in runs if run["seed"] is not None]
    if len(known_seeds) != len(set(known_seeds)):
        raise ValueError(f"Duplicate seeds in run set: {known_seeds}")
    best_states = [run["best_checkpoint"] for run in runs if run["best_checkpoint"]]
    return {
        "run_count": len(runs),
        "seeds": known_seeds,
        "best_checkpoint": {
            "epoch": _summary(
                [float(state["epoch"]) for state in best_states if state.get("epoch") is not None]
            ),
            "global_step": _summary(
                [
                    float(state["global_step"])
                    for state in best_states
                    if state.get("global_step") is not None
                ]
            ),
        },
        "validation": _aggregate_split([run["validation"] for run in runs if run["validation"]]),
        "test": _aggregate_split([run["test"] for run in runs if run["test"]]),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Signal-v2 multi-seed evaluations")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--glob", default="seed_*")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.runs_root)
    run_dirs = sorted(path for path in root.glob(args.glob) if path.is_dir())
    if not run_dirs:
        raise ValueError(f"No run directories matched {args.glob!r} under {root}")
    report = summarize_runs(run_dirs)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    output = Path(args.output) if args.output else root / "multiseed_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
