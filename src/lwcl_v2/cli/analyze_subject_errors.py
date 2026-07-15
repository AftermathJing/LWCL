from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lwcl_v2.training.metrics import classification_metrics


def _group_metrics(
    targets: np.ndarray, predictions: np.ndarray, groups: np.ndarray, num_labels: int
) -> dict[str, Any]:
    output = {}
    for value in sorted(set(groups.tolist())):
        selected = groups == value
        metrics = classification_metrics(targets[selected], predictions[selected], num_labels)
        output[str(value)] = {
            "count": int(selected.sum()),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
    return output


def _calibration(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> dict[str, float]:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        selected = (confidences > edges[index]) & (confidences <= edges[index + 1])
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidences[selected].mean())
    nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()
    return {
        "ece": float(ece),
        "nll": float(nll),
        "mean_confidence": float(confidences.mean()),
        "mean_error_confidence": float(confidences[~correct].mean()) if (~correct).any() else 0.0,
    }


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    return float(1.0 - float(np.dot(left, right)) / denominator)


def _reference_distances(
    focus: dict[str, np.ndarray], reference_path: Path
) -> dict[str, Any]:
    with np.load(reference_path, allow_pickle=False) as archive:
        ref_subjects = archive["subjects"].astype(str)
        ref_labels = archive["labels"].astype(np.int64)
        ref_embeddings = archive["embeddings"].astype(np.float64)
    output = {}
    for subject in sorted(set(ref_subjects.tolist())):
        per_class = {}
        for label in sorted(set(focus["labels"].tolist())):
            focus_selected = focus["labels"] == label
            reference_selected = (ref_subjects == subject) & (ref_labels == label)
            if focus_selected.any() and reference_selected.any():
                per_class[str(label)] = _cosine_distance(
                    focus["embeddings"][focus_selected].mean(axis=0),
                    ref_embeddings[reference_selected].mean(axis=0),
                )
        output[subject] = {
            "shared_classes": len(per_class),
            "mean_cosine_distance": float(np.mean(list(per_class.values()))) if per_class else None,
            "per_class_cosine_distance": per_class,
        }
    return output


def analyze_subject(
    outputs_path: Path,
    focus_subject: str,
    reference_path: Path | None = None,
    num_labels: int = 6,
) -> dict[str, Any]:
    with np.load(outputs_path, allow_pickle=False) as archive:
        subjects = archive["subjects"].astype(str)
        selected = subjects == focus_subject
        if not selected.any():
            raise ValueError(f"Subject {focus_subject!r} is absent from {outputs_path}")
        focus = {name: archive[name][selected] for name in archive.files if len(archive[name]) == len(subjects)}
    targets = focus["labels"].astype(np.int64)
    predictions = focus["predictions"].astype(np.int64)
    probabilities = focus["probabilities"].astype(np.float64)
    metrics = classification_metrics(targets, predictions, num_labels)
    lengths = focus["sequence_lengths"].astype(np.int64)
    lower, upper = np.quantile(lengths, [1 / 3, 2 / 3])
    length_bins = np.where(lengths <= lower, "short", np.where(lengths >= upper, "long", "medium"))
    report: dict[str, Any] = {
        "subject": focus_subject,
        "count": int(len(targets)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "support": metrics["support"],
        "confusion_matrix": metrics["confusion_matrix"],
        "calibration": _calibration(probabilities, targets),
        "by_position": _group_metrics(targets, predictions, focus["positions"].astype(str), num_labels),
        "by_orientation": _group_metrics(
            targets, predictions, focus["orientations"].astype(str), num_labels
        ),
        "by_sequence_length": {
            "thresholds": {"short_max": float(lower), "long_min": float(upper)},
            "groups": _group_metrics(targets, predictions, length_bins, num_labels),
        },
        "by_valid_receivers": _group_metrics(
            targets, predictions, focus["valid_receivers"].astype(str), num_labels
        ),
        "receiver_quality_mean": focus["receiver_quality_mean"].mean(axis=0).tolist(),
    }
    if reference_path is not None:
        report["distance_to_reference_subjects"] = _reference_distances(focus, reference_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a held-out subject from exported outputs")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--focus-subject", default="user17")
    parser.add_argument("--reference-outputs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze_subject(
        Path(args.outputs),
        args.focus_subject,
        Path(args.reference_outputs) if args.reference_outputs else None,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
