from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray, num_labels: int) -> np.ndarray:
    matrix = np.zeros((num_labels, num_labels), dtype=np.int64)
    for target, prediction in zip(targets.astype(int), predictions.astype(int), strict=True):
        matrix[target, prediction] += 1
    return matrix


def classification_metrics(targets: np.ndarray, predictions: np.ndarray, num_labels: int) -> dict[str, Any]:
    matrix = confusion_matrix(targets, predictions, num_labels)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / np.maximum(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(true_positive.sum() / max(matrix.sum(), 1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": matrix.sum(axis=1).tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def subject_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    subjects: list[str],
    num_labels: int,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, subject in enumerate(subjects):
        groups[subject].append(index)
    output = {}
    for subject, indexes in sorted(groups.items()):
        metrics = classification_metrics(targets[indexes], predictions[indexes], num_labels)
        present = [index for index, support in enumerate(metrics["support"]) if support > 0]
        present_macro_f1 = float(
            np.mean([metrics["per_class_f1"][index] for index in present])
        )
        output[subject] = {
            "count": len(indexes),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_f1_present_labels": present_macro_f1,
            "present_labels": present,
        }
    return output
