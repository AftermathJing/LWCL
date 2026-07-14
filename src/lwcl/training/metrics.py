from __future__ import annotations

import numpy as np


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray, num_labels: int) -> np.ndarray:
    matrix = np.zeros((num_labels, num_labels), dtype=np.int64)
    for target, prediction in zip(targets.astype(int), predictions.astype(int), strict=True):
        matrix[target, prediction] += 1
    return matrix


def classification_metrics(targets: np.ndarray, predictions: np.ndarray, num_labels: int) -> dict[str, object]:
    matrix = confusion_matrix(targets, predictions, num_labels)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / np.maximum(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = matrix.sum(axis=1)
    accuracy = float(true_positive.sum() / max(matrix.sum(), 1))
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def grouped_accuracy(
    targets: np.ndarray,
    predictions: np.ndarray,
    metadata: list[dict[str, str]],
    fields: tuple[str, ...] = ("subject", "environment", "position", "orientation"),
) -> dict[str, dict[str, dict[str, float | int]]]:
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in fields:
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(metadata):
            if row.get(field):
                groups.setdefault(row[field], []).append(index)
        output[field] = {}
        for group, indexes in sorted(groups.items()):
            group_targets = targets[indexes]
            group_predictions = predictions[indexes]
            output[field][group] = {
                "accuracy": float((group_targets == group_predictions).mean()),
                "count": len(indexes),
            }
    return output
