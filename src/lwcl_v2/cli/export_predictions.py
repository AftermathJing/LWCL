from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lwcl_v2.config import load_config
from lwcl_v2.data.dataset import SignalV2Dataset, collate_signal_v2
from lwcl_v2.models import SignalV2Classifier
from lwcl_v2.training.checkpoint import load_checkpoint
from lwcl_v2.training.ema import ExponentialMovingAverage
from lwcl_v2.training.metrics import classification_metrics, subject_metrics
from lwcl_v2.training.trainer import seed_everything


def calibration_metrics(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 15
) -> dict[str, float]:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidences > lower) & (confidences <= upper)
        if not selected.any():
            continue
        ece += selected.mean() * abs(correct[selected].mean() - confidences[selected].mean())
    nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()
    return {
        "ece": float(ece),
        "nll": float(nll),
        "mean_confidence": float(confidences.mean()),
        "mean_error_confidence": float(confidences[~correct].mean()) if (~correct).any() else 0.0,
    }


def _metadata_value(metadata: dict[str, str], name: str) -> str:
    return metadata.get(name) or "unknown"


@torch.no_grad()
def export_predictions(
    config: dict[str, Any],
    checkpoint: Path,
    output_dir: Path,
    split: str,
    weights: str = "raw",
) -> dict[str, Any]:
    seed_everything(int(config["training"].get("seed", 2025)))
    dataset = SignalV2Dataset(
        config["data"]["manifest"],
        split=split,
        max_seq_len=int(config["data"]["max_seq_len"]),
        seed=int(config["training"].get("seed", 2025)),
        min_valid_receivers=int(config["data"].get("min_valid_receivers", 5)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("eval_batch_size", 192)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=False,
        collate_fn=partial(collate_signal_v2, max_seq_len=int(config["data"]["max_seq_len"])),
    )
    model = SignalV2Classifier(config)
    ema = ExponentialMovingAverage(model, float(config["training"].get("ema_decay", 0.999))) if weights == "ema" else None
    payload = load_checkpoint(checkpoint, model, ema=ema, restore_rng=False)
    if weights == "ema" and payload.get("ema") is None:
        raise ValueError(f"Checkpoint {checkpoint} does not contain EMA weights")
    device_name = config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model.to(device).eval()
    precision = config["training"].get("precision", "bf16")
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    parameter_context = ema.average_parameters(model) if ema is not None else nullcontext()

    arrays: dict[str, list[Any]] = {
        "sample_ids": [],
        "subjects": [],
        "labels": [],
        "predictions": [],
        "probabilities": [],
        "embeddings": [],
        "sequence_lengths": [],
        "valid_receivers": [],
        "receiver_quality_mean": [],
        "environments": [],
        "positions": [],
        "orientations": [],
    }
    with parameter_context:
        for batch in loader:
            arrays["sample_ids"].extend(batch["sample_ids"])
            arrays["subjects"].extend(batch["subjects"])
            arrays["environments"].extend(
                [_metadata_value(metadata, "environment") for metadata in batch["metadata"]]
            )
            arrays["positions"].extend(
                [_metadata_value(metadata, "position") for metadata in batch["metadata"]]
            )
            arrays["orientations"].extend(
                [_metadata_value(metadata, "orientation") for metadata in batch["metadata"]]
            )
            receiver_mask = batch["receiver_mask"]
            quality = batch["receiver_quality"]
            quality_mean = (quality * receiver_mask.unsqueeze(-1)).sum(dim=1) / receiver_mask.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
            arrays["sequence_lengths"].extend(batch["time_mask"].sum(dim=1).tolist())
            arrays["valid_receivers"].extend(receiver_mask.sum(dim=1).tolist())
            arrays["receiver_quality_mean"].append(quality_mean.numpy())
            moved = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            autocast = (
                torch.autocast("cuda", dtype=autocast_dtype)
                if device.type == "cuda" and autocast_dtype is not None
                else nullcontext()
            )
            with autocast:
                outputs = model(
                    rssi=moved["rssi"],
                    doppler=moved["doppler"],
                    differential_csi=moved["differential_csi"],
                    time_mask=moved["time_mask"],
                    receiver_mask=moved["receiver_mask"],
                    receiver_quality=moved["receiver_quality"],
                    position_ids=moved["position_ids"],
                    frame_times_ms=moved["frame_times_ms"],
                    csi_ratio_phase=moved.get("csi_ratio_phase"),
                )
            probabilities = torch.softmax(outputs["logits"].float(), dim=-1).cpu().numpy()
            arrays["labels"].extend(batch["labels"].tolist())
            arrays["predictions"].extend(probabilities.argmax(axis=1).tolist())
            arrays["probabilities"].append(probabilities)
            arrays["embeddings"].append(outputs["embedding"].float().cpu().numpy())

    labels = np.asarray(arrays["labels"], dtype=np.int64)
    predictions = np.asarray(arrays["predictions"], dtype=np.int64)
    probabilities = np.concatenate(arrays["probabilities"]).astype(np.float32)
    embeddings = np.concatenate(arrays["embeddings"]).astype(np.float32)
    subjects = [str(value) for value in arrays["subjects"]]
    metrics = classification_metrics(labels, predictions, int(config["data"]["num_labels"]))
    metrics["subjects"] = subject_metrics(
        labels, predictions, subjects, int(config["data"]["num_labels"])
    )
    metrics.update(calibration_metrics(probabilities, labels))
    metrics.update(
        {
            "split": split,
            "weights": weights,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_state": payload.get("state", {}),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{split}_outputs.npz",
        sample_ids=np.asarray(arrays["sample_ids"]),
        subjects=np.asarray(subjects),
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        embeddings=embeddings,
        sequence_lengths=np.asarray(arrays["sequence_lengths"], dtype=np.int64),
        valid_receivers=np.asarray(arrays["valid_receivers"], dtype=np.int64),
        receiver_quality_mean=np.concatenate(arrays["receiver_quality_mean"]).astype(np.float32),
        environments=np.asarray(arrays["environments"]),
        positions=np.asarray(arrays["positions"]),
        orientations=np.asarray(arrays["orientations"]),
    )
    (output_dir / f"{split}_output_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-sample Signal-v2 predictions and embeddings")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.manifest:
        config["data"]["manifest"] = args.manifest
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    metrics = export_predictions(
        config,
        Path(args.checkpoint),
        Path(args.output_dir),
        args.split,
        args.weights,
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
