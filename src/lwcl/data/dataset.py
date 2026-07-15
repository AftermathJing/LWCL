from __future__ import annotations

import csv
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .csi_bench import load_csi_bench_h5


class SignalDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        max_seq_len: int,
        training: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.max_seq_len = max_seq_len
        self.training = training
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row in rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"No rows for split={split} in {manifest_path}")
        self.labels = [int(row["label"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(row["feature_path"])
        if not path.is_absolute():
            path = self.root / path
        feature_format = row.get("feature_format", "").lower() or path.suffix.lower().lstrip(".")
        if feature_format in {"h5", "hdf5"}:
            num_subcarriers = int(float(row["num_sub"])) if row.get("num_sub") else None
            features = load_csi_bench_h5(
                path,
                data_key=row.get("data_key") or "CSI_amps",
                num_subcarriers=num_subcarriers,
                num_devices=int(row.get("device_count") or 1),
                target_time=int(row.get("target_time") or 500),
                target_receivers=int(row.get("target_receivers") or 1),
                target_features=int(row.get("target_features") or 232),
            )
        elif feature_format == "npz":
            with np.load(path, allow_pickle=False) as archive:
                features = archive["features"].astype(np.float32)
        else:
            raise ValueError(f"Unsupported feature format {feature_format!r} for {path}")
        if features.ndim != 3:
            raise ValueError(f"Expected [T,N,F] in {path}, got {features.shape}")
        if features.shape[0] > self.max_seq_len:
            if self.training:
                start = np.random.randint(0, features.shape[0] - self.max_seq_len + 1)
            else:
                start = (features.shape[0] - self.max_seq_len) // 2
            features = features[start : start + self.max_seq_len]
        return {
            "features": torch.from_numpy(features),
            "label": int(row["label"]),
            "sample_id": row["sample_id"],
            "metadata": row,
        }


def collate_signal_batch(batch: list[dict[str, Any]], max_seq_len: int) -> dict[str, Any]:
    max_length = min(max(item["features"].shape[0] for item in batch), max_seq_len)
    receivers = batch[0]["features"].shape[1]
    feature_dim = batch[0]["features"].shape[2]
    features = torch.zeros(len(batch), max_length, receivers, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    for index, item in enumerate(batch):
        length = min(item["features"].shape[0], max_length)
        features[index, :length] = item["features"][:length]
        mask[index, :length] = True
    return {
        "features": features,
        "attention_mask": mask,
        "position_ids": torch.arange(max_length).unsqueeze(0).repeat(len(batch), 1),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "sample_ids": [item["sample_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def build_weighted_sampler(dataset: SignalDataset) -> WeightedRandomSampler:
    counts = Counter(dataset.labels)
    weights = [1.0 / counts[label] for label in dataset.labels]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True)


def build_dataloader(
    dataset: SignalDataset,
    batch_size: int,
    num_workers: int,
    balanced_sampling: bool,
    shuffle: bool,
    pin_memory: bool = True,
) -> DataLoader:
    sampler = build_weighted_sampler(dataset) if balanced_sampling else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=partial(collate_signal_batch, max_seq_len=dataset.max_seq_len),
    )
