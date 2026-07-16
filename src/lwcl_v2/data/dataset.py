from __future__ import annotations

import csv
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augment import SignalAugmenter
from .csi_bench import load_csi_bench_amplitude
from .sampler import CrossSubjectBatchSampler


class SignalV2Dataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        max_seq_len: int,
        augmentation: dict[str, Any] | None = None,
        seed: int = 2025,
        min_valid_receivers: int = 5,
        feature_mode: str = "current",
        phase_subcarriers: int = 30,
        amplitude_bins: int = 64,
        frame_interval_ms: float = 10.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.epoch = 0
        self.feature_mode = feature_mode
        self.phase_subcarriers = phase_subcarriers
        self.amplitude_bins = amplitude_bins
        self.frame_interval_ms = frame_interval_ms
        self.requires_phase = feature_mode in {"phase_dfs", "current_plus_phase"}
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row in rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"No rows for split={split} in {manifest_path}")
        self.labels = [int(row["label"]) for row in self.rows]
        self.subjects = [row["subject"] for row in self.rows]
        self.augmenter = SignalAugmenter(augmentation, min_valid_receivers) if augmentation else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, row: dict[str, str]) -> Path:
        path = Path(row["feature_path"])
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if self.feature_mode == "csi_bench_amplitude" or row.get("feature_format") == "h5":
            sample = load_csi_bench_amplitude(
                self._path(row),
                num_subcarriers=int(float(row.get("num_sub", 0) or 0)) or None,
                num_devices=int(row.get("device_count", 1) or 1),
                amplitude_bins=self.amplitude_bins,
                frame_interval_ms=self.frame_interval_ms,
            )
        else:
            with np.load(self._path(row), allow_pickle=False) as archive:
                if self.requires_phase and "csi_ratio_phase" not in archive.files:
                    raise ValueError(
                        f"feature_mode={self.feature_mode} requires csi_ratio_phase in {self._path(row)}"
                    )
                sample = {
                    "rssi": archive["rssi"].astype(np.float32),
                    "doppler": archive["doppler"].astype(np.float32),
                    "differential_csi": archive["differential_csi"].astype(np.float32),
                    "time_mask": archive["time_mask"].astype(bool),
                    "frame_times_ms": archive["frame_times_ms"].astype(np.float32),
                    "receiver_mask": archive["receiver_mask"].astype(bool),
                    "receiver_quality": archive["receiver_quality"].astype(np.float32),
                }
                if self.requires_phase:
                    sample["csi_ratio_phase"] = archive["csi_ratio_phase"].astype(np.float32)
        reference = sample.get("amplitude", sample.get("rssi"))
        length = reference.shape[0]
        if length > self.max_seq_len:
            if self.augmenter is not None:
                rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
                start = int(rng.integers(0, length - self.max_seq_len + 1))
            else:
                start = (length - self.max_seq_len) // 2
            for key in (
                "rssi",
                "amplitude",
                "doppler",
                "differential_csi",
                "csi_ratio_phase",
                "time_mask",
                "frame_times_ms",
            ):
                if key not in sample:
                    continue
                sample[key] = sample[key][start : start + self.max_seq_len]
        if self.augmenter is not None:
            rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
            sample = self.augmenter(sample, rng)
        sample.update(
            {
                "label": int(row["label"]),
                "subject": row["subject"],
                "sample_id": row["sample_id"],
                "metadata": row,
            }
        )
        return sample


def collate_signal_v2(batch: list[dict[str, Any]], max_seq_len: int) -> dict[str, Any]:
    amplitude_mode = "amplitude" in batch[0]
    reference_key = "amplitude" if amplitude_mode else "rssi"
    length = min(max(item[reference_key].shape[0] for item in batch), max_seq_len)
    batch_size = len(batch)
    receivers = batch[0][reference_key].shape[1]
    rssi = None if amplitude_mode else torch.zeros(batch_size, length, receivers, 4)
    doppler = None if amplitude_mode else torch.zeros(batch_size, length, receivers, batch[0]["doppler"].shape[-1])
    differential = None if amplitude_mode else torch.zeros(
        batch_size, length, receivers, batch[0]["differential_csi"].shape[-2], 2
    )
    amplitude = torch.zeros(batch_size, length, receivers, batch[0]["amplitude"].shape[-1]) if amplitude_mode else None
    has_phase = "csi_ratio_phase" in batch[0]
    phase = (
        torch.zeros(
            batch_size, length, receivers, batch[0]["csi_ratio_phase"].shape[-2], 2
        )
        if has_phase
        else None
    )
    time_mask = torch.zeros(batch_size, length, dtype=torch.bool)
    frame_times_ms = torch.zeros(batch_size, length, dtype=torch.float32)
    receiver_mask = torch.zeros(batch_size, receivers, dtype=torch.bool)
    receiver_quality = torch.zeros(batch_size, receivers, batch[0]["receiver_quality"].shape[-1])
    for batch_index, item in enumerate(batch):
        valid_length = min(item[reference_key].shape[0], length)
        if amplitude_mode:
            amplitude[batch_index, :valid_length] = torch.from_numpy(item["amplitude"][:valid_length])
        else:
            rssi[batch_index, :valid_length] = torch.from_numpy(item["rssi"][:valid_length])
            doppler[batch_index, :valid_length] = torch.from_numpy(item["doppler"][:valid_length])
            differential[batch_index, :valid_length] = torch.from_numpy(item["differential_csi"][:valid_length])
        if phase is not None:
            phase[batch_index, :valid_length] = torch.from_numpy(
                item["csi_ratio_phase"][:valid_length]
            )
        time_mask[batch_index, :valid_length] = torch.from_numpy(item["time_mask"][:valid_length])
        frame_times_ms[batch_index, :valid_length] = torch.from_numpy(item["frame_times_ms"][:valid_length])
        receiver_mask[batch_index] = torch.from_numpy(item["receiver_mask"])
        receiver_quality[batch_index] = torch.from_numpy(item["receiver_quality"])
    subjects = [item["subject"] for item in batch]
    subject_mapping = {subject: index for index, subject in enumerate(sorted(set(subjects)))}
    output = {
        "rssi": rssi,
        "doppler": doppler,
        "differential_csi": differential,
        "time_mask": time_mask,
        "frame_times_ms": frame_times_ms,
        "receiver_mask": receiver_mask,
        "receiver_quality": receiver_quality,
        "position_ids": torch.arange(length).unsqueeze(0).repeat(batch_size, 1),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "subject_ids": torch.tensor([subject_mapping[subject] for subject in subjects], dtype=torch.long),
        "subjects": subjects,
        "sample_ids": [item["sample_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
    if amplitude_mode:
        output["amplitude"] = amplitude
    if phase is not None:
        output["csi_ratio_phase"] = phase
    return output


def build_loaders(config: dict[str, Any], include_test: bool = False):
    data = config["data"]
    training = config["training"]
    manifest = data["manifest"]
    common = {
        "manifest_path": manifest,
        "max_seq_len": int(data["max_seq_len"]),
        "seed": int(training.get("seed", 2025)),
        "min_valid_receivers": int(data.get("min_valid_receivers", 5)),
        "feature_mode": str(data.get("feature_mode", "current")),
        "phase_subcarriers": int(data.get("phase_subcarriers", 30)),
        "amplitude_bins": int(data.get("amplitude_bins", 64)),
        "frame_interval_ms": float(data.get("frame_interval_ms", 10.0)),
    }
    train_dataset = SignalV2Dataset(
        split="train", augmentation=config.get("augmentation"), **common
    )
    validation_dataset = SignalV2Dataset(split="validation", **common)
    sampler_config = training.get("subject_balanced_sampler", {})
    batch_sampler = CrossSubjectBatchSampler(
        train_dataset.labels,
        train_dataset.subjects,
        gestures_per_batch=int(sampler_config.get("gestures_per_batch", data["num_labels"])),
        subjects_per_gesture=int(sampler_config.get("subjects_per_gesture", 4)),
        samples_per_subject=int(sampler_config.get("samples_per_subject", 4)),
        seed=int(training.get("seed", 2025)),
        batches_per_epoch=sampler_config.get("batches_per_epoch"),
    )
    collate = partial(collate_signal_v2, max_seq_len=int(data["max_seq_len"]))
    num_workers = int(training.get("num_workers", 4))
    persistent_workers = num_workers > 0 and bool(training.get("persistent_workers", False))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training.get("eval_batch_size", training.get("batch_size", 96))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        collate_fn=collate,
    )
    if not include_test:
        return train_loader, validation_loader
    test_dataset = SignalV2Dataset(split="test", **common)
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(training.get("eval_batch_size", training.get("batch_size", 96))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        collate_fn=collate,
    )
    return train_loader, validation_loader, test_loader
