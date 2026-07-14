from __future__ import annotations

from pathlib import Path
from typing import Any

from lwcl.data.dataset import SignalDataset, build_dataloader


def build_loaders(config: dict[str, Any], include_test: bool = False):
    data = config["data"]
    training = config["training"]
    manifest = Path(data["manifest"])
    train_dataset = SignalDataset(manifest, "train", int(data["max_seq_len"]), training=True)
    validation_dataset = SignalDataset(manifest, "validation", int(data["max_seq_len"]), training=False)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(training.get("batch_size", 8)),
        num_workers=int(training.get("num_workers", 4)),
        balanced_sampling=bool(training.get("balanced_sampling", True)),
        shuffle=True,
    )
    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=int(training.get("eval_batch_size", training.get("batch_size", 8))),
        num_workers=int(training.get("num_workers", 4)),
        balanced_sampling=False,
        shuffle=False,
    )
    if not include_test:
        return train_loader, validation_loader
    test_dataset = SignalDataset(manifest, "test", int(data["max_seq_len"]), training=False)
    test_loader = build_dataloader(
        test_dataset,
        batch_size=int(training.get("eval_batch_size", training.get("batch_size", 8))),
        num_workers=int(training.get("num_workers", 4)),
        balanced_sampling=False,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader
