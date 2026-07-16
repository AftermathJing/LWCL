from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


class WiCBRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        image_size: int = 224,
        phase_field: str = "phase_path",
        dfs_field: str = "dfs_path",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            all_rows = list(csv.DictReader(handle))
        self.rows = [row for row in all_rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"No rows for split={split} in {self.manifest_path}")
        self.phase_field = phase_field
        self.dfs_field = dfs_field
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        phase = Image.open(self.root / row[self.phase_field]).convert("RGB")
        dfs = Image.open(self.root / row[self.dfs_field]).convert("RGB")
        metadata = {
            "sample_id": row.get("sample_id", ""),
            "subject": row.get("subject", ""),
            "environment": row.get("environment", ""),
            "position": row.get("position", ""),
            "orientation": row.get("orientation", ""),
        }
        return {
            "phase_inputs": self.transform(phase),
            "dfs_inputs": self.transform(dfs),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
            "metadata": metadata,
        }


def build_wicbr_dataloader(
    dataset: WiCBRDataset,
    *,
    batch_size: int,
    num_workers: int,
    balanced_sampling: bool,
    shuffle: bool,
) -> DataLoader:
    sampler = None
    if balanced_sampling:
        counts: dict[int, int] = {}
        for row in dataset.rows:
            label = int(row["label"])
            counts[label] = counts.get(label, 0) + 1
        weights = [1.0 / counts[int(row["label"])] for row in dataset.rows]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
