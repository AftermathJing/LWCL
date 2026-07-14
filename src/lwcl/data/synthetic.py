from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def generate_synthetic_dataset(
    output_dir: str | Path,
    samples_per_label: int = 12,
    num_labels: int = 6,
    num_receivers: int = 6,
    feature_dim: int = 49,
    min_length: int = 48,
    max_length: int = 96,
    seed: int = 2025,
) -> Path:
    output_dir = Path(output_dir)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str | int]] = []
    for label in range(num_labels):
        for sample_index in range(samples_per_label):
            length = int(rng.integers(min_length, max_length + 1))
            time = np.linspace(0.0, 1.0, length, dtype=np.float32)
            features = rng.normal(0.0, 0.25, (length, num_receivers, feature_dim)).astype(np.float32)
            frequency = label + 1
            for receiver in range(num_receivers):
                phase = receiver * 0.25
                features[:, receiver, :8] += np.sin(2 * np.pi * frequency * time[:, None] + phase)
                features[:, receiver, 8:16] += np.cos(2 * np.pi * (frequency + 0.5) * time[:, None] - phase)
            subject = f"S{sample_index % 6:02d}"
            environment = f"R{sample_index % 3 + 1}"
            sample_id = f"g{label}_s{sample_index:03d}"
            path = feature_dir / f"{sample_id}.npz"
            np.savez_compressed(path, features=features)
            rows.append(
                {
                    "sample_id": sample_id,
                    "feature_path": path.relative_to(output_dir).as_posix(),
                    "label": label,
                    "subject": subject,
                    "environment": environment,
                    "position": f"P{sample_index % 5 + 1}",
                    "orientation": f"O{sample_index % 5 + 1}",
                }
            )
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest
