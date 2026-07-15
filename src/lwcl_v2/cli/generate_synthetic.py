from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contract-correct synthetic Signal-v2 smoke data")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples-per-split", type=int, default=96)
    args = parser.parse_args()
    root = Path(args.output_root)
    features = root / "features"
    features.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2025)
    rows = []
    split_subjects = {
        "train": ["train1", "train2", "train3", "train4"],
        "validation": ["valid1", "valid2"],
        "test": ["test1", "test2"],
    }
    for split, subjects in split_subjects.items():
        for index in range(args.samples_per_split):
            label = index % 6
            subject = subjects[(index // 6) % len(subjects)]
            length = int(rng.integers(28, 41))
            sample_id = f"{split}_{subject}_g{label}_{index:04d}"
            receiver_mask = np.ones(6, dtype=bool)
            if index % 7 == 0:
                receiver_mask[-1] = False
            base = np.sin(np.linspace(0, np.pi * (label + 1), length, dtype=np.float32))[:, None, None]
            rssi = base + rng.normal(0, 0.1, (length, 6, 4)).astype(np.float32)
            doppler = np.abs(base + rng.normal(0, 0.1, (length, 6, 25))).astype(np.float32)
            doppler /= np.maximum(doppler.sum(axis=-1, keepdims=True), 1e-6)
            differential = (base[..., None] + rng.normal(0, 0.1, (length, 6, 10, 2))).astype(np.float32)
            for key in (rssi, doppler, differential):
                key[:, ~receiver_mask] = 0
            path = features / f"{sample_id}.npz"
            np.savez_compressed(
                path,
                rssi=rssi,
                doppler=doppler,
                differential_csi=differential,
                time_mask=np.ones(length, dtype=bool),
                frame_times_ms=(125.5 + np.arange(length) * 50.0).astype(np.float32),
                receiver_mask=receiver_mask,
                receiver_quality=np.tile(np.asarray([1.0, 0.98, 0.99, 0.2, 1.0], np.float32), (6, 1)),
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "feature_path": str(Path("features") / path.name),
                    "label": label,
                    "gesture_name": f"gesture_{label}",
                    "subject": subject,
                    "split": split,
                }
            )
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(root / "manifest.csv")


if __name__ == "__main__":
    main()
