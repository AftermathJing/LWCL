import csv

import numpy as np

from lwcl.data.dataset import SignalDataset, build_dataloader
from lwcl.data.splits import create_group_splits, create_leave_one_domain_out_splits
from lwcl.data.synthetic import generate_synthetic_dataset


def test_synthetic_split_and_batch(tmp_path):
    manifest = generate_synthetic_dataset(tmp_path, samples_per_label=6, seed=7)
    split_manifest = tmp_path / "splits" / "manifest_split.csv"
    counts = create_group_splits(manifest, split_manifest, group_field="subject", seed=7)
    assert sum(counts.values()) == 36
    dataset = SignalDataset(split_manifest, "train", max_seq_len=64, training=True)
    loader = build_dataloader(dataset, batch_size=4, num_workers=0, balanced_sampling=True, shuffle=True)
    batch = next(iter(loader))
    assert batch["features"].shape[0] == 4
    assert batch["features"].shape[2:] == (6, 49)
    assert batch["attention_mask"].shape[:2] == batch["features"].shape[:2]


def test_leave_one_domain_out_keeps_target_exclusive_and_stratifies_validation(tmp_path):
    source = tmp_path / "processed" / "manifest.csv"
    source.parent.mkdir(parents=True)
    feature_dir = source.parent / "features"
    feature_dir.mkdir()
    rows = []
    for domain in ("room_a", "room_b"):
        for label in ("0", "1"):
            for index in range(4):
                feature_path = feature_dir / f"{domain}_{label}_{index}.npz"
                np.savez_compressed(feature_path, features=np.zeros((4, 2, 3), dtype=np.float32))
                rows.append(
                    {
                        "sample_id": feature_path.stem,
                        "feature_path": f"features/{feature_path.name}",
                        "feature_format": "npz",
                        "label": label,
                        "subject": f"{domain}_user",
                        "environment": domain,
                    }
                )
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "splits"
    summary = create_leave_one_domain_out_splits(source, output_dir, validation_ratio=0.25)
    assert set(summary) == {"room_a", "room_b"}
    for target in summary:
        with (output_dir / target / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
            fold_rows = list(csv.DictReader(handle))
        test_rows = [row for row in fold_rows if row["split"] == "test"]
        source_rows = [row for row in fold_rows if row["split"] != "test"]
        assert {row["environment"] for row in test_rows} == {target}
        assert target not in {row["environment"] for row in source_rows}
        assert {row["label"] for row in test_rows} == {"0", "1"}
        assert {row["label"] for row in fold_rows if row["split"] == "validation"} == {"0", "1"}
        assert all((output_dir / target / row["feature_path"]).resolve().exists() for row in fold_rows)
