from lwcl.data.dataset import SignalDataset, build_dataloader
from lwcl.data.splits import create_group_splits
from lwcl.data.synthetic import generate_synthetic_dataset


def test_synthetic_split_and_batch(tmp_path):
    manifest = generate_synthetic_dataset(tmp_path, samples_per_label=6, seed=7)
    split_manifest = tmp_path / "manifest_split.csv"
    counts = create_group_splits(manifest, split_manifest, group_field="subject", seed=7)
    assert sum(counts.values()) == 36
    dataset = SignalDataset(split_manifest, "train", max_seq_len=64, training=True)
    loader = build_dataloader(dataset, batch_size=4, num_workers=0, balanced_sampling=True, shuffle=True)
    batch = next(iter(loader))
    assert batch["features"].shape[0] == 4
    assert batch["features"].shape[2:] == (6, 49)
    assert batch["attention_mask"].shape[:2] == batch["features"].shape[:2]
