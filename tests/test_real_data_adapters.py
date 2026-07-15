import csv
import json

import h5py
import numpy as np

from lwcl.data.csi_bench import build_csi_bench_manifests, standardize_csi_bench_array
from lwcl.data.dataset import SignalDataset
from lwcl.data.preprocessing import complex_pca
from lwcl.data.widar3 import build_widar3_raw_manifest


def test_csi_bench_standardization_uses_frequency_first_layout():
    values = np.arange(12, dtype=np.float32).reshape(4, 3, 1)
    features = standardize_csi_bench_array(
        values,
        num_subcarriers=4,
        target_time=5,
        target_receivers=1,
        target_features=6,
    )
    assert features.shape == (5, 1, 6)
    assert np.allclose(features[3:], 0.0)
    assert np.allclose(features[:3, 0, 4:], 0.0)
    assert abs(float(features[:3, 0, :4].mean())) < 1e-6


def test_csi_bench_standardization_restores_device_axis():
    values = np.arange(24, dtype=np.float32).reshape(8, 3, 1)
    features = standardize_csi_bench_array(
        values,
        num_subcarriers=8,
        num_devices=2,
        target_time=5,
        target_receivers=2,
        target_features=4,
    )
    assert features.shape == (5, 2, 4)
    assert np.count_nonzero(features[:3]) == 24


def test_csi_bench_manifest_and_dataset_loading(tmp_path):
    root = tmp_path / "CSI-Bench"
    task = root / "FallDetection"
    (task / "metadata").mkdir(parents=True)
    (task / "splits").mkdir()
    sample_path = task / "samples" / "sample.h5"
    sample_path.parent.mkdir()
    with h5py.File(sample_path, "w") as handle:
        handle.create_dataset("CSI_amps", data=np.arange(12, dtype=np.float32).reshape(4, 3, 1))
    with (task / "metadata" / "sample_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "file_path", "label", "num_sub", "subject"])
        writer.writeheader()
        writer.writerow({"id": "s1", "file_path": "./samples/sample.h5", "label": "Fall", "num_sub": 4, "subject": "U01"})
    (task / "metadata" / "label_mapping.json").write_text(
        json.dumps({"label_to_idx": {"Fall": 0}, "idx_to_label": {"0": "Fall"}, "num_classes": 1}),
        encoding="utf-8",
    )
    (task / "splits" / "train_id.json").write_text(json.dumps(["s1"]), encoding="utf-8")
    (task / "splits" / "val_id.json").write_text(json.dumps([]), encoding="utf-8")
    (task / "splits" / "test_id.json").write_text(json.dumps([]), encoding="utf-8")

    output_root = tmp_path / "processed"
    summary = build_csi_bench_manifests(
        root,
        output_root,
        target_time=5,
        target_receivers=1,
        target_features=6,
    )
    assert summary["FallDetection"]["train"] == 1
    dataset = SignalDataset(output_root / "FallDetection" / "manifest.csv", "train", max_seq_len=5)
    item = dataset[0]
    assert tuple(item["features"].shape) == (5, 1, 6)
    assert item["label"] == 0


def test_widar3_manifest_groups_six_receivers_and_filters_paper_gestures(tmp_path):
    root = tmp_path / "widar3"
    user_dir = root / "20181130" / "user1"
    user_dir.mkdir(parents=True)
    for gesture in (1, 7):
        for receiver in range(1, 7):
            (user_dir / f"user1-{gesture}-2-3-4-r{receiver}.dat").write_bytes(b"")
    output_manifest = tmp_path / "manifest.csv"
    summary = build_widar3_raw_manifest(root, output_manifest)
    assert summary["samples"] == 1
    with output_manifest.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["label"] == "0"
    assert row["gesture_name"] == "push_pull"
    assert row["subject"] == "user1"


def test_complex_pca_pads_missing_components():
    values = np.ones((8, 2), dtype=np.complex64)
    projected = complex_pca(values, components=5)
    assert projected.shape == (8, 5)
    empty = complex_pca(np.zeros((8, 0), dtype=np.complex64), components=3)
    assert empty.shape == (8, 3)
