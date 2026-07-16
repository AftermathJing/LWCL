from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lwcl_v2.config import load_config
from lwcl_v2.data.csi_bench import _orient_frequency_time, resample_frequency
from lwcl_v2.data.dataset import collate_signal_v2
from lwcl_v2.models import SignalV2Classifier


ROOT = Path(__file__).resolve().parents[1]


def test_csi_bench_orientation_and_frequency_resampling():
    source = np.arange(56 * 500, dtype=np.float32).reshape(56, 500)
    oriented = _orient_frequency_time(source[..., None], 56)
    assert oriented.shape == (500, 56)
    tensor = oriented[:, None, :]
    resized = resample_frequency(tensor, 64)
    assert resized.shape == (500, 1, 64)
    assert np.isfinite(resized).all()


def test_csi_bench_collate_and_t00_forward():
    samples = []
    for index in range(2):
        samples.append(
            {
                "amplitude": np.random.default_rng(index).normal(size=(32, 1, 64)).astype(np.float32),
                "time_mask": np.ones(32, dtype=bool),
                "frame_times_ms": np.arange(32, dtype=np.float32) * 10,
                "receiver_mask": np.ones(1, dtype=bool),
                "receiver_quality": np.ones((1, 5), dtype=np.float32),
                "label": index,
                "subject": f"U0{index + 1}",
                "sample_id": f"s{index}",
                "metadata": {},
            }
        )
    batch = collate_signal_v2(samples, max_seq_len=500)
    assert batch["amplitude"].shape == (2, 32, 1, 64)
    assert batch["rssi"] is None
    config = load_config(ROOT / "configs" / "csi_bench" / "t00_har_smoke.yaml")
    model = SignalV2Classifier(config)
    outputs = model(
        rssi=batch["rssi"],
        doppler=batch["doppler"],
        differential_csi=batch["differential_csi"],
        amplitude=batch["amplitude"],
        time_mask=batch["time_mask"],
        receiver_mask=batch["receiver_mask"],
        receiver_quality=batch["receiver_quality"],
        position_ids=batch["position_ids"],
        frame_times_ms=batch["frame_times_ms"],
    )
    assert outputs["logits"].shape == (2, 5)
    assert outputs["embedding"].shape == (2, 256)
    outputs["logits"].sum().backward()
    assert model.classifier.weight.grad is not None


def test_har_protocol_manifest_builder_rejects_overlap(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "dataset"
    task = root / "Multitask" / "HumanActivityRecognition"
    (task / "metadata").mkdir(parents=True)
    (task / "splits").mkdir(parents=True)
    (task / "sub_Human_h5").mkdir()
    (task / "metadata" / "label_mapping.json").write_text(
        json.dumps({"label_to_idx": {"jumping": 0}}), encoding="utf-8"
    )
    feature = task / "sub_Human_h5" / "sample.h5"
    with h5py.File(feature, "w") as handle:
        handle["CSI_amps"] = np.zeros((56, 8, 1), dtype=np.float32)
    (task / "metadata" / "sample_metadata.csv").write_text(
        "id,file_path,user,label,num_sub\n"
        "s1,../../sub_Human_h5/sample.h5,U01,jumping,56\n",
        encoding="utf-8",
    )
    for name, values in {
        "train_id": ["s1"],
        "val_id": [],
        "test_id": ["s1"],
    }.items():
        (task / "splits" / f"{name}.json").write_text(json.dumps(values), encoding="utf-8")
    from lwcl_v2.data.csi_bench import build_har_protocol_manifests

    with pytest.raises(ValueError, match="overlap"):
        build_har_protocol_manifests(root, tmp_path / "out", protocols=["test_id"])
