from __future__ import annotations

import numpy as np
import torch

from lwcl_v2.data.augment import SignalAugmenter
from lwcl_v2.data.preprocessing import QualityAwareCSIProcessor
from lwcl_v2.data.quality import estimate_quality
from lwcl_v2.data.sampler import CrossSubjectBatchSampler
from lwcl_v2.training.losses import CrossSubjectSupervisedContrastiveLoss


def test_differential_csi_preserves_stream_and_subcarrier_axes():
    processor = QualityAwareCSIProcessor(
        {
            "data": {"num_receivers": 6, "min_valid_receivers": 5},
            "preprocessing": {
                "sample_rate": 1000,
                "filter_low_hz": 2.0,
                "filter_high_hz": 60.0,
                "stft_window": 251,
                "stft_hop": 50,
                "stft_overlap": 201,
                "stft_nfft": 256,
            },
            "quality_control": {},
        }
    )
    rng = np.random.default_rng(11)
    csi = (rng.normal(size=(64, 90)) + 1j * rng.normal(size=(64, 90))).astype(np.complex64)
    differential = processor._differential_csi(csi)
    assert differential.shape == (64, 60)
    assert np.isfinite(differential).all()


def test_quality_rejects_severely_short_receiver():
    rng = np.random.default_rng(3)
    csi = (rng.normal(size=(130, 90)) + 1j * rng.normal(size=(130, 90))).astype(np.complex64)
    timestamps = np.arange(130, dtype=np.int64) * 1000
    quality = estimate_quality(
        csi,
        timestamps,
        group_median_length=1190,
        sample_rate=1000,
        minimum_length=451,
        minimum_length_ratio=0.60,
        maximum_gap_ratio=0.10,
    )
    assert quality.length_ratio < 0.60
    assert not quality.valid


def test_receiver_dropout_never_drops_below_five():
    augmenter = SignalAugmenter(
        {"receiver_dropout": {"probability": 1.0, "max_receivers": 1}},
        min_valid_receivers=5,
    )
    sample = {
        "rssi": np.ones((10, 6, 4), np.float32),
        "doppler": np.ones((10, 6, 25), np.float32),
        "differential_csi": np.ones((10, 6, 10, 2), np.float32),
        "time_mask": np.ones(10, bool),
        "frame_times_ms": np.arange(10, dtype=np.float32) * 50,
        "receiver_mask": np.ones(6, bool),
        "receiver_quality": np.ones((6, 5), np.float32),
    }
    output = augmenter(sample, np.random.default_rng(4))
    assert int(output["receiver_mask"].sum()) == 5


def test_cross_subject_sampler_and_loss():
    labels = []
    subjects = []
    for label in range(6):
        for subject in range(4):
            for _ in range(4):
                labels.append(label)
                subjects.append(f"s{subject}")
    sampler = CrossSubjectBatchSampler(labels, subjects, 6, 4, 4, batches_per_epoch=1)
    batch = next(iter(sampler))
    assert len(batch) == 96
    for label in range(6):
        selected_subjects = {subjects[index] for index in batch if labels[index] == label}
        assert len(selected_subjects) == 4

    embeddings = torch.randn(8, 16, requires_grad=True)
    target = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    subject_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    loss = CrossSubjectSupervisedContrastiveLoss(0.1)(embeddings, target, subject_ids)
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
