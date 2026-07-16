from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from lwcl_v2.cli.audit_feature_contract import build_contract_audit
from lwcl_v2.cli.audit_tuning_configs import audit_configs
from lwcl_v2.config import load_config
from lwcl_v2.data.dataset import SignalV2Dataset, collate_signal_v2
from lwcl_v2.data.preprocessing import QualityAwareCSIProcessor
from lwcl_v2.models import SignalV2Classifier


ROOT = Path(__file__).resolve().parents[1]


def _processor_config(
    *, doppler_bins: int = 25, nfft: int = 256, include_phase: bool = False
) -> dict:
    return {
        "data": {"num_receivers": 6, "min_valid_receivers": 5},
        "preprocessing": {
            "sample_rate": 1000,
            "filter_low_hz": 2.0,
            "filter_high_hz": 60.0,
            "stft_window": 251,
            "stft_hop": 50,
            "stft_overlap": 201,
            "stft_nfft": nfft,
            "doppler_min_hz": -60,
            "doppler_max_hz": 60,
            "doppler_bins": doppler_bins,
            "doppler_projection": "interpolate" if doppler_bins > 25 else "max_pool",
            "doppler_pca_components": 3,
            "differential_pca_components": 10,
            "include_csi_ratio_phase": include_phase,
            "phase_subcarriers": 30,
        },
        "quality_control": {},
    }


def _feature_arrays(length: int = 12, bins: int = 25, phase: bool = False) -> dict:
    arrays = {
        "rssi": np.zeros((length, 6, 4), np.float32),
        "doppler": np.full((length, 6, bins), 1.0 / bins, np.float32),
        "differential_csi": np.zeros((length, 6, 10, 2), np.float32),
        "time_mask": np.ones(length, bool),
        "frame_times_ms": 125.5 + np.arange(length, dtype=np.float32) * 50,
        "receiver_mask": np.ones(6, bool),
        "receiver_quality": np.ones((6, 5), np.float32),
    }
    if phase:
        values = np.zeros((length, 6, 30, 2), np.float32)
        values[..., 1] = 1.0
        arrays["csi_ratio_phase"] = values
        arrays["csi_ratio_pair"] = np.tile(np.asarray([0, 2], np.int16), (6, 1))
    return arrays


def _write_manifest(path: Path, feature_path: Path, split: str = "train") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "feature_path", "subject", "label", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample1",
                "feature_path": str(feature_path),
                "subject": "user1",
                "label": "0",
                "split": split,
            }
        )


def test_csi_ratio_phase_pair_is_finite_and_unit_norm():
    processor = QualityAwareCSIProcessor(_processor_config(include_phase=True))
    rng = np.random.default_rng(41)
    amplitude = np.stack(
        (
            np.full((512, 30), 2.0),
            rng.uniform(0.1, 3.0, size=(512, 30)),
            rng.uniform(0.8, 1.2, size=(512, 30)),
        ),
        axis=1,
    )
    phase = rng.uniform(-np.pi, np.pi, size=amplitude.shape)
    csi = (amplitude * np.exp(1j * phase)).reshape(512, -1).astype(np.complex64)
    pairs, stream_pair = processor._csi_ratio_phase(csi)
    assert pairs.shape == (512, 30, 2)
    assert stream_pair.shape == (2,)
    assert stream_pair[0] != stream_pair[1]
    assert np.isfinite(pairs).all()
    assert np.allclose(np.linalg.norm(pairs, axis=-1), 1.0, atol=1e-5)


@pytest.mark.parametrize(("bins", "nfft"), ((61, 512), (121, 1024)))
def test_high_resolution_doppler_has_requested_shape_and_normalization(bins: int, nfft: int):
    processor = QualityAwareCSIProcessor(_processor_config(doppler_bins=bins, nfft=nfft))
    rng = np.random.default_rng(7 + bins)
    csi = (rng.normal(size=(600, 90)) + 1j * rng.normal(size=(600, 90))).astype(
        np.complex64
    )
    differential = processor._differential_csi(csi)
    doppler, times = processor._doppler(differential)
    assert doppler.shape == (times.size, bins)
    assert np.isfinite(doppler).all()
    assert np.allclose(doppler.sum(axis=-1), 1.0, atol=1e-5)


def test_old_npz_remains_compatible_and_phase_mode_is_strict(tmp_path: Path):
    feature_path = tmp_path / "sample.npz"
    np.savez_compressed(feature_path, **_feature_arrays())
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, feature_path)
    current = SignalV2Dataset(manifest, "train", 96, feature_mode="current")
    item = current[0]
    assert "csi_ratio_phase" not in item
    batch = collate_signal_v2([item], 96)
    assert "csi_ratio_phase" not in batch
    phase = SignalV2Dataset(manifest, "train", 96, feature_mode="phase_dfs")
    with pytest.raises(ValueError, match="requires csi_ratio_phase"):
        _ = phase[0]


def test_feature_contract_audit_is_deterministic(tmp_path: Path):
    feature_path = tmp_path / "sample.npz"
    np.savez_compressed(feature_path, **_feature_arrays(phase=True))
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, feature_path)
    config = tmp_path / "phase.yaml"
    config.write_text(
        """
data:
  feature_mode: current_plus_phase
  num_receivers: 6
  doppler_bins: 25
  differential_components: 10
  phase_subcarriers: 30
""".strip(),
        encoding="utf-8",
    )
    first = build_contract_audit(manifest, config)
    second = build_contract_audit(manifest, config)
    assert first["failed_samples"] == 0
    assert first["nonfinite"] == second["nonfinite"]
    assert first["contract_sha256"] == second["contract_sha256"]
    assert first["valid_receivers"]["valid_rate"] == 1.0


@pytest.mark.parametrize(
    ("feature_mode", "bins"),
    (("phase_dfs", 25), ("current_plus_phase", 25), ("current", 61), ("current", 121)),
)
def test_tuning_feature_modes_forward_backward_under_four_million_parameters(
    feature_mode: str, bins: int
):
    config = copy.deepcopy(load_config(ROOT / "configs" / "signal_v2_base.yaml"))
    config["data"]["feature_mode"] = feature_mode
    config["data"]["doppler_bins"] = bins
    config["data"]["phase_subcarriers"] = 30
    config["model"]["feature_stems"]["csi_ratio_phase_dim"] = 48
    model = SignalV2Classifier(config)
    outputs = model(
        rssi=torch.randn(2, 24, 6, 4),
        doppler=torch.rand(2, 24, 6, bins),
        differential_csi=torch.randn(2, 24, 6, 10, 2),
        csi_ratio_phase=torch.randn(2, 24, 6, 30, 2),
        time_mask=torch.ones(2, 24, dtype=torch.bool),
        receiver_mask=torch.ones(2, 6, dtype=torch.bool),
        receiver_quality=torch.rand(2, 6, 5),
        frame_times_ms=125.5 + torch.arange(24).repeat(2, 1) * 50.0,
    )
    assert outputs["logits"].shape == (2, 6)
    assert model.parameter_report()["total"] < 4_000_000
    outputs["logits"].sum().backward()
    assert model.classifier.weight.grad is not None


def test_frozen_tuning_matrix_has_nineteen_unique_validation_only_candidates():
    names = [
        "signal_v2_tuning_base.yaml",
        "signal_v2_hp_lr_low.yaml",
        "signal_v2_hp_lr_high.yaml",
        "signal_v2_hp_wd_low.yaml",
        "signal_v2_hp_wd_high.yaml",
        "signal_v2_hp_regularization_low.yaml",
        "signal_v2_hp_regularization_high.yaml",
        "signal_v2_hp_supcon_low.yaml",
        "signal_v2_hp_supcon_high.yaml",
        "signal_v2_hp_temperature_low.yaml",
        "signal_v2_hp_temperature_high.yaml",
        "signal_v2_structure_window3_stride2.yaml",
        "signal_v2_structure_attention_pool.yaml",
        "signal_v2_structure_local_mean.yaml",
        "signal_v2_structure_multiscale.yaml",
        "signal_v2_phase_dfs.yaml",
        "signal_v2_current_plus_phase.yaml",
        "signal_v2_highres_dfs61.yaml",
        "signal_v2_highres_dfs121.yaml",
    ]
    report = audit_configs([ROOT / "configs" / "tuning" / name for name in names])
    assert report["count"] == 19
    assert report["errors"] == []
