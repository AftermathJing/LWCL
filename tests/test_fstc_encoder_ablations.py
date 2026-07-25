from __future__ import annotations

from pathlib import Path

import torch

from lwcl_v2.ablations.fstc_encoder import (
    FSTCEncoderAblationClassifier,
    FramewiseTemporalProjection,
    build_fstc_encoder_model,
)
from lwcl_v2.config import load_config
from lwcl_v2.models.signal_v2 import SignalV2Classifier


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "ablations" / "fstc_encoder"


def _inputs(batch: int = 2, length: int = 20, receivers: int = 6):
    time_mask = torch.ones(batch, length, dtype=torch.bool)
    time_mask[0, -2:] = False
    receiver_mask = torch.ones(batch, receivers, dtype=torch.bool)
    receiver_mask[1, -1] = False
    return {
        "rssi": torch.randn(batch, length, receivers, 4),
        "doppler": torch.rand(batch, length, receivers, 25),
        "differential_csi": torch.randn(batch, length, receivers, 10, 2),
        "time_mask": time_mask,
        "receiver_mask": receiver_mask,
        "receiver_quality": torch.rand(batch, receivers, 5),
        "position_ids": torch.arange(length).repeat(batch, 1),
        "frame_times_ms": 125.5 + torch.arange(length).repeat(batch, 1) * 50.0,
    }


def test_full_control_uses_original_fstc_class():
    config = load_config(CONFIG_ROOT / "full_control.yaml")
    model = build_fstc_encoder_model(config)
    assert type(model) is SignalV2Classifier
    assert model.ablation_variant == "full_control"


def test_three_isolated_ablations_forward_backward_and_parameter_reports():
    expected = {
        "no_feature_encoder": {"feature_encoder"},
        "no_spatial_encoder": {"receiver_encoder", "receiver_fusion"},
        "no_temporal_encoder": {"temporal_stem", "temporal_encoder"},
    }
    for variant, removed_components in expected.items():
        config = load_config(CONFIG_ROOT / f"{variant}.yaml")
        model = build_fstc_encoder_model(config)
        assert type(model) is FSTCEncoderAblationClassifier
        outputs = model(**_inputs())
        assert outputs["logits"].shape == (2, 6)
        assert outputs["embedding"].shape == (2, 256)
        assert torch.isfinite(outputs["logits"]).all()
        outputs["logits"].sum().backward()
        assert model.classifier.weight.grad is not None
        report = model.parameter_report()
        assert report["ablation"]["variant"] == variant
        assert report["total"] == report["trainable"]
        for component in removed_components:
            if component == "temporal_stem":
                assert model.temporal_stem is None
            elif component in {"receiver_encoder", "receiver_fusion"}:
                assert report["components"][component] == 0


def test_temporal_ablation_is_frame_permutation_equivariant_before_pooling():
    projection = FramewiseTemporalProjection(8, 12, dropout=0.0).eval()
    x = torch.randn(2, 9, 8)
    mask = torch.ones(2, 9, dtype=torch.bool)
    positions = torch.arange(9).repeat(2, 1)
    permutation = torch.tensor([7, 1, 4, 0, 8, 2, 6, 5, 3])
    encoded, _, _, _ = projection(x, positions, mask)
    permuted, _, _, _ = projection(x[:, permutation], positions[:, permutation], mask[:, permutation])
    assert torch.allclose(permuted, encoded[:, permutation], atol=1e-6)


def test_ablation_parameter_counts_drop_relative_to_control():
    control = build_fstc_encoder_model(load_config(CONFIG_ROOT / "full_control.yaml"))
    control_total = control.parameter_report()["total"]
    for variant in ("no_feature_encoder", "no_spatial_encoder", "no_temporal_encoder"):
        model = build_fstc_encoder_model(load_config(CONFIG_ROOT / f"{variant}.yaml"))
        assert model.parameter_report()["total"] < control_total
