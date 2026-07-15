from __future__ import annotations

from pathlib import Path

import torch

from lwcl_v2.config import load_config
from lwcl_v2.models import SignalV2Classifier


ROOT = Path(__file__).resolve().parents[1]


def test_signal_v2_forward_backward_and_physical_positions():
    config = load_config(ROOT / "configs" / "signal_v2_base.yaml")
    model = SignalV2Classifier(config)
    batch, length, receivers = 3, 32, 6
    time_mask = torch.ones(batch, length, dtype=torch.bool)
    time_mask[0, -3:] = False
    receiver_mask = torch.ones(batch, receivers, dtype=torch.bool)
    receiver_mask[1, -1] = False
    outputs = model(
        rssi=torch.randn(batch, length, receivers, 4),
        doppler=torch.rand(batch, length, receivers, 25),
        differential_csi=torch.randn(batch, length, receivers, 10, 2),
        time_mask=time_mask,
        receiver_mask=receiver_mask,
        receiver_quality=torch.rand(batch, receivers, 5),
        frame_times_ms=125.5 + torch.arange(length).repeat(batch, 1) * 50.0,
    )
    assert outputs["logits"].shape == (batch, 6)
    assert outputs["embedding"].shape == (batch, 256)
    assert outputs["temporal_tokens"].shape[1] == 14
    assert outputs["receiver_attention"].shape == (batch, length, receivers)
    assert model.temporal_encoder.absolute_encoding is None
    assert torch.allclose(outputs["reduced_times_ms"][1, :3], torch.tensor([225.5, 325.5, 425.5]))
    outputs["logits"].sum().backward()
    assert model.classifier.weight.grad is not None


def test_multiscale_and_swiglu_variants_run():
    multiscale = load_config(ROOT / "configs" / "experiments" / "b9_multiscale.yaml")
    swiglu = load_config(ROOT / "configs" / "experiments" / "activation_c_swiglu.yaml")
    for config in (multiscale, swiglu):
        model = SignalV2Classifier(config)
        outputs = model(
            rssi=torch.randn(2, 24, 6, 4),
            doppler=torch.rand(2, 24, 6, 25),
            differential_csi=torch.randn(2, 24, 6, 10, 2),
            time_mask=torch.ones(2, 24, dtype=torch.bool),
            receiver_mask=torch.ones(2, 6, dtype=torch.bool),
            receiver_quality=torch.rand(2, 6, 5),
            frame_times_ms=125.5 + torch.arange(24).repeat(2, 1) * 50.0,
        )
        assert outputs["logits"].shape == (2, 6)
        assert torch.isfinite(outputs["logits"]).all()


def test_parameter_report_has_no_language_model_components():
    config = load_config(ROOT / "configs" / "signal_v2_base.yaml")
    report = SignalV2Classifier(config).parameter_report()
    assert 1_000_000 < report["total"] < 5_000_000
    assert all(name not in report["components"] for name in ("llm", "adapter", "lora", "qwen"))


def test_temporal_and_position_ablation_flags_resolve_to_model_components():
    no_difference = SignalV2Classifier(
        load_config(ROOT / "configs" / "ablations" / "a5_no_first_difference.yaml")
    )
    assert no_difference.temporal_stem.velocity_projection is None
    assert no_difference.temporal_stem.depthwise is not None

    no_convolution = SignalV2Classifier(
        load_config(ROOT / "configs" / "ablations" / "a5_no_temporal_convolution.yaml")
    )
    assert no_convolution.temporal_stem.velocity_projection is not None
    assert no_convolution.temporal_stem.depthwise is None

    no_position = SignalV2Classifier(
        load_config(ROOT / "configs" / "ablations" / "a6_no_transformer_position.yaml")
    )
    assert no_position.temporal_encoder.local_position_enabled is False
    assert no_position.temporal_encoder.global_position_enabled is False


def test_cuda_bf16_masked_forward_when_available():
    if not torch.cuda.is_available():
        return
    config = load_config(ROOT / "configs" / "signal_v2_base.yaml")
    model = SignalV2Classifier(config).cuda().train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            rssi=torch.randn(2, 16, 6, 4, device="cuda"),
            doppler=torch.rand(2, 16, 6, 25, device="cuda"),
            differential_csi=torch.randn(2, 16, 6, 10, 2, device="cuda"),
            time_mask=torch.tensor([[1] * 14 + [0, 0], [1] * 16], dtype=torch.bool, device="cuda"),
            receiver_mask=torch.tensor([[1, 1, 1, 1, 1, 0], [1] * 6], dtype=torch.bool, device="cuda"),
            receiver_quality=torch.rand(2, 6, 5, device="cuda"),
            frame_times_ms=125.5 + torch.arange(16, device="cuda").repeat(2, 1) * 50.0,
        )
        loss = outputs["logits"].float().sum()
    loss.backward()
    assert torch.isfinite(outputs["logits"]).all()
