from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from lwcl_v2.config import load_config
from lwcl_v2.training.ema import ExponentialMovingAverage


ROOT = Path(__file__).resolve().parents[1]


def test_ema_zero_decay_matches_online_parameters_exactly():
    model = nn.Linear(3, 2, bias=False)
    ema = ExponentialMovingAverage(model, decay=0.0)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    assert ema.num_updates == 1
    assert torch.allclose(ema.shadow["weight"], model.weight)


def test_ema_positive_decay_is_between_previous_shadow_and_current_model():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ExponentialMovingAverage(model, decay=0.5)
    previous = ema.shadow["weight"].clone()
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    expected = previous * 0.5 + model.weight * 0.5
    assert ema.num_updates == 1
    assert torch.allclose(ema.shadow["weight"], expected)


def test_signal_v2_defaults_disable_ema_but_keep_decay_configured():
    config = load_config(ROOT / "configs" / "signal_v2_base.yaml")
    training = config["training"]
    assert training["use_ema"] is False
    assert float(training["ema_decay"]) == 0.999
