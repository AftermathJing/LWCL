from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                if self.shadow[name].device != parameter.device:
                    self.shadow[name] = self.shadow[name].to(parameter.device)
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
        self.num_updates += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state.get("num_updates", 0))
        self.shadow = {name: value for name, value in state["shadow"].items()}

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        backup = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name].to(parameter.device))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in backup:
                        parameter.copy_(backup[name])
