from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def _model_state(model: nn.Module, trainable_only: bool) -> tuple[dict[str, torch.Tensor], list[str]]:
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    state = model.state_dict()
    if trainable_only:
        state = {name: value.detach().cpu() for name, value in state.items() if name in trainable_names}
    else:
        state = {name: value.detach().cpu() for name, value in state.items()}
    return state, trainable_names


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    state: dict[str, Any],
    config: dict[str, Any],
    trainable_only: bool,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state, trainable_names = _model_state(model, trainable_only)
    payload = {
        "model": model_state,
        "model_trainable_only": trainable_only,
        "trainable_parameter_names": trainable_names,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "state": state,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "config_hash": config.get("_config_hash"),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=not payload["model_trainable_only"])
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    if payload["model_trainable_only"]:
        allowed_missing = set(model.state_dict()) - set(payload["model"])
        if set(missing) - allowed_missing:
            raise RuntimeError(f"Unexpected missing checkpoint keys: {missing}")
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("rng"):
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
        if torch.cuda.is_available() and payload["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return payload
