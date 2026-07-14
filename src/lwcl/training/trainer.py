from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .metrics import classification_metrics, grouped_accuracy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        payload = {"time": time.time(), **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        config: dict[str, Any],
        output_dir: str | Path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.config = config
        self.training_config = config["training"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        device_name = self.training_config.get("device", "auto")
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.max_steps = self.training_config.get("max_steps")
        epochs = int(self.training_config.get("epochs", 10))
        total_steps = int(self.max_steps or max(1, epochs * len(train_loader)))
        warmup_steps = int(self.training_config.get("warmup_steps", 0))

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(step, 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        precision = self.training_config.get("precision", "fp32")
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and self.device.type == "cuda")
        self.logger = JsonlLogger(self.output_dir / "metrics.jsonl")
        self.state = {"epoch": 0, "global_step": 0, "best_macro_f1": -1.0}

    def _build_optimizer(self) -> torch.optim.Optimizer:
        encoder_lr = float(self.training_config.get("learning_rate", 3e-4))
        backbone_lr = float(self.training_config.get("backbone_learning_rate", encoder_lr))
        weight_decay = float(self.training_config.get("weight_decay", 0.01))
        encoder_parameters: list[nn.Parameter] = []
        backbone_parameters: list[nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            (backbone_parameters if name.startswith("backbone.") else encoder_parameters).append(parameter)
        groups = []
        if encoder_parameters:
            groups.append({"params": encoder_parameters, "lr": encoder_lr})
        if backbone_parameters:
            groups.append({"params": backbone_parameters, "lr": backbone_lr})
        if not groups:
            raise ValueError("No trainable parameters were found")
        return torch.optim.AdamW(groups, weight_decay=weight_decay)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _autocast(self):
        if self.autocast_dtype is None or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader | None = None, split: str = "validation") -> dict[str, Any]:
        loader = loader or self.validation_loader
        self.model.eval()
        losses: list[float] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        metadata: list[dict[str, str]] = []
        for batch in loader:
            batch = self._move_batch(batch)
            with self._autocast():
                outputs = self.model(
                    features=batch["features"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    labels=batch["labels"],
                )
            if outputs["loss"] is not None:
                losses.append(float(outputs["loss"].detach().cpu()))
            predictions.append(outputs["logits"].argmax(dim=-1).cpu().numpy())
            targets.append(batch["labels"].cpu().numpy())
            metadata.extend(batch["metadata"])
        all_targets = np.concatenate(targets)
        all_predictions = np.concatenate(predictions)
        metrics = classification_metrics(all_targets, all_predictions, self.model.num_labels)
        metrics["group_accuracy"] = grouped_accuracy(all_targets, all_predictions, metadata)
        metrics["loss"] = float(np.mean(losses)) if losses else None
        metrics["split"] = split
        metrics["global_step"] = self.state["global_step"]
        self.logger.log({"event": "evaluation", **metrics})
        return metrics

    def _save(self, name: str) -> Path:
        return save_checkpoint(
            self.output_dir / "checkpoints" / name,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.state,
            self.config,
            trainable_only=bool(self.training_config.get("trainable_only_checkpoints", False)),
        )

    def resume(self, checkpoint_path: str | Path) -> None:
        payload = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
        )
        self.state.update(payload["state"])
        self.logger.log({"event": "resume", "checkpoint": str(checkpoint_path), **self.state})

    def fit(self, resume_from: str | Path | None = None) -> dict[str, Any]:
        if resume_from is not None:
            self.resume(resume_from)
        epochs = int(self.training_config.get("epochs", 10))
        log_every = int(self.training_config.get("log_every_steps", 10))
        eval_every = int(self.training_config.get("eval_every_steps", 100))
        save_every = int(self.training_config.get("save_every_steps", 100))
        gradient_clip = float(self.training_config.get("gradient_clip_norm", 1.0))
        accumulation = int(self.training_config.get("gradient_accumulation_steps", 1))
        self.optimizer.zero_grad(set_to_none=True)
        try:
            for epoch in range(self.state["epoch"], epochs):
                self.state["epoch"] = epoch
                self.model.train()
                for batch_index, batch in enumerate(self.train_loader):
                    batch = self._move_batch(batch)
                    with self._autocast():
                        outputs = self.model(
                            features=batch["features"],
                            attention_mask=batch["attention_mask"],
                            position_ids=batch["position_ids"],
                            labels=batch["labels"],
                        )
                        loss = outputs["loss"] / accumulation
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at step {self.state['global_step']}")
                    self.scaler.scale(loss).backward()
                    if (batch_index + 1) % accumulation:
                        continue
                    self.scaler.unscale_(self.optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.state["global_step"] += 1
                    step = self.state["global_step"]
                    debug_fail_step = self.training_config.get("debug_fail_after_step")
                    if debug_fail_step is not None and step >= int(debug_fail_step):
                        raise RuntimeError(f"Intentional smoke-test failure at step {step}")
                    if step % log_every == 0:
                        self.logger.log(
                            {
                                "event": "train",
                                "epoch": epoch,
                                "global_step": step,
                                "loss": float(loss.detach().cpu()) * accumulation,
                                "gradient_norm": float(gradient_norm),
                                "learning_rates": [group["lr"] for group in self.optimizer.param_groups],
                            }
                        )
                    if step % eval_every == 0:
                        metrics = self.evaluate()
                        self.model.train()
                        if metrics["macro_f1"] > self.state["best_macro_f1"]:
                            self.state["best_macro_f1"] = metrics["macro_f1"]
                            self._save("best.pt")
                    if step % save_every == 0:
                        self._save(f"step_{step:08d}.pt")
                        self._save("last.pt")
                    if self.max_steps is not None and step >= int(self.max_steps):
                        self._save("last.pt")
                        return self.state
                self.state["epoch"] = epoch + 1
            self._save("last.pt")
            return self.state
        except Exception:
            try:
                emergency = self._save(f"emergency_step_{self.state['global_step']:08d}.pt")
                self.logger.log({"event": "emergency_checkpoint", "path": str(emergency), **self.state})
            finally:
                raise
