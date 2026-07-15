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
from .ema import ExponentialMovingAverage
from .losses import SignalV2Loss
from .metrics import classification_metrics, subject_metrics


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
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.time(), **payload}, ensure_ascii=False) + "\n")


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
        self.training = config["training"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        device_name = self.training.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        epochs = int(self.training.get("epochs", 120))
        self.max_steps = self.training.get("max_steps")
        self.total_steps = int(self.max_steps or epochs * len(train_loader))
        warmup_steps = int(round(float(self.training.get("warmup_ratio", 0.08)) * self.total_steps))

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(step, 1) / warmup_steps
            progress = (step - warmup_steps) / max(self.total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        precision = self.training.get("precision", "bf16")
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and self.device.type == "cuda")
        self.use_ema = bool(self.training.get("use_ema", False))
        self.ema = (
            ExponentialMovingAverage(self.model, float(self.training.get("ema_decay", 0.999)))
            if self.use_ema
            else None
        )
        loss_config = config["loss"]
        self.criterion = SignalV2Loss(
            label_smoothing=float(config["model"]["classifier"].get("label_smoothing", 0.02)),
            cross_entropy_weight=float(loss_config.get("cross_entropy_weight", 1.0)),
            contrastive_weight=float(loss_config.get("supervised_contrastive_weight", 0.1)),
            contrastive_temperature=float(loss_config.get("contrastive_temperature", 0.1)),
        )
        self.logger = JsonlLogger(self.output_dir / "metrics.jsonl")
        self.state = {
            "epoch": 0,
            "global_step": 0,
            "best_selection_score": -1.0,
            "best_macro_f1": -1.0,
            "best_worst_subject_macro_f1": -1.0,
            "bad_evaluations": 0,
        }

    def _build_optimizer(self) -> torch.optim.Optimizer:
        backbone_lr = float(self.training.get("backbone_lr", 4e-4))
        classifier_lr = float(self.training.get("classifier_lr", 8e-4))
        classifier_prefixes = ("pool.", "embedding.", "classifier.")
        backbone_parameters = []
        classifier_parameters = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            target = classifier_parameters if name.startswith(classifier_prefixes) else backbone_parameters
            target.append(parameter)
        return torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": backbone_lr},
                {"params": classifier_parameters, "lr": classifier_lr},
            ],
            weight_decay=float(self.training.get("weight_decay", 0.03)),
        )

    def _autocast(self):
        if self.device.type != "cuda" or self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _forward(self, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        outputs = self.model(
            rssi=batch["rssi"],
            doppler=batch["doppler"],
            differential_csi=batch["differential_csi"],
            time_mask=batch["time_mask"],
            receiver_mask=batch["receiver_mask"],
            receiver_quality=batch["receiver_quality"],
            position_ids=batch["position_ids"],
            frame_times_ms=batch["frame_times_ms"],
        )
        losses = self.criterion(
            outputs["logits"], outputs["embedding"], batch["labels"], batch["subject_ids"]
        )
        return outputs, losses

    @torch.no_grad()
    def evaluate(self, loader=None, split: str = "validation", use_ema: bool = True) -> dict[str, Any]:
        loader = loader or self.validation_loader
        self.model.eval()
        predictions = []
        targets = []
        subjects: list[str] = []
        losses = []
        parameter_context = (
            self.ema.average_parameters(self.model) if (use_ema and self.ema is not None) else nullcontext()
        )
        with parameter_context:
            for batch in loader:
                subjects.extend(batch["subjects"])
                batch = self._move(batch)
                with self._autocast():
                    outputs, batch_losses = self._forward(batch)
                predictions.append(outputs["logits"].argmax(dim=-1).cpu().numpy())
                targets.append(batch["labels"].cpu().numpy())
                losses.append(float(batch_losses["loss"].cpu()))
        all_targets = np.concatenate(targets)
        all_predictions = np.concatenate(predictions)
        metrics = classification_metrics(all_targets, all_predictions, self.model.num_labels)
        metrics["subjects"] = subject_metrics(
            all_targets, all_predictions, subjects, self.model.num_labels
        )
        worst_subject = min(value["macro_f1"] for value in metrics["subjects"].values())
        selection = self.training.get("selection", {})
        score = (
            float(selection.get("macro_f1_weight", 0.7)) * metrics["macro_f1"]
            + float(selection.get("worst_subject_macro_f1_weight", 0.3)) * worst_subject
        )
        metrics.update(
            {
                "worst_subject_macro_f1": float(worst_subject),
                "selection_score": float(score),
                "loss": float(np.mean(losses)),
                "split": split,
                "weights": "ema" if (use_ema and self.ema is not None) else "raw",
                "epoch": self.state["epoch"],
                "global_step": self.state["global_step"],
            }
        )
        self.logger.log({"event": "evaluation", **metrics})
        return metrics

    def _save(self, name: str) -> Path:
        return save_checkpoint(
            self.output_dir / "checkpoints" / name,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.ema,
            self.state,
            self.config,
        )

    def resume(self, path: str | Path) -> None:
        payload = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler, self.scaler, self.ema
        )
        self.state.update(payload["state"])
        self.logger.log({"event": "resume", "checkpoint": str(path), **self.state})

    def fit(self, resume_from: str | Path | None = None) -> dict[str, Any]:
        if resume_from:
            self.resume(resume_from)
        epochs = int(self.training.get("epochs", 120))
        log_every = int(self.training.get("log_every_steps", 10))
        eval_every = int(self.training.get("eval_every_steps", len(self.train_loader)))
        save_every = int(self.training.get("save_every_steps", len(self.train_loader)))
        patience = int(self.training.get("early_stopping_patience", 20))
        min_delta = float(self.training.get("early_stopping_min_delta", 5e-4))
        clip = float(self.training.get("gradient_clip", 1.0))
        self.optimizer.zero_grad(set_to_none=True)
        try:
            for epoch in range(self.state["epoch"], epochs):
                self.state["epoch"] = epoch
                if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                    self.train_loader.batch_sampler.set_epoch(epoch)
                if hasattr(self.train_loader.dataset, "set_epoch"):
                    self.train_loader.dataset.set_epoch(epoch)
                self.model.train()
                for batch in self.train_loader:
                    batch = self._move(batch)
                    with self._autocast():
                        _, losses = self._forward(batch)
                    if not torch.isfinite(losses["loss"]):
                        raise FloatingPointError(f"Non-finite loss at step {self.state['global_step']}")
                    self.scaler.scale(losses["loss"]).backward()
                    self.scaler.unscale_(self.optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    if self.ema is not None:
                        self.ema.update(self.model)
                    self.state["global_step"] += 1
                    step = self.state["global_step"]
                    debug_fail = self.training.get("debug_fail_after_step")
                    if debug_fail is not None and step >= int(debug_fail):
                        raise RuntimeError(f"Intentional smoke failure at step {step}")
                    if step % log_every == 0:
                        self.logger.log(
                            {
                                "event": "train",
                                "epoch": epoch,
                                "global_step": step,
                                "loss": float(losses["loss"].detach().cpu()),
                                "cross_entropy": float(losses["cross_entropy"].detach().cpu()),
                                "cross_subject_supcon": float(losses["cross_subject_supcon"].detach().cpu()),
                                "gradient_norm": float(gradient_norm),
                                "learning_rates": [group["lr"] for group in self.optimizer.param_groups],
                            }
                        )
                    if step % eval_every == 0:
                        metrics = self.evaluate()
                        self.model.train()
                        if metrics["selection_score"] > self.state["best_selection_score"] + min_delta:
                            self.state["best_selection_score"] = metrics["selection_score"]
                            self.state["best_macro_f1"] = metrics["macro_f1"]
                            self.state["best_worst_subject_macro_f1"] = metrics["worst_subject_macro_f1"]
                            self.state["bad_evaluations"] = 0
                            self._save("best.pt")
                        else:
                            self.state["bad_evaluations"] += 1
                        if patience and self.state["bad_evaluations"] >= patience:
                            self.logger.log({"event": "early_stopping", **self.state})
                            self._save("last.pt")
                            return self.state
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
                path = self._save(f"emergency_step_{self.state['global_step']:08d}.pt")
                self.logger.log({"event": "emergency_checkpoint", "path": str(path), **self.state})
            finally:
                raise
