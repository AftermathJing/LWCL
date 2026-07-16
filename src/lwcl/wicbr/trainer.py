from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from lwcl.training.checkpoint import load_checkpoint, save_checkpoint
from lwcl.training.metrics import classification_metrics, grouped_accuracy
from lwcl.training.trainer import JsonlLogger
from lwcl.wicbr.model import ProxyContrastiveLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WiCBRTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader | None,
        config: dict[str, Any],
        output_dir: str | Path,
        *,
        selection_loader: torch.utils.data.DataLoader | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.selection_loader = selection_loader or validation_loader
        self.config = config
        self.data_config = config["data"]
        self.training_config = config["training"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        device_name = self.training_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.training_config.get("learning_rate", 1e-4)),
            weight_decay=float(self.training_config.get("weight_decay", 0.0)),
        )
        scheduler_cfg = self.training_config.get("step_lr", {})
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(scheduler_cfg.get("step_size", 3)),
            gamma=float(scheduler_cfg.get("gamma", 0.5)),
        )
        precision = self.training_config.get("precision", "fp32")
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and self.device.type == "cuda")
        self.classification_loss = nn.CrossEntropyLoss()
        self.proxy_loss = ProxyContrastiveLoss(float(self.training_config.get("temperature", 0.1)))
        self.beta_1 = float(self.training_config.get("beta_1", 0.1))
        self.logger = JsonlLogger(self.output_dir / "metrics.jsonl")
        self.monitor_metric = str(self.training_config.get("monitor_metric", "macro_f1"))
        self.selection_split = str(self.training_config.get("selection_split", "validation"))
        self.state = {
            "epoch": 0,
            "global_step": 0,
            "best_macro_f1": -1.0,
            "best_metric": -1.0,
            "best_metric_name": self.monitor_metric,
            "selection_split": self.selection_split,
            "bad_evaluations": 0,
        }
        self.max_steps = self.training_config.get("max_steps")

    def _autocast(self):
        if self.autocast_dtype is None or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _metadata_rows(self, metadata: Any) -> list[dict[str, str]]:
        if isinstance(metadata, list):
            return metadata
        if isinstance(metadata, dict):
            keys = list(metadata.keys())
            if not keys:
                return []
            length = len(metadata[keys[0]])
            rows: list[dict[str, str]] = []
            for index in range(length):
                row = {key: str(metadata[key][index]) for key in keys}
                rows.append(row)
            return rows
        return []

    def _compute_loss(self, outputs: dict[str, torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        ce = self.classification_loss(outputs["logits"], labels)
        contrastive = self.proxy_loss(outputs["embedding"], labels, self.model.fc)
        total = ce + self.beta_1 * contrastive
        return total, {
            "ce_loss": float(ce.detach().cpu()),
            "contrastive_loss": float(contrastive.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }

    def _save(self, name: str) -> Path:
        return save_checkpoint(
            self.output_dir / "checkpoints" / name,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.state,
            self.config,
            trainable_only=False,
        )

    def resume(self, checkpoint_path: str | Path) -> None:
        payload = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler, self.scaler)
        self.state.update(payload["state"])
        self.logger.log({"event": "resume", "checkpoint": str(checkpoint_path), **self.state})

    @torch.no_grad()
    def evaluate(
        self,
        loader: torch.utils.data.DataLoader | None = None,
        *,
        split: str = "validation",
    ) -> dict[str, Any]:
        loader = loader or self.validation_loader
        if loader is None:
            raise ValueError(f"No loader available for split={split}")
        self.model.eval()
        losses: list[float] = []
        ce_losses: list[float] = []
        contrastive_losses: list[float] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        metadata: list[dict[str, str]] = []
        for batch in loader:
            batch = self._move_batch(batch)
            with self._autocast():
                outputs = self.model(batch["phase_inputs"], batch["dfs_inputs"])
                loss, parts = self._compute_loss(outputs, batch["labels"])
            losses.append(float(loss.detach().cpu()))
            ce_losses.append(parts["ce_loss"])
            contrastive_losses.append(parts["contrastive_loss"])
            predictions.append(outputs["logits"].argmax(dim=-1).cpu().numpy())
            targets.append(batch["labels"].cpu().numpy())
            metadata.extend(self._metadata_rows(batch["metadata"]))
        all_targets = np.concatenate(targets)
        all_predictions = np.concatenate(predictions)
        metrics = classification_metrics(all_targets, all_predictions, int(self.data_config["num_labels"]))
        metrics["group_accuracy"] = grouped_accuracy(all_targets, all_predictions, metadata)
        metrics["loss"] = float(np.mean(losses))
        metrics["ce_loss"] = float(np.mean(ce_losses))
        metrics["contrastive_loss"] = float(np.mean(contrastive_losses))
        metrics["split"] = split
        metrics["global_step"] = self.state["global_step"]
        self.logger.log({"event": "evaluation", **metrics})
        return metrics

    def _record_best(self, metrics: dict[str, Any], min_delta: float) -> bool:
        monitor_value = float(metrics[self.monitor_metric])
        if float(metrics.get("macro_f1", -1.0)) > float(self.state.get("best_macro_f1", -1.0)):
            self.state["best_macro_f1"] = float(metrics["macro_f1"])
        if monitor_value > float(self.state.get("best_metric", -1.0)) + min_delta:
            self.state["best_metric"] = monitor_value
            self.state["bad_evaluations"] = 0
            self._save("best.pt")
            return True
        self.state["bad_evaluations"] = self.state.get("bad_evaluations", 0) + 1
        return False

    def _selection_evaluate(self, *, min_delta: float) -> bool:
        if self.selection_loader is None:
            raise ValueError(f"selection_split={self.selection_split!r} requested but no selection loader was provided")
        metrics = self.evaluate(self.selection_loader, split=self.selection_split)
        self.model.train()
        return self._record_best(metrics, min_delta)

    def fit(self, resume_from: str | Path | None = None) -> dict[str, Any]:
        if resume_from is not None:
            self.resume(resume_from)
        epochs = int(self.training_config.get("epochs", 30))
        log_every = int(self.training_config.get("log_every_steps", 10))
        eval_every_raw = self.training_config.get("eval_every_steps", 100)
        eval_every = int(eval_every_raw) if eval_every_raw not in (None, 0, "0") else 0
        eval_every_epochs_raw = self.training_config.get("eval_every_epochs")
        eval_every_epochs = int(eval_every_epochs_raw) if eval_every_epochs_raw not in (None, 0, "0") else 0
        save_every_raw = self.training_config.get("save_every_steps", 100)
        save_every = int(save_every_raw) if save_every_raw not in (None, 0, "0") else 0
        save_every_epochs_raw = self.training_config.get("save_every_epochs")
        save_every_epochs = int(save_every_epochs_raw) if save_every_epochs_raw not in (None, 0, "0") else 0
        gradient_clip = float(self.training_config.get("gradient_clip_norm", 1.0))
        early_stopping_patience = int(self.training_config.get("early_stopping_patience", 0))
        early_stopping_min_delta = float(self.training_config.get("early_stopping_min_delta", 0.0))
        self.optimizer.zero_grad(set_to_none=True)
        try:
            for epoch in range(self.state["epoch"], epochs):
                self.state["epoch"] = epoch
                self.model.train()
                hit_max_steps = False
                for batch in self.train_loader:
                    batch = self._move_batch(batch)
                    with self._autocast():
                        outputs = self.model(batch["phase_inputs"], batch["dfs_inputs"])
                        loss, parts = self._compute_loss(outputs, batch["labels"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at step {self.state['global_step']}")
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state["global_step"] += 1
                    step = self.state["global_step"]
                    if step % log_every == 0:
                        self.logger.log(
                            {
                                "event": "train",
                                "epoch": epoch,
                                "global_step": step,
                                "loss": parts["total_loss"],
                                "ce_loss": parts["ce_loss"],
                                "contrastive_loss": parts["contrastive_loss"],
                                "gradient_norm": float(gradient_norm),
                                "learning_rates": [group["lr"] for group in self.optimizer.param_groups],
                            }
                        )
                    if eval_every and step % eval_every == 0:
                        self._selection_evaluate(min_delta=early_stopping_min_delta)
                        if early_stopping_patience and self.state["bad_evaluations"] >= early_stopping_patience:
                            self.logger.log({"event": "early_stopping", **self.state})
                            self._save("last.pt")
                            return self.state
                    if save_every and step % save_every == 0:
                        self._save(f"step_{step:08d}.pt")
                        self._save("last.pt")
                    debug_fail_step = self.training_config.get("debug_fail_after_step")
                    if debug_fail_step is not None and step >= int(debug_fail_step):
                        raise RuntimeError(f"Intentional smoke-test failure at step {step}")
                    if self.max_steps is not None and step >= int(self.max_steps):
                        hit_max_steps = True
                        break
                if eval_every_epochs and (epoch + 1) % eval_every_epochs == 0:
                    self._selection_evaluate(min_delta=early_stopping_min_delta)
                    if early_stopping_patience and self.state["bad_evaluations"] >= early_stopping_patience:
                        self.logger.log({"event": "early_stopping", **self.state})
                        self._save("last.pt")
                        return self.state
                if save_every_epochs and (epoch + 1) % save_every_epochs == 0:
                    self._save("last.pt")
                if hit_max_steps:
                    self._save("last.pt")
                    return self.state
                self.scheduler.step()
                self.state["epoch"] = epoch + 1
            self._save("last.pt")
            return self.state
        except Exception:
            try:
                emergency = self._save(f"emergency_step_{self.state['global_step']:08d}.pt")
                self.logger.log({"event": "emergency_checkpoint", "path": str(emergency), **self.state})
            finally:
                raise
