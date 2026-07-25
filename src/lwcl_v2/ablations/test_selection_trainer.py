from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from lwcl_v2.training.metrics import classification_metrics, subject_metrics
from lwcl_v2.training.trainer import Trainer


class TestAccuracySelectionTrainer(Trainer):
    """Isolated Wi-CBR-style trainer that selects checkpoints on test accuracy."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.state["selection_split"] = "test"
        self.state["selection_metric"] = "accuracy"

    @torch.no_grad()
    def evaluate(
        self,
        loader=None,
        split: str = "test",
        use_ema: bool = True,
    ) -> dict[str, Any]:
        loader = loader or self.validation_loader
        self.model.eval()
        predictions = []
        targets = []
        subjects: list[str] = []
        losses = []
        parameter_context = (
            self.ema.average_parameters(self.model)
            if (use_ema and self.ema is not None)
            else nullcontext()
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
            all_targets,
            all_predictions,
            subjects,
            self.model.num_labels,
        )
        worst_subject = min(value["macro_f1"] for value in metrics["subjects"].values())
        metrics.update(
            {
                "worst_subject_macro_f1": float(worst_subject),
                "selection_score": float(metrics["accuracy"]),
                "selection_metric": "accuracy",
                "loss": float(np.mean(losses)),
                "split": split,
                "weights": "ema" if (use_ema and self.ema is not None) else "raw",
                "epoch": self.state["epoch"],
                "global_step": self.state["global_step"],
            }
        )
        self.logger.log({"event": "evaluation", **metrics})
        return metrics
