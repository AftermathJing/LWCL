from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CrossSubjectSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        logits = embeddings @ embeddings.T / self.temperature
        batch = embeddings.shape[0]
        self_mask = torch.eye(batch, device=embeddings.device, dtype=torch.bool)
        positive_mask = (
            labels[:, None].eq(labels[None, :])
            & subject_ids[:, None].ne(subject_ids[None, :])
            & ~self_mask
        )
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * (~self_mask).to(logits.dtype)
        log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            return embeddings.sum() * 0.0
        mean_positive_log_probability = (
            (log_probability * positive_mask.to(log_probability.dtype)).sum(dim=1)
            / positive_count.clamp_min(1)
        )
        return -mean_positive_log_probability[valid].mean()


class SignalV2Loss(nn.Module):
    def __init__(
        self,
        label_smoothing: float = 0.02,
        cross_entropy_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.cross_entropy_weight = cross_entropy_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive = CrossSubjectSupervisedContrastiveLoss(contrastive_temperature)

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cross_entropy = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        contrastive = self.contrastive(embeddings, labels, subject_ids)
        total = self.cross_entropy_weight * cross_entropy + self.contrastive_weight * contrastive
        return {
            "loss": total,
            "cross_entropy": cross_entropy,
            "cross_subject_supcon": contrastive,
        }
