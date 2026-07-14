from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Paper-aligned channel attention for canonical input [B, T, N, F]."""

    def __init__(
        self,
        input_features: int,
        num_receivers: int,
        output_dim: int = 64,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.num_receivers = num_receivers
        self.weight_network = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(num_receivers, output_dim, kernel_size=(1, input_features)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,T,N,F], got {tuple(x.shape)}")
        _, _, receivers, features = x.shape
        if receivers != self.num_receivers or features != self.input_features:
            raise ValueError(
                f"Expected N={self.num_receivers}, F={self.input_features}; got N={receivers}, F={features}"
            )
        if time_mask is None:
            average = x.mean(dim=1, keepdim=True)
            maximum = x.amax(dim=1, keepdim=True)
        else:
            valid = time_mask[:, :, None, None].to(x.dtype)
            average = (x * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            masked = x.masked_fill(~time_mask[:, :, None, None], torch.finfo(x.dtype).min)
            maximum = masked.amax(dim=1, keepdim=True)
            maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        pooled = torch.cat((average, maximum), dim=1)
        weights = self.weight_network(pooled)
        weighted = x * weights
        fused = self.fusion(weighted.permute(0, 2, 1, 3)).squeeze(-1).transpose(1, 2)
        output = self.norm(fused)
        if time_mask is not None:
            output = output * time_mask.unsqueeze(-1).to(output.dtype)
        return output
