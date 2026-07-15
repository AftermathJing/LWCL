from __future__ import annotations

import math

import torch
from torch import nn

from .feature_stems import activation


class SharedReceiverEncoder(nn.Module):
    def __init__(
        self, hidden_dim: int = 128, layers: int = 1, dropout: float = 0.1, activation_name: str = "silu"
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    activation(activation_name),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
            x = x * mask[:, None, :, None].to(x.dtype)
        return x


class QualityAwareReceiverFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        quality_dim: int = 5,
        attention_heads: int = 4,
        dropout: float = 0.1,
        use_quality_bias: bool = True,
        activation_name: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.hidden_dim = hidden_dim
        self.heads = attention_heads
        self.head_dim = hidden_dim // attention_heads
        self.use_quality_bias = use_quality_bias
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention_out = nn.Linear(hidden_dim, hidden_dim)
        self.quality_bias = nn.Sequential(
            nn.LayerNorm(quality_dim),
            nn.Linear(quality_dim, attention_heads),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(activation_name),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _statistics(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = mask[:, None, :, None].to(x.dtype)
        count = weights.sum(dim=2).clamp_min(1.0)
        mean = (x * weights).sum(dim=2) / count
        variance = ((x - mean[:, :, None]) ** 2 * weights).sum(dim=2) / count
        std = torch.sqrt(variance.clamp_min(1e-6))
        masked = x.masked_fill(~mask[:, None, :, None], -1e4)
        maximum = masked.amax(dim=2)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return mean, maximum, std

    def forward(
        self,
        x: torch.Tensor,
        receiver_mask: torch.Tensor,
        receiver_quality: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, receivers, _ = x.shape
        mean, maximum, std = self._statistics(x, receiver_mask)
        query = self.q_proj(mean).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(x).view(batch, length, receivers, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        values = self.v_proj(x).view(batch, length, receivers, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        scores = torch.einsum("bhtd,bhtrd->bhtr", query, keys) / math.sqrt(self.head_dim)
        if self.use_quality_bias:
            quality_bias = self.quality_bias(receiver_quality).permute(0, 2, 1)[:, :, None, :]
            scores = scores + quality_bias
        scores = scores.masked_fill(~receiver_mask[:, None, None, :], -1e4)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.nan_to_num(attention)
        pooled = torch.einsum("bhtr,bhtrd->bhtd", attention, values)
        pooled = pooled.transpose(1, 2).reshape(batch, length, self.hidden_dim)
        pooled = self.attention_out(pooled)
        fused = self.output(torch.cat((mean, maximum, std, pooled), dim=-1))
        fused = fused * time_mask.unsqueeze(-1).to(fused.dtype)
        return fused, attention.mean(dim=1)


class MaskedMeanReceiverFusion(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        receiver_mask: torch.Tensor,
        receiver_quality: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del receiver_quality
        weights = receiver_mask[:, None, :, None].to(x.dtype)
        fused = (x * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        fused = fused * time_mask.unsqueeze(-1).to(fused.dtype)
        attention = receiver_mask[:, None, :].expand(-1, x.shape[1], -1).to(x.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return fused, attention
