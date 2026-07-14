from __future__ import annotations

import torch
from torch import nn

from .rotary import RotaryEncoder


class SignalAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_factor: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = RotaryEncoder(
            num_layers, output_dim, num_heads, output_dim * ffn_factor, dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(self.projection(x), position_ids, attention_mask)
