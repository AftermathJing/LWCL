from __future__ import annotations

import math

import torch
from torch import nn

from .rotary import RotaryEncoder


def sinusoidal_encoding(max_length: int, dim: int) -> torch.Tensor:
    positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    encoding = torch.zeros(max_length, dim)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class HierarchicalSpatioTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        max_seq_len: int,
        projection_dim: int = 128,
        window_size: int = 8,
        window_stride: int = 4,
        local_heads: int = 4,
        local_layers: int = 1,
        output_dim: int = 256,
        global_heads: int = 8,
        global_layers: int = 2,
        ffn_factor: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.window_stride = window_stride
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer("absolute_encoding", sinusoidal_encoding(max_seq_len, projection_dim), persistent=False)
        self.local_encoder = RotaryEncoder(
            local_layers, projection_dim, local_heads, projection_dim * ffn_factor, dropout
        )
        self.window_aggregator = nn.Conv1d(projection_dim, output_dim, kernel_size=window_size)
        self.aggregate_norm = nn.LayerNorm(output_dim)
        self.global_encoder = RotaryEncoder(
            global_layers, output_dim, global_heads, output_dim * ffn_factor, dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        if position_ids.shape != (batch, length):
            raise ValueError("position_ids must match [B,T]")
        if position_ids.max().item() >= self.max_seq_len or position_ids.min().item() < 0:
            raise ValueError("position_ids exceed configured max_seq_len")
        if attention_mask is None:
            attention_mask = torch.ones(batch, length, device=x.device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(torch.bool)

        x = self.projection(x)
        x = x + self.absolute_encoding[position_ids]
        if length < self.window_size:
            pad = self.window_size - length
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            position_ids = torch.nn.functional.pad(position_ids, (0, pad), value=0)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=False)

        windows = x.unfold(1, self.window_size, self.window_stride).permute(0, 1, 3, 2)
        mask_windows = attention_mask.unfold(1, self.window_size, self.window_stride)
        position_windows = position_ids.unfold(1, self.window_size, self.window_stride)
        num_windows = windows.shape[1]
        flat_windows = windows.reshape(batch * num_windows, self.window_size, -1)
        flat_masks = mask_windows.reshape(batch * num_windows, self.window_size)
        local_positions = torch.arange(self.window_size, device=x.device).expand(batch * num_windows, -1)
        local = self.local_encoder(flat_windows, local_positions, flat_masks)
        aggregated = self.window_aggregator(local.transpose(1, 2)).squeeze(-1)
        aggregated = self.aggregate_norm(aggregated).view(batch, num_windows, -1)
        output_mask = mask_windows.any(dim=-1)
        center = self.window_size // 2
        output_positions = position_windows[:, :, center]
        global_output = self.global_encoder(aggregated, output_positions, output_mask)
        return global_output, output_positions, output_mask
