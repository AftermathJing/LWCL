from __future__ import annotations

import math

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Apply RoPE to x shaped [B, H, T, D]."""
    if rotary_dim == 0:
        return x
    rotary_dim = min(rotary_dim, x.shape[-1])
    rotary_dim -= rotary_dim % 2
    inv_freq = theta ** (-torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    angles = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
    angles = torch.repeat_interleave(angles, 2, dim=-1).unsqueeze(1)
    cos = angles.cos().to(dtype=x.dtype)
    sin = angles.sin().to(dtype=x.dtype)
    rotated = x[..., :rotary_dim]
    rotated = rotated * cos + _rotate_half(rotated) * sin
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rotary_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        default_rotary = self.head_dim if self.head_dim % 2 == 0 else self.head_dim - 1
        self.rotary_dim = default_rotary if rotary_dim is None else rotary_dim
        if self.rotary_dim < 0 or self.rotary_dim > self.head_dim or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be an even value within the attention head dimension")
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = apply_rope(self._split_heads(self.q_proj(x)), position_ids, self.rotary_dim)
        k = apply_rope(self._split_heads(self.k_proj(x)), position_ids, self.rotary_dim)
        v = self._split_heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(~key_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)
        output = torch.matmul(self.dropout(weights), v)
        output = output.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        output = self.out_proj(output)
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).to(output.dtype)
        return output


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, baseline_ffn_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(8, int(round((baseline_ffn_dim * 2.0 / 3.0) / 8.0)) * 8)
        self.value_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.value_proj(x) * torch.nn.functional.silu(self.gate_proj(x))
        return self.out_proj(self.dropout(gated))


class RotaryEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rotary_dim: int | None,
        ffn_type: str = "gelu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = RotarySelfAttention(d_model, num_heads, rotary_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        if ffn_type == "gelu":
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_dim),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model),
            )
        elif ffn_type == "swiglu":
            self.ffn = SwiGLUFFN(d_model, ffn_dim, dropout)
        else:
            raise ValueError(f"Unsupported transformer FFN type: {ffn_type}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), position_ids, attention_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        return x


class RotaryEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        rotary_dim: int | None = None,
        ffn_type: str = "gelu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RotaryEncoderLayer(
                    d_model, num_heads, ffn_dim, dropout, rotary_dim, ffn_type=ffn_type
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, position_ids, attention_mask)
        return self.final_norm(x)
