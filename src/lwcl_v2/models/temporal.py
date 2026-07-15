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


class TemporalStem(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        kernels: tuple[int, ...] = (3, 5),
        dropout: float = 0.1,
        activation_name: str = "silu",
        use_first_difference: bool = True,
        use_temporal_convolution: bool = True,
    ) -> None:
        super().__init__()
        self.use_first_difference = use_first_difference
        self.use_temporal_convolution = use_temporal_convolution
        self.velocity_projection = (
            nn.Linear(hidden_dim * 2, hidden_dim) if use_first_difference else None
        )
        self.depthwise = (
            nn.ModuleList(
                [
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel,
                        padding=kernel // 2,
                        groups=hidden_dim,
                    )
                    for kernel in kernels
                ]
            )
            if use_temporal_convolution
            else None
        )
        self.pointwise = (
            nn.Conv1d(hidden_dim * len(kernels), hidden_dim, kernel_size=1)
            if use_temporal_convolution
            else None
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation_name = activation_name

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        if self.velocity_projection is not None:
            velocity = torch.zeros_like(x)
            velocity[:, 1:] = x[:, 1:] - x[:, :-1]
            base = self.velocity_projection(torch.cat((x, velocity), dim=-1))
        else:
            base = x
        if self.depthwise is not None and self.pointwise is not None:
            transposed = base.transpose(1, 2)
            local = self.pointwise(
                torch.cat([layer(transposed) for layer in self.depthwise], dim=1)
            ).transpose(1, 2)
            activated = (
                torch.nn.functional.silu(local)
                if self.activation_name == "silu"
                else torch.nn.functional.gelu(local, approximate="tanh")
            )
            output = self.norm(base + self.dropout(activated))
        else:
            output = self.norm(base)
        return output * time_mask.unsqueeze(-1).to(output.dtype)


class WindowAttentiveMeanPool(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)
        self.output = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(approximate="tanh"),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attentive = torch.sum(x * weights.unsqueeze(-1), dim=1)
        valid = mask.unsqueeze(-1).to(x.dtype)
        mean = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.output(torch.cat((attentive, mean), dim=-1))


class LocalGlobalTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        max_seq_len: int = 96,
        projection_dim: int = 192,
        window_size: int = 5,
        window_stride: int = 2,
        local_heads: int = 4,
        local_layers: int = 1,
        output_dim: int = 256,
        global_heads: int = 8,
        global_layers: int = 2,
        ffn_factor: int = 2,
        dropout: float = 0.1,
        input_absolute_position_encoding: bool = False,
        global_use_physical_time: bool = True,
        rope_time_unit_ms: float = 50.0,
        normalized_phase_enabled: bool = False,
        transformer_ffn: str = "gelu",
        local_position_enabled: bool = True,
        global_position_enabled: bool = True,
        multiscale_enabled: bool = False,
        long_window_size: int = 9,
        long_window_stride: int = 4,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.window_stride = window_stride
        self.input_absolute_position_encoding = input_absolute_position_encoding
        self.global_use_physical_time = global_use_physical_time
        self.rope_time_unit_ms = rope_time_unit_ms
        self.normalized_phase_enabled = normalized_phase_enabled
        self.local_position_enabled = local_position_enabled
        self.global_position_enabled = global_position_enabled
        self.multiscale_enabled = multiscale_enabled
        self.long_window_size = long_window_size
        self.long_window_stride = long_window_stride
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if input_absolute_position_encoding:
            self.register_buffer(
                "absolute_encoding", sinusoidal_encoding(max_seq_len, projection_dim), persistent=False
            )
        else:
            self.absolute_encoding = None
        if normalized_phase_enabled:
            self.phase_mlp = nn.Sequential(
                nn.Linear(3, projection_dim), nn.GELU(), nn.Linear(projection_dim, projection_dim)
            )
            self.phase_alpha = nn.Parameter(torch.zeros(()))
        else:
            self.phase_mlp = None
            self.phase_alpha = None
        self.local_encoder = RotaryEncoder(
            local_layers,
            projection_dim,
            local_heads,
            projection_dim * ffn_factor,
            dropout,
            ffn_type=transformer_ffn,
        )
        self.window_pool = WindowAttentiveMeanPool(projection_dim, output_dim)
        if multiscale_enabled:
            self.long_local_encoder = RotaryEncoder(
                local_layers,
                projection_dim,
                local_heads,
                projection_dim * ffn_factor,
                dropout,
                ffn_type=transformer_ffn,
            )
            self.long_window_pool = WindowAttentiveMeanPool(projection_dim, output_dim)
            self.cross_scale_attention = nn.MultiheadAttention(
                output_dim, global_heads, dropout=dropout, batch_first=True
            )
            self.cross_scale_norm = nn.LayerNorm(output_dim)
        else:
            self.long_local_encoder = None
            self.long_window_pool = None
            self.cross_scale_attention = None
            self.cross_scale_norm = None
        self.global_encoder = RotaryEncoder(
            global_layers,
            output_dim,
            global_heads,
            output_dim * ffn_factor,
            dropout,
            ffn_type=transformer_ffn,
        )

    def _encode_windows(
        self,
        x: torch.Tensor,
        time_mask: torch.Tensor,
        position_ids: torch.Tensor,
        frame_times_ms: torch.Tensor,
        window_size: int,
        window_stride: int,
        encoder: RotaryEncoder,
        pool: WindowAttentiveMeanPool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        if length < window_size:
            pad = window_size - length
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            time_mask = torch.nn.functional.pad(time_mask, (0, pad), value=False)
            position_ids = torch.nn.functional.pad(position_ids, (0, pad), value=0)
            frame_times_ms = torch.nn.functional.pad(frame_times_ms, (0, pad), value=0.0)
        windows = x.unfold(1, window_size, window_stride).permute(0, 1, 3, 2)
        mask_windows = time_mask.unfold(1, window_size, window_stride)
        position_windows = position_ids.unfold(1, window_size, window_stride)
        time_windows_ms = frame_times_ms.unfold(1, window_size, window_stride)
        num_windows = windows.shape[1]
        flat = windows.reshape(batch * num_windows, window_size, -1)
        flat_mask = mask_windows.reshape(batch * num_windows, window_size)
        local_positions = torch.arange(window_size, device=x.device).expand(batch * num_windows, -1)
        if not self.local_position_enabled:
            local_positions = torch.zeros_like(local_positions)
        encoded = encoder(flat, local_positions, flat_mask)
        tokens = pool(encoded, flat_mask).reshape(batch, num_windows, -1)
        output_mask = mask_windows.any(dim=-1)
        center = window_size // 2
        return tokens, output_mask, position_windows[:, :, center], time_windows_ms[:, :, center]

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        time_mask: torch.Tensor,
        frame_times_ms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        if length > self.max_seq_len:
            raise ValueError(f"Input T={length} exceeds max_seq_len={self.max_seq_len}")
        x = self.projection(x)
        if self.absolute_encoding is not None:
            x = x + self.absolute_encoding[position_ids.long()]
        if self.phase_mlp is not None and self.phase_alpha is not None:
            valid_length = time_mask.sum(dim=1, keepdim=True).clamp_min(2).to(x.dtype)
            phase = position_ids.to(x.dtype) / (valid_length - 1.0)
            phase = phase.clamp(0.0, 1.0)
            phase_input = torch.stack(
                (phase, torch.sin(math.pi * phase), torch.cos(math.pi * phase)), dim=-1
            )
            x = x + self.phase_alpha * self.phase_mlp(phase_input)
        if frame_times_ms is None:
            frame_times_ms = position_ids.to(torch.float32) * self.rope_time_unit_ms
        tokens, output_mask, center_positions, output_times_ms = self._encode_windows(
            x,
            time_mask,
            position_ids,
            frame_times_ms,
            self.window_size,
            self.window_stride,
            self.local_encoder,
            self.window_pool,
        )
        if self.multiscale_enabled:
            assert self.long_local_encoder is not None
            assert self.long_window_pool is not None
            assert self.cross_scale_attention is not None
            assert self.cross_scale_norm is not None
            long_tokens, long_mask, _, _ = self._encode_windows(
                x,
                time_mask,
                position_ids,
                frame_times_ms,
                self.long_window_size,
                self.long_window_stride,
                self.long_local_encoder,
                self.long_window_pool,
            )
            cross, _ = self.cross_scale_attention(
                query=tokens,
                key=long_tokens,
                value=long_tokens,
                key_padding_mask=~long_mask,
                need_weights=False,
            )
            tokens = self.cross_scale_norm(tokens + cross)
        if self.global_use_physical_time:
            output_positions = output_times_ms / self.rope_time_unit_ms
        else:
            output_positions = center_positions.to(torch.float32)
        if not self.global_position_enabled:
            output_positions = torch.zeros_like(output_positions)
        output = self.global_encoder(tokens, output_positions, output_mask)
        return output, output_positions, output_mask, output_times_ms


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mean = torch.sum(x * weights.unsqueeze(-1), dim=1)
        variance = torch.sum((x - mean[:, None]) ** 2 * weights.unsqueeze(-1), dim=1)
        std = torch.sqrt(variance.clamp_min(1e-6))
        maximum = x.masked_fill(~mask.unsqueeze(-1), -1e4).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, std, maximum), dim=-1)


class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)
