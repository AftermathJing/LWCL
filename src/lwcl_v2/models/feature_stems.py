from __future__ import annotations

import torch
from torch import nn


def activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    raise ValueError(f"Unsupported front-end activation: {name}")


def _apply_masks(x: torch.Tensor, time_mask: torch.Tensor, receiver_mask: torch.Tensor) -> torch.Tensor:
    mask = time_mask[:, :, None, None] & receiver_mask[:, None, :, None]
    return x * mask.to(x.dtype)


class RSSIStem(nn.Module):
    def __init__(self, output_dim: int = 16, dropout: float = 0.1, activation_name: str = "silu") -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(4, output_dim),
            nn.LayerNorm(output_dim),
            activation(activation_name),
        )
        self.temporal = nn.Conv1d(
            output_dim, output_dim, kernel_size=5, padding=2, groups=output_dim
        )
        self.dropout = nn.Dropout(dropout)
        self.activation_name = activation_name

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor, receiver_mask: torch.Tensor) -> torch.Tensor:
        batch, length, receivers, _ = x.shape
        x = self.projection(x)
        x = x.permute(0, 2, 3, 1).reshape(batch * receivers, -1, length)
        x = self.temporal(x)
        x = x.reshape(batch, receivers, -1, length).permute(0, 3, 1, 2)
        x = torch.nn.functional.silu(x) if self.activation_name == "silu" else torch.nn.functional.gelu(x, approximate="tanh")
        return _apply_masks(self.dropout(x), time_mask, receiver_mask)


class DopplerStem(nn.Module):
    def __init__(
        self, input_bins: int = 25, output_dim: int = 64, dropout: float = 0.1, activation_name: str = "silu"
    ) -> None:
        super().__init__()
        self.input_bins = input_bins
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            activation(activation_name),
            nn.Conv1d(32, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            activation(activation_name),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor, receiver_mask: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_bins:
            raise ValueError(f"Expected {self.input_bins} Doppler bins, got {x.shape[-1]}")
        batch, length, receivers, bins = x.shape
        encoded = self.encoder(x.reshape(batch * length * receivers, 1, bins)).squeeze(-1)
        encoded = encoded.reshape(batch, length, receivers, -1)
        return _apply_masks(self.dropout(encoded), time_mask, receiver_mask)


class DifferentialCSIStem(nn.Module):
    def __init__(
        self,
        components: int = 10,
        pair_dim: int = 8,
        output_dim: int = 48,
        dropout: float = 0.1,
        activation_name: str = "silu",
    ) -> None:
        super().__init__()
        self.components = components
        self.pair_projection = nn.Sequential(nn.Linear(2, pair_dim), activation(activation_name))
        self.output = nn.Sequential(
            nn.Linear(components * pair_dim, output_dim),
            nn.LayerNorm(output_dim),
            activation(activation_name),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor, receiver_mask: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.components, 2):
            raise ValueError(f"Expected differential CSI [...,{self.components},2], got {tuple(x.shape)}")
        encoded = self.pair_projection(x).flatten(-2)
        return _apply_masks(self.output(encoded), time_mask, receiver_mask)


class FeatureFamilyEncoder(nn.Module):
    def __init__(
        self,
        rssi_dim: int = 16,
        doppler_dim: int = 64,
        differential_dim: int = 48,
        fused_dim: int = 128,
        doppler_bins: int = 25,
        differential_components: int = 10,
        dropout: float = 0.1,
        mode: str = "family_stems",
        activation_name: str = "silu",
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode not in {"family_stems", "flat_49"}:
            raise ValueError(f"Unsupported feature encoder mode: {mode}")
        self.rssi = RSSIStem(rssi_dim, dropout, activation_name)
        self.doppler = DopplerStem(doppler_bins, doppler_dim, dropout, activation_name)
        self.differential = DifferentialCSIStem(
            differential_components, 8, differential_dim, dropout, activation_name
        )
        total = rssi_dim + doppler_dim + differential_dim
        self.fusion = nn.Sequential(
            nn.Linear(total, fused_dim),
            nn.LayerNorm(fused_dim),
            activation(activation_name),
            nn.Dropout(dropout),
        )
        self.flat_fusion = nn.Sequential(
            nn.Linear(4 + doppler_bins + differential_components * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            activation(activation_name),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        rssi: torch.Tensor,
        doppler: torch.Tensor,
        differential_csi: torch.Tensor,
        time_mask: torch.Tensor,
        receiver_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.mode == "flat_49":
            flat = torch.cat((rssi, doppler, differential_csi.flatten(-2)), dim=-1)
            fused = _apply_masks(self.flat_fusion(flat), time_mask, receiver_mask)
            return {"fused": fused}
        rssi_features = self.rssi(rssi, time_mask, receiver_mask)
        doppler_features = self.doppler(doppler, time_mask, receiver_mask)
        differential_features = self.differential(differential_csi, time_mask, receiver_mask)
        fused = self.fusion(torch.cat((rssi_features, doppler_features, differential_features), dim=-1))
        fused = _apply_masks(fused, time_mask, receiver_mask)
        return {
            "fused": fused,
            "rssi": rssi_features,
            "doppler": doppler_features,
            "differential_csi": differential_features,
        }
