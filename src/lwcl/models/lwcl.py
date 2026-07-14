from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .adapter import SignalAdapter
from .backbones import build_backbone
from .channel_attention import ChannelAttention
from .hste import HierarchicalSpatioTemporalEncoder


class LWCLModel(nn.Module):
    """Paper-aligned end-to-end model with canonical input [B,T,N,F]."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config["model"]
        data_config = config["data"]
        self.num_labels = int(data_config["num_labels"])
        self.max_seq_len = int(data_config["max_seq_len"])
        self.num_receivers = int(data_config["num_receivers"])
        self.input_features = int(data_config["input_features"])

        backbone = build_backbone(model_config["backbone"], self.num_labels)
        target_hidden_size = int(backbone.hidden_size)

        channel_config = model_config["channel_attention"]
        self.channel_attention = ChannelAttention(
            input_features=self.input_features,
            num_receivers=self.num_receivers,
            output_dim=int(channel_config.get("output_dim", 64)),
            hidden_dim=int(channel_config.get("hidden_dim", 32)),
            dropout=float(channel_config.get("dropout", 0.1)),
        )

        hste_config = model_config["hste"]
        self.hste = HierarchicalSpatioTemporalEncoder(
            input_dim=int(channel_config.get("output_dim", 64)),
            max_seq_len=self.max_seq_len,
            projection_dim=int(hste_config.get("projection_dim", 128)),
            window_size=int(hste_config.get("window_size", 8)),
            window_stride=int(hste_config.get("window_stride", 4)),
            local_heads=int(hste_config.get("local_heads", 4)),
            local_layers=int(hste_config.get("local_layers", 1)),
            output_dim=int(hste_config.get("output_dim", 256)),
            global_heads=int(hste_config.get("global_heads", 8)),
            global_layers=int(hste_config.get("global_layers", 2)),
            ffn_factor=int(hste_config.get("ffn_factor", 2)),
            dropout=float(hste_config.get("dropout", 0.1)),
        )

        adapter_config = model_config["adapter"]
        self.adapter = SignalAdapter(
            input_dim=int(hste_config.get("output_dim", 256)),
            output_dim=target_hidden_size,
            num_layers=int(adapter_config.get("num_layers", 2)),
            num_heads=int(adapter_config.get("num_heads", 8)),
            ffn_factor=int(adapter_config.get("ffn_factor", 2)),
            dropout=float(adapter_config.get("dropout", 0.1)),
        )
        self.backbone = backbone

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if features.ndim != 4:
            raise ValueError("features must have shape [B,T,N,F]")
        batch, length, receivers, feature_dim = features.shape
        if receivers != self.num_receivers or feature_dim != self.input_features:
            raise ValueError("Input receiver/feature dimensions do not match the resolved configuration")
        if length > self.max_seq_len:
            raise ValueError(f"Input length {length} exceeds max_seq_len={self.max_seq_len}")
        if attention_mask is None:
            attention_mask = torch.ones(batch, length, device=features.device, dtype=torch.bool)
        if position_ids is None:
            position_ids = torch.arange(length, device=features.device).expand(batch, -1)

        signal = self.channel_attention(features, attention_mask)
        signal, reduced_positions, reduced_mask = self.hste(signal, position_ids, attention_mask)
        aligned = self.adapter(signal, reduced_positions, reduced_mask)
        outputs = self.backbone(aligned, reduced_positions, reduced_mask, labels)
        outputs["signal_embeddings"] = aligned
        outputs["reduced_attention_mask"] = reduced_mask
        return outputs

    def trainable_parameter_report(self) -> dict[str, float | int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        report: dict[str, Any] = {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_percent": 100.0 * trainable / max(total, 1),
        }
        component_report: dict[str, dict[str, float | int]] = {}
        for component_name in ("channel_attention", "hste", "adapter", "backbone"):
            component = getattr(self, component_name)
            component_total = sum(parameter.numel() for parameter in component.parameters())
            component_trainable = sum(
                parameter.numel() for parameter in component.parameters() if parameter.requires_grad
            )
            component_report[component_name] = {
                "total": component_total,
                "trainable": component_trainable,
                "trainable_percent": 100.0 * component_trainable / max(component_total, 1),
            }
        report["components"] = component_report
        return report
