from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .feature_stems import FeatureFamilyEncoder
from .receiver_fusion import MaskedMeanReceiverFusion, QualityAwareReceiverFusion, SharedReceiverEncoder
from .temporal import (
    AttentiveStatisticsPooling,
    LocalGlobalTemporalEncoder,
    MaskedAttentionPooling,
    TemporalStem,
)


class SignalV2Classifier(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        data = config["data"]
        model = config["model"]
        stems = model["feature_stems"]
        receiver = model["receiver_fusion"]
        temporal_stem = model["temporal_stem"]
        hste = model["hste"]
        classifier = model["classifier"]
        activations = config.get("activations", {})
        front_activation = str(activations.get("feature_fusion", "silu"))
        self.num_labels = int(data["num_labels"])
        self.max_seq_len = int(data["max_seq_len"])
        self.feature_mode = str(data.get("feature_mode", "current"))
        fused_dim = int(stems.get("fused_dim", 128))
        self.feature_encoder = FeatureFamilyEncoder(
            rssi_dim=int(stems.get("rssi_dim", 16)),
            doppler_dim=int(stems.get("doppler_dim", 64)),
            differential_dim=int(stems.get("differential_csi_dim", 48)),
            fused_dim=fused_dim,
            doppler_bins=int(data.get("doppler_bins", 25)),
            differential_components=int(data.get("differential_components", 10)),
            phase_dim=int(stems.get("csi_ratio_phase_dim", 48)),
            phase_subcarriers=int(data.get("phase_subcarriers", 30)),
            amplitude_bins=int(data.get("amplitude_bins", 64)),
            dropout=float(stems.get("dropout", 0.1)),
            mode=str(stems.get("mode", "family_stems")),
            feature_mode=self.feature_mode,
            activation_name=front_activation,
        )
        self.receiver_encoder = SharedReceiverEncoder(
            hidden_dim=fused_dim,
            layers=int(receiver.get("shared_encoder_layers", 1)),
            dropout=float(receiver.get("dropout", 0.1)),
            activation_name=str(activations.get("receiver_shared_encoder", "silu")),
        )
        receiver_mode = str(receiver.get("mode", "quality_statistics"))
        if receiver_mode == "quality_statistics":
            self.receiver_fusion = QualityAwareReceiverFusion(
                hidden_dim=fused_dim,
                quality_dim=int(data.get("quality_dim", 5)),
                attention_heads=int(receiver.get("attention_heads", 4)),
                dropout=float(receiver.get("dropout", 0.1)),
                use_quality_bias=bool(receiver.get("use_quality_bias", True)),
                activation_name=str(activations.get("feature_fusion", "silu")),
            )
        elif receiver_mode == "masked_mean":
            self.receiver_fusion = MaskedMeanReceiverFusion()
        else:
            raise ValueError(f"Unsupported receiver fusion mode: {receiver_mode}")
        self.use_temporal_stem = bool(temporal_stem.get("enabled", True))
        self.temporal_stem = (
            TemporalStem(
                hidden_dim=int(temporal_stem.get("hidden_dim", fused_dim)),
                kernels=tuple(int(value) for value in temporal_stem.get("kernels", [3, 5])),
                dropout=float(temporal_stem.get("dropout", 0.1)),
                activation_name=str(activations.get("temporal_convolution", "silu")),
                use_first_difference=bool(temporal_stem.get("use_first_difference", True)),
                use_temporal_convolution=bool(
                    temporal_stem.get("use_temporal_convolution", True)
                ),
            )
            if self.use_temporal_stem
            else None
        )
        output_dim = int(hste.get("output_dim", 256))
        self.temporal_encoder = LocalGlobalTemporalEncoder(
            input_dim=fused_dim,
            max_seq_len=self.max_seq_len,
            projection_dim=int(hste.get("projection_dim", 192)),
            window_size=int(hste.get("window_size", 5)),
            window_stride=int(hste.get("window_stride", 2)),
            local_heads=int(hste.get("local_heads", 4)),
            local_layers=int(hste.get("local_layers", 1)),
            output_dim=output_dim,
            global_heads=int(hste.get("global_heads", 8)),
            global_layers=int(hste.get("global_layers", 2)),
            ffn_factor=int(hste.get("ffn_factor", 2)),
            dropout=float(hste.get("dropout", 0.1)),
            input_absolute_position_encoding=bool(
                hste.get("input_absolute_position_encoding", False)
            ),
            global_use_physical_time=bool(
                hste.get("global_position", {}).get("use_physical_time", True)
            ),
            rope_time_unit_ms=float(
                hste.get("global_position", {}).get("rope_time_unit_ms", 50.0)
            ),
            normalized_phase_enabled=bool(
                hste.get("normalized_phase_embedding", {}).get("enabled", False)
            ),
            transformer_ffn=str(hste.get("transformer_ffn", "gelu")),
            local_position_enabled=bool(hste.get("local_position", {}).get("enabled", True)),
            global_position_enabled=bool(
                hste.get("global_position", {}).get("enabled", True)
            ),
            multiscale_enabled=bool(hste.get("multiscale", {}).get("enabled", False)),
            long_window_size=int(hste.get("multiscale", {}).get("long_window_size", 9)),
            long_window_stride=int(hste.get("multiscale", {}).get("long_window_stride", 4)),
            window_pooling=str(hste.get("window_pooling", "attentive_mean")),
        )
        pooling = str(classifier.get("pooling", "attentive_statistics"))
        if pooling == "attentive_statistics":
            self.pool = AttentiveStatisticsPooling(output_dim)
            pooling_dim = output_dim * 3
        elif pooling == "attention":
            self.pool = MaskedAttentionPooling(output_dim)
            pooling_dim = output_dim
        else:
            raise ValueError(f"Unsupported sequence pooling: {pooling}")
        classifier_hidden = int(classifier.get("hidden_dim", 256))
        self.embedding = nn.Sequential(
            nn.LayerNorm(pooling_dim),
            nn.Linear(pooling_dim, classifier_hidden),
            nn.GELU(approximate="tanh"),
        )
        self.dropout = nn.Dropout(float(classifier.get("dropout", 0.2)))
        self.classifier = nn.Linear(classifier_hidden, self.num_labels)

    def forward(
        self,
        rssi: torch.Tensor | None,
        doppler: torch.Tensor | None,
        differential_csi: torch.Tensor | None,
        time_mask: torch.Tensor,
        receiver_mask: torch.Tensor,
        receiver_quality: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        frame_times_ms: torch.Tensor | None = None,
        csi_ratio_phase: torch.Tensor | None = None,
        amplitude: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        reference = amplitude if amplitude is not None else rssi
        if reference is None:
            raise ValueError("SignalV2Classifier requires rssi or amplitude input")
        batch, length = reference.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(length, device=rssi.device).expand(batch, -1)
        families = self.feature_encoder(
            rssi,
            doppler,
            differential_csi,
            time_mask,
            receiver_mask,
            csi_ratio_phase,
            amplitude=amplitude,
        )
        receivers = self.receiver_encoder(families["fused"], receiver_mask)
        spatial, receiver_attention = self.receiver_fusion(
            receivers, receiver_mask, receiver_quality, time_mask
        )
        temporal = self.temporal_stem(spatial, time_mask) if self.temporal_stem is not None else spatial
        encoded, reduced_positions, reduced_mask, reduced_times_ms = self.temporal_encoder(
            temporal, position_ids, time_mask, frame_times_ms
        )
        pooled = self.pool(encoded, reduced_mask)
        embedding = self.embedding(pooled)
        logits = self.classifier(self.dropout(embedding))
        return {
            "logits": logits,
            "embedding": embedding,
            "temporal_tokens": encoded,
            "reduced_attention_mask": reduced_mask,
            "reduced_position_ids": reduced_positions,
            "reduced_times_ms": reduced_times_ms,
            "receiver_attention": receiver_attention,
            "family_features": families,
        }

    def parameter_report(self) -> dict[str, Any]:
        components = {}
        for name in (
            "feature_encoder",
            "receiver_encoder",
            "receiver_fusion",
            "temporal_stem",
            "temporal_encoder",
            "pool",
            "embedding",
            "classifier",
        ):
            module = getattr(self, name)
            if module is not None:
                components[name] = sum(parameter.numel() for parameter in module.parameters())
        return {"total": sum(components.values()), "trainable": sum(components.values()), "components": components}
