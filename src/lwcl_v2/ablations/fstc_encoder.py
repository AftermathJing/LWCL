from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lwcl_v2.models.feature_stems import FeatureFamilyEncoder
from lwcl_v2.models.receiver_fusion import (
    MaskedMeanReceiverFusion,
    QualityAwareReceiverFusion,
    SharedReceiverEncoder,
)
from lwcl_v2.models.signal_v2 import SignalV2Classifier
from lwcl_v2.models.temporal import (
    AttentiveStatisticsPooling,
    LocalGlobalTemporalEncoder,
    MaskedAttentionPooling,
    TemporalStem,
)


ABLATION_VARIANTS = {
    "full_control",
    "no_feature_encoder",
    "no_spatial_encoder",
    "no_temporal_encoder",
}


def fstc_ablation_spec(variant: str) -> dict[str, Any]:
    specs = {
        "full_control": {
            "removed": [],
            "replacement": "Original SignalV2Classifier without source changes",
        },
        "no_feature_encoder": {
            "removed": ["RSSIStem", "DopplerStem", "DifferentialCSIStem", "feature-family fusion"],
            "replacement": "Per-frame/per-receiver raw 49D linear projection to fused_dim",
        },
        "no_spatial_encoder": {
            "removed": ["SharedReceiverEncoder", "quality-aware receiver attention", "receiver statistics fusion"],
            "replacement": "Non-parametric receiver-mask-aware mean",
        },
        "no_temporal_encoder": {
            "removed": [
                "TemporalStem",
                "local-window Transformer",
                "global Transformer",
                "temporal/position encoding",
            ],
            "replacement": "Independent per-frame linear projection followed by existing masked sequence pooling",
        },
    }
    if variant not in specs:
        raise ValueError(f"Unsupported FSTC ablation variant: {variant}")
    return {"variant": variant, **specs[variant]}


class FlatRawFeatureProjection(nn.Module):
    """Dimension-matching replacement with no feature-family encoder."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        activation_name: str,
    ) -> None:
        super().__init__()
        if activation_name == "silu":
            activation: nn.Module = nn.SiLU()
        elif activation_name == "gelu":
            activation = nn.GELU(approximate="tanh")
        else:
            raise ValueError(f"Unsupported flat projection activation: {activation_name}")
        self.input_dim = int(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, output_dim),
            nn.LayerNorm(output_dim),
            activation,
            nn.Dropout(dropout),
        )

    def forward(
        self,
        rssi: torch.Tensor,
        doppler: torch.Tensor,
        differential_csi: torch.Tensor,
        time_mask: torch.Tensor,
        receiver_mask: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        flat = torch.cat((rssi, doppler, differential_csi.flatten(-2)), dim=-1)
        if flat.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} raw feature dimensions, got {flat.shape[-1]}"
            )
        fused = self.projection(flat)
        mask = time_mask[:, :, None, None] & receiver_mask[:, None, :, None]
        return {"fused": fused * mask.to(fused.dtype), "raw_flat": flat}


class IdentityReceiverEncoder(nn.Module):
    """Non-parametric replacement for the shared receiver encoder."""

    def forward(self, x: torch.Tensor, receiver_mask: torch.Tensor) -> torch.Tensor:
        return x * receiver_mask[:, None, :, None].to(x.dtype)


class FramewiseTemporalProjection(nn.Module):
    """Projects each frame independently and performs no cross-time interaction."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        time_mask: torch.Tensor,
        frame_times_ms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.projection(x)
        encoded = encoded * time_mask.unsqueeze(-1).to(encoded.dtype)
        if frame_times_ms is None:
            frame_times_ms = position_ids.to(torch.float32)
        return encoded, position_ids.to(torch.float32), time_mask, frame_times_ms


class FSTCEncoderAblationClassifier(nn.Module):
    """Isolated module-level FSTC ablations for the Widar3 signal pipeline.

    This class intentionally lives outside ``lwcl_v2.models`` so the original
    FSTC implementation and all existing experiment entrypoints remain unchanged.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        variant = str(config.get("ablation", {}).get("variant", ""))
        if variant not in ABLATION_VARIANTS - {"full_control"}:
            raise ValueError(
                "FSTCEncoderAblationClassifier requires one of "
                "no_feature_encoder/no_spatial_encoder/no_temporal_encoder"
            )
        data = config["data"]
        if str(data.get("feature_mode", "current")) != "current":
            raise ValueError("Widar3 FSTC encoder ablations require data.feature_mode=current")
        model = config["model"]
        stems = model["feature_stems"]
        receiver = model["receiver_fusion"]
        temporal_stem = model["temporal_stem"]
        hste = model["hste"]
        classifier = model["classifier"]
        activations = config.get("activations", {})

        self.ablation_variant = variant
        self.num_labels = int(data["num_labels"])
        self.max_seq_len = int(data["max_seq_len"])
        fused_dim = int(stems.get("fused_dim", 128))
        front_activation = str(activations.get("feature_fusion", "silu"))

        if variant == "no_feature_encoder":
            raw_dim = (
                4
                + int(data.get("doppler_bins", 25))
                + int(data.get("differential_components", 10)) * 2
            )
            self.feature_encoder: nn.Module = FlatRawFeatureProjection(
                input_dim=raw_dim,
                output_dim=fused_dim,
                dropout=float(stems.get("dropout", 0.1)),
                activation_name=front_activation,
            )
        else:
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
                mode="family_stems",
                feature_mode="current",
                activation_name=front_activation,
            )

        if variant == "no_spatial_encoder":
            self.receiver_encoder: nn.Module = IdentityReceiverEncoder()
            self.receiver_fusion: nn.Module = MaskedMeanReceiverFusion()
        else:
            self.receiver_encoder = SharedReceiverEncoder(
                hidden_dim=fused_dim,
                layers=int(receiver.get("shared_encoder_layers", 1)),
                dropout=float(receiver.get("dropout", 0.1)),
                activation_name=str(activations.get("receiver_shared_encoder", "silu")),
            )
            self.receiver_fusion = QualityAwareReceiverFusion(
                hidden_dim=fused_dim,
                quality_dim=int(data.get("quality_dim", 5)),
                attention_heads=int(receiver.get("attention_heads", 4)),
                dropout=float(receiver.get("dropout", 0.1)),
                use_quality_bias=bool(receiver.get("use_quality_bias", True)),
                activation_name=front_activation,
            )

        output_dim = int(hste.get("output_dim", 256))
        if variant == "no_temporal_encoder":
            self.temporal_stem: nn.Module | None = None
            self.temporal_encoder: nn.Module = FramewiseTemporalProjection(
                input_dim=fused_dim,
                output_dim=output_dim,
                dropout=float(hste.get("dropout", 0.1)),
            )
        else:
            self.temporal_stem = TemporalStem(
                hidden_dim=int(temporal_stem.get("hidden_dim", fused_dim)),
                kernels=tuple(int(value) for value in temporal_stem.get("kernels", [3, 5])),
                dropout=float(temporal_stem.get("dropout", 0.1)),
                activation_name=str(activations.get("temporal_convolution", "silu")),
                use_first_difference=bool(temporal_stem.get("use_first_difference", True)),
                use_temporal_convolution=bool(
                    temporal_stem.get("use_temporal_convolution", True)
                ),
            )
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
                local_position_enabled=bool(
                    hste.get("local_position", {}).get("enabled", True)
                ),
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
            self.pool: nn.Module = AttentiveStatisticsPooling(output_dim)
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
        amplitude_branch: torch.Tensor | None = None,
        dynamic_residual_branch: torch.Tensor | None = None,
        temporal_difference_branch: torch.Tensor | None = None,
        subcarrier_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del amplitude, amplitude_branch, dynamic_residual_branch, temporal_difference_branch, subcarrier_mask
        if rssi is None or doppler is None or differential_csi is None:
            raise ValueError("Widar3 FSTC ablations require RSSI, Doppler, and differential CSI")
        batch, length = rssi.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(length, device=rssi.device).expand(batch, -1)
        families = self.feature_encoder(
            rssi,
            doppler,
            differential_csi,
            time_mask,
            receiver_mask,
            csi_ratio_phase=csi_ratio_phase,
        )
        receivers = self.receiver_encoder(families["fused"], receiver_mask)
        spatial, receiver_attention = self.receiver_fusion(
            receivers, receiver_mask, receiver_quality, time_mask
        )
        temporal = (
            self.temporal_stem(spatial, time_mask)
            if self.temporal_stem is not None
            else spatial
        )
        encoded, reduced_positions, reduced_mask, reduced_times_ms = self.temporal_encoder(
            temporal, position_ids, time_mask, frame_times_ms
        )
        pooled = self.pool(encoded, reduced_mask)
        embedding = self.embedding(pooled)
        logits = self.classifier(self.dropout(embedding))
        return {
            "logits": logits,
            "embedding": embedding,
            "pooled_features": pooled,
            "spatial_features": spatial,
            "receiver_features": receivers,
            "temporal_tokens": encoded,
            "reduced_attention_mask": reduced_mask,
            "reduced_position_ids": reduced_positions,
            "reduced_times_ms": reduced_times_ms,
            "receiver_attention": receiver_attention,
            "family_features": families,
        }

    def parameter_report(self) -> dict[str, Any]:
        components: dict[str, int] = {}
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
        total = sum(components.values())
        return {
            "total": total,
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "components": components,
            "ablation": fstc_ablation_spec(self.ablation_variant),
        }


def build_fstc_encoder_model(config: dict[str, Any]) -> nn.Module:
    variant = str(config.get("ablation", {}).get("variant", ""))
    if variant not in ABLATION_VARIANTS:
        raise ValueError(
            f"ablation.variant must be one of {sorted(ABLATION_VARIANTS)}, got {variant!r}"
        )
    if variant == "full_control":
        model = SignalV2Classifier(config)
        model.ablation_variant = variant
        return model
    return FSTCEncoderAblationClassifier(config)
