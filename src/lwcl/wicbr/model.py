from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_size = 7
        self.spatial = BasicConv(
            3,
            1,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bn=True,
            relu=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.sigmoid(self.spatial(x))
        return x * scale + x, scale


class DPFusion(nn.Module):
    def __init__(self, channels: int, group_num: int = 4, gate_threshold: float = 0.5) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(num_channels=channels, num_groups=group_num)
        self.gate_threshold = gate_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.gn(x)
        gamma = self.gn.weight / torch.sum(self.gn.weight)
        gamma = gamma.view(1, -1, 1, 1)
        reweights = torch.sigmoid(normalized * gamma)
        strength = torch.where(reweights >= self.gate_threshold, torch.ones_like(reweights), reweights)
        weak = torch.where(reweights < self.gate_threshold, torch.zeros_like(reweights), reweights)
        strong_part = strength * x
        weak_part = weak * x
        phase_strong, dfs_strong = torch.split(strong_part, strong_part.shape[1] // 2, dim=1)
        phase_weak, dfs_weak = torch.split(weak_part, weak_part.shape[1] // 2, dim=1)
        return torch.cat([phase_strong + dfs_weak, phase_weak + dfs_strong], dim=1)


class ProxyContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, classifier: nn.Linear) -> torch.Tensor:
        proxies = F.normalize(classifier.weight, p=2, dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarities = embeddings @ proxies.T / self.temperature
        labels = labels.long()
        mask = F.one_hot(labels, num_classes=proxies.shape[0]).float()
        similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
        log_prob = similarities - torch.log(torch.exp(similarities).sum(dim=1, keepdim=True))
        return -((mask * log_prob).sum(dim=1) / mask.sum(dim=1)).mean()


class WiCBRNet(nn.Module):
    def __init__(
        self,
        *,
        num_labels: int = 6,
        pretrained: bool = True,
        group_num: int = 4,
        gate_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.phase_gate = SpatialGate()
        self.dfs_gate = SpatialGate()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        phase_backbone = resnet18(weights=weights)
        dfs_backbone = resnet18(weights=weights)
        self.phase_features = nn.Sequential(*list(phase_backbone.children())[:-2])
        self.dfs_features = nn.Sequential(*list(dfs_backbone.children())[:-2])
        self.dp_fusion = DPFusion(1024, group_num=group_num, gate_threshold=gate_threshold)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_labels)

    def forward(self, phase_inputs: torch.Tensor, dfs_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        phase_inputs, phase_attention = self.phase_gate(phase_inputs)
        dfs_inputs, dfs_attention = self.dfs_gate(dfs_inputs)
        phase_features = self.phase_features(phase_inputs)
        dfs_features = self.dfs_features(dfs_inputs)
        fused = torch.cat([phase_features, dfs_features], dim=1)
        fused = self.dp_fusion(fused)
        pooled = self.avgpool(fused).flatten(1)
        logits = self.fc(pooled)
        return {
            "logits": logits,
            "embedding": pooled,
            "phase_attention": phase_attention,
            "dfs_attention": dfs_attention,
        }
