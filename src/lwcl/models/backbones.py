from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .rotary import RotaryEncoder


class TinySignalBackbone(nn.Module):
    """Small offline backbone used to prove the complete training path before Qwen runs."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_factor: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = RotaryEncoder(
            num_layers, hidden_size, num_heads, hidden_size * ffn_factor, dropout
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_labels))

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        encoded = self.encoder(inputs_embeds, position_ids, attention_mask)
        weights = attention_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits, "embeddings": pooled}


class HSTEClassificationHead(nn.Module):
    """Masked pooling and classification directly on HSTE outputs."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int | None = None,
        pooling: str = "attention",
        dropout: float = 0.2,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError(f"Unsupported classifier pooling: {pooling}")
        self.hidden_size = input_dim
        self.pooling = pooling
        self.label_smoothing = label_smoothing
        self.pool_score = nn.Linear(input_dim, 1) if pooling == "attention" else None
        hidden_dim = int(hidden_dim or input_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        encoded: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        mask = attention_mask.to(torch.bool)
        if self.pool_score is None:
            weights = mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            scores = self.pool_score(encoded).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            weights = torch.where(mask, weights, torch.zeros_like(weights))
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
        loss = (
            F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            if labels is not None
            else None
        )
        return {"loss": loss, "logits": logits, "embeddings": pooled}


class HuggingFaceSignalBackbone(nn.Module):
    """Qwen/LLM sequence-classification backend using continuous inputs_embeds and LoRA."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: list[str] | None = None,
        modules_to_save: list[str] | None = None,
        torch_dtype: str = "bfloat16",
        device_map: str | dict[str, Any] | None = None,
        trust_remote_code: bool = True,
        train_norm_layers: bool = True,
    ) -> None:
        super().__init__()
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoConfig, AutoModelForSequenceClassification
        except ImportError as exc:
            raise RuntimeError("Install the project with the 'llm' extra to use the Hugging Face backend") from exc

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        config.num_labels = num_labels
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = getattr(config, "eos_token_id", None)
        self.hidden_size = int(getattr(config, "hidden_size"))
        dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"
        load_kwargs: dict[str, Any] = {
            "config": config,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype,
        }
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **load_kwargs)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=modules_to_save or ["score"],
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_config)
        if train_norm_layers:
            for name, parameter in self.model.named_parameters():
                if "norm" in name.lower():
                    parameter.requires_grad = True

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return {"loss": outputs.loss, "logits": outputs.logits, "embeddings": None}


def build_backbone(config: dict[str, Any], num_labels: int) -> nn.Module:
    backend_type = config.get("type", "tiny")
    if backend_type == "tiny":
        return TinySignalBackbone(
            hidden_size=int(config.get("hidden_size", 256)),
            num_labels=num_labels,
            num_layers=int(config.get("num_layers", 2)),
            num_heads=int(config.get("num_heads", 8)),
            ffn_factor=int(config.get("ffn_factor", 4)),
            dropout=float(config.get("dropout", 0.1)),
        )
    if backend_type in {"huggingface", "qwen"}:
        return HuggingFaceSignalBackbone(
            model_name_or_path=config["model_name_or_path"],
            num_labels=num_labels,
            lora_rank=int(config.get("lora_rank", 8)),
            lora_alpha=int(config.get("lora_alpha", 16)),
            lora_dropout=float(config.get("lora_dropout", 0.1)),
            target_modules=config.get("target_modules"),
            modules_to_save=config.get("modules_to_save"),
            torch_dtype=config.get("torch_dtype", "bfloat16"),
            device_map=config.get("device_map"),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            train_norm_layers=bool(config.get("train_norm_layers", True)),
        )
    raise ValueError(f"Unsupported backbone type: {backend_type}")
