import torch

from lwcl.models.lwcl import LWCLModel


def test_end_to_end_forward_backward():
    config = {
        "data": {"num_labels": 6, "num_receivers": 6, "input_features": 49, "max_seq_len": 64},
        "model": {
            "channel_attention": {"output_dim": 32, "hidden_dim": 8, "dropout": 0.0},
            "hste": {
                "projection_dim": 32,
                "window_size": 8,
                "window_stride": 4,
                "local_heads": 4,
                "local_layers": 1,
                "output_dim": 64,
                "global_heads": 4,
                "global_layers": 1,
                "ffn_factor": 2,
                "dropout": 0.0,
            },
            "adapter": {"num_layers": 1, "num_heads": 4, "ffn_factor": 2, "dropout": 0.0},
            "backbone": {"type": "tiny", "hidden_size": 64, "num_layers": 1, "num_heads": 4},
        },
    }
    model = LWCLModel(config)
    features = torch.randn(2, 32, 6, 49)
    mask = torch.ones(2, 32, dtype=torch.bool)
    labels = torch.tensor([1, 4])
    outputs = model(features, attention_mask=mask, labels=labels)
    assert outputs["logits"].shape == (2, 6)
    assert torch.isfinite(outputs["loss"])
    outputs["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_hste_classifier_forward_backward_without_adapter_or_backbone():
    config = {
        "data": {"num_labels": 6, "num_receivers": 6, "input_features": 49, "max_seq_len": 64},
        "model": {
            "architecture": "hste_classifier",
            "channel_attention": {"output_dim": 32, "hidden_dim": 8, "dropout": 0.0},
            "hste": {
                "projection_dim": 32,
                "window_size": 4,
                "window_stride": 2,
                "local_heads": 4,
                "local_layers": 1,
                "output_dim": 64,
                "global_heads": 4,
                "global_layers": 1,
                "ffn_factor": 2,
                "dropout": 0.0,
            },
            "classifier": {
                "hidden_dim": 64,
                "pooling": "attention",
                "dropout": 0.0,
                "label_smoothing": 0.0,
            },
        },
    }
    model = LWCLModel(config)
    assert model.adapter is None
    assert model.backbone is None
    features = torch.randn(3, 13, 6, 49)
    mask = torch.ones(3, 13, dtype=torch.bool)
    labels = torch.tensor([0, 2, 5])
    outputs = model(features, attention_mask=mask, labels=labels)
    assert outputs["logits"].shape == (3, 6)
    assert torch.isfinite(outputs["loss"])
    outputs["loss"].backward()
    assert model.classifier.classifier[-1].weight.grad is not None
