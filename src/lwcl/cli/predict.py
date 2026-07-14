from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lwcl.config import load_config
from lwcl.models.lwcl import LWCLModel
from lwcl.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one processed LWCL .npz sample")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = LWCLModel(config)
    load_checkpoint(args.checkpoint, model, restore_rng=False)
    model.to(device).eval()
    with np.load(Path(args.sample), allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32)
    max_length = int(config["data"]["max_seq_len"])
    if features.shape[0] > max_length:
        start = (features.shape[0] - max_length) // 2
        features = features[start : start + max_length]
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    mask = torch.ones(1, tensor.shape[1], dtype=torch.bool, device=device)
    with torch.inference_mode():
        outputs = model(tensor, attention_mask=mask)
        probabilities = torch.softmax(outputs["logits"], dim=-1)[0].cpu().tolist()
    result = {
        "sample": str(Path(args.sample).resolve()),
        "predicted_label": int(np.argmax(probabilities)),
        "probabilities": probabilities,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
