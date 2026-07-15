from __future__ import annotations

import argparse
import json
from pathlib import Path

from lwcl_v2.config import load_config
from lwcl_v2.data.dataset import build_loaders
from lwcl_v2.models import SignalV2Classifier
from lwcl_v2.training.checkpoint import load_checkpoint
from lwcl_v2.training.trainer import Trainer, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an LWCL-v2 checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["training"].get("seed", 2025)))
    train_loader, validation_loader, test_loader = build_loaders(config, include_test=True)
    model = SignalV2Classifier(config)
    trainer = Trainer(model, train_loader, validation_loader, config, args.output_dir)
    payload = load_checkpoint(args.checkpoint, model, ema=trainer.ema, restore_rng=False)
    trainer.state.update(payload["state"])
    loader = validation_loader if args.split == "validation" else test_loader
    metrics = trainer.evaluate(loader, split=args.split)
    output = Path(args.output_dir) / f"{args.split}_metrics.json"
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
