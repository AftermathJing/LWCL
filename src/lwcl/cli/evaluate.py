from __future__ import annotations

import argparse
import json
from pathlib import Path

from lwcl.cli.common import build_loaders
from lwcl.config import load_config
from lwcl.models.lwcl import LWCLModel
from lwcl.training.checkpoint import load_checkpoint
from lwcl.training.trainer import Trainer, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an LWCL checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["training"].get("seed", 2025)))
    train_loader, validation_loader, test_loader = build_loaders(config, include_test=True)
    model = LWCLModel(config)
    payload = load_checkpoint(args.checkpoint, model, restore_rng=False)
    trainer = Trainer(model, train_loader, validation_loader, config, args.output_dir)
    trainer.state.update(payload["state"])
    loader = validation_loader if args.split == "validation" else test_loader
    metrics = trainer.evaluate(loader, split=args.split)
    output_path = Path(args.output_dir) / f"{args.split}_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
