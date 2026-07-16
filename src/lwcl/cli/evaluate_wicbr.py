from __future__ import annotations

import argparse
import json
from pathlib import Path

from lwcl.config import load_config
from lwcl.wicbr.model import WiCBRNet
from lwcl.wicbr.trainer import WiCBRTrainer
from lwcl.cli.train_wicbr import build_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Wi-CBR checkpoint on validation or test data")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--manifest")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.manifest is not None:
        config["data"]["manifest"] = args.manifest
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader, test_loader = build_loaders(config, include_test=True)
    model = WiCBRNet(
        num_labels=int(config["data"]["num_labels"]),
        pretrained=bool(config["model"].get("pretrained", True)),
        group_num=int(config["model"].get("group_num", 4)),
        gate_threshold=float(config["model"].get("gate_threshold", 0.5)),
    )
    trainer = WiCBRTrainer(model, train_loader, validation_loader, config, output_dir)
    trainer.resume(args.checkpoint)
    loader = validation_loader if args.split == "validation" else test_loader
    metrics = trainer.evaluate(loader, split=args.split)
    (output_dir / f"{args.split}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
