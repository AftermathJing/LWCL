from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lwcl_v2.ablations.fstc_encoder import build_fstc_encoder_model
from lwcl_v2.config import load_config
from lwcl_v2.data.dataset import SignalV2Dataset, collate_signal_v2
from lwcl_v2.training.checkpoint import load_checkpoint
from lwcl_v2.training.trainer import Trainer, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an isolated FSTC encoder ablation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--weights", choices=("ema", "raw"), default="raw")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.manifest is not None:
        config["data"]["manifest"] = args.manifest
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    seed_everything(int(config["training"].get("seed", 2025)))
    data = config["data"]
    training = config["training"]
    dataset = SignalV2Dataset(
        manifest_path=data["manifest"],
        split=args.split,
        max_seq_len=int(data["max_seq_len"]),
        seed=int(training.get("seed", 2025)),
        min_valid_receivers=int(data.get("min_valid_receivers", 5)),
        feature_mode=str(data.get("feature_mode", "current")),
        phase_subcarriers=int(data.get("phase_subcarriers", 30)),
        amplitude_bins=int(data.get("amplitude_bins", 64)),
        frame_interval_ms=float(data.get("frame_interval_ms", 10.0)),
    )
    num_workers = int(training.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("eval_batch_size", training.get("batch_size", 96))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0 and bool(training.get("persistent_workers", False)),
        collate_fn=partial(collate_signal_v2, max_seq_len=int(data["max_seq_len"])),
    )
    model = build_fstc_encoder_model(config)
    trainer = Trainer(model, loader, loader, config, args.output_dir)
    payload = load_checkpoint(args.checkpoint, model, ema=trainer.ema, restore_rng=False)
    trainer.state.update(payload["state"])
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
        torch.cuda.synchronize(trainer.device)
    started = time.perf_counter()
    metrics = trainer.evaluate(loader, split=args.split, use_ema=args.weights == "ema")
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    elapsed = time.perf_counter() - started
    metrics.update(
        {
            "evaluation_seconds": elapsed,
            "samples_per_second": len(loader.dataset) / max(elapsed, 1e-12),
            "mean_batch_latency_ms": 1000.0 * elapsed / max(len(loader), 1),
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(trainer.device))
                if trainer.device.type == "cuda"
                else 0
            ),
            "ablation_variant": str(config["ablation"]["variant"]),
        }
    )
    filename = (
        f"{args.split}_ema_metrics.json"
        if args.weights == "ema"
        else f"{args.split}_metrics.json"
    )
    output = Path(args.output_dir) / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
