from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lwcl_v2.ablations.fstc_encoder import build_fstc_encoder_model, fstc_ablation_spec
from lwcl_v2.ablations.test_selection_trainer import TestAccuracySelectionTrainer
from lwcl_v2.config import config_hash, load_config, save_resolved_config
from lwcl_v2.data.dataset import SignalV2Dataset, build_loaders, collate_signal_v2
from lwcl_v2.data.sampler import CrossSubjectBatchSampler
from lwcl_v2.training import Trainer, seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, encoding="utf-8", stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _build_test_selection_loaders(config: dict):
    data = config["data"]
    training = config["training"]
    common = {
        "manifest_path": data["manifest"],
        "max_seq_len": int(data["max_seq_len"]),
        "seed": int(training.get("seed", 2025)),
        "min_valid_receivers": int(data.get("min_valid_receivers", 5)),
        "feature_mode": str(data.get("feature_mode", "current")),
        "phase_subcarriers": int(data.get("phase_subcarriers", 30)),
        "amplitude_bins": int(data.get("amplitude_bins", 64)),
        "frame_interval_ms": float(data.get("frame_interval_ms", 10.0)),
    }
    train_dataset = SignalV2Dataset(
        split="train",
        augmentation=config.get("augmentation"),
        **common,
    )
    test_dataset = SignalV2Dataset(split="test", **common)
    collate = partial(collate_signal_v2, max_seq_len=int(data["max_seq_len"]))
    num_workers = int(training.get("num_workers", 2))
    persistent_workers = num_workers > 0 and bool(training.get("persistent_workers", False))
    sampler_config = training.get("subject_balanced_sampler", {})
    batch_sampler = CrossSubjectBatchSampler(
        train_dataset.labels,
        train_dataset.subjects,
        gestures_per_batch=int(sampler_config.get("gestures_per_batch", data["num_labels"])),
        subjects_per_gesture=int(sampler_config.get("subjects_per_gesture", 4)),
        samples_per_subject=int(sampler_config.get("samples_per_subject", 4)),
        seed=int(training.get("seed", 2025)),
        batches_per_epoch=sampler_config.get("batches_per_epoch"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(training.get("eval_batch_size", training.get("batch_size", 96))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        collate_fn=collate,
    )
    return train_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train isolated Widar3 FSTC encoder ablations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--debug-fail-after-step", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.manifest is not None:
        config["data"]["manifest"] = args.manifest
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.debug_fail_after_step is not None:
        config["training"]["debug_fail_after_step"] = args.debug_fail_after_step

    seed_everything(int(config["training"].get("seed", 2025)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    manifest = Path(config["data"]["manifest"])
    variant = str(config["ablation"]["variant"])
    metadata = {
        "git_commit": os.environ.get("LWCL_CODE_COMMIT") or _git("rev-parse", "HEAD"),
        "git_branch": os.environ.get("LWCL_CODE_BRANCH") or _git("branch", "--show-current"),
        "git_status": _git("status", "--porcelain"),
        "config_hash": config_hash(config),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "seed": int(config["training"].get("seed", 2025)),
        "python": platform.python_version(),
        "ablation": fstc_ablation_spec(variant),
        "implementation": "lwcl_v2.ablations.fstc_encoder",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    selection_split = str(config["training"].get("selection_split", "validation"))
    if selection_split not in {"validation", "test"}:
        raise ValueError(
            "The isolated FSTC encoder ablation protocol requires "
            "training.selection_split to be validation or test"
        )
    if selection_split == "test":
        selection = config["training"].get("selection", {})
        if str(selection.get("metric")) != "accuracy":
            raise ValueError("Test-selection ablations require training.selection.metric=accuracy")
        train_loader, validation_loader = _build_test_selection_loaders(config)
        trainer_class = TestAccuracySelectionTrainer
    else:
        train_loader, validation_loader = build_loaders(config)
        trainer_class = Trainer
    model = build_fstc_encoder_model(config)
    report = model.parameter_report()
    report["ablation"] = fstc_ablation_spec(variant)
    (output_dir / "parameter_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    trainer = trainer_class(model, train_loader, validation_loader, config, output_dir)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
        torch.cuda.synchronize(trainer.device)
    started = time.perf_counter()
    state = trainer.fit(args.resume_from)
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    elapsed = time.perf_counter() - started
    runtime = {
        "train_seconds": elapsed,
        "global_steps": int(state["global_step"]),
        "seconds_per_step": elapsed / max(int(state["global_step"]), 1),
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(trainer.device))
            if trainer.device.type == "cuda"
            else 0
        ),
    }
    (output_dir / "runtime_report.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )
    print(json.dumps(state))


if __name__ == "__main__":
    main()
